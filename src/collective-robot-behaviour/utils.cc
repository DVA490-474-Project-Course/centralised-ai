//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-11-01 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/modules/utils.h>
#include <stdint.h>
#include "utils.h"
#include <cmath>
#include <iostream>

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor ComputeRewardToGo(const torch::Tensor & rewards, double discount)
{

  int32_t num_time_steps = rewards.size(0);

  /* Calculate the finite-horizon undiscounted reward-to-go.*/
  torch::Tensor output = torch::zeros(num_time_steps);
        
  output[num_time_steps - 1] = pow(discount, num_time_steps-1) * rewards[num_time_steps - 1];
  for (int32_t t = num_time_steps - 2; t >= 0; t--)
	{
    output[t] = pow(discount, t) * rewards[t] + output[t + 1];
	}

	return output;
}

torch::Tensor NormalizeRewardToGo(const torch::Tensor & reward_to_go)
{

	torch::Tensor output = reward_to_go.clone();

	/* Normalize the discounted reward-to-go. */
  torch::Tensor  mean = output.mean();
	torch::Tensor  std  = output.std();

	/* Handle zero standard deviation. */
	if (torch::allclose(std, torch::tensor(0.0f), 1e-5))
	{
		return torch::zeros_like(output);
	}

	return (output - mean) / std;
}

torch::Tensor  ComputeTemporalDifference(const torch::Tensor & critic_values, const torch::Tensor & rewards, double discount)
{
	int32_t num_agents = rewards.size(0);
	int32_t num_time_steps = rewards.size(1);

	torch::Tensor temporal_difference = torch::zeros({num_agents, num_time_steps});
	for (int32_t j = 0; j < num_agents; j++)
	{
		for (int32_t t = 0; t < num_time_steps - 1; t++)
		{
			temporal_difference[j][t] = rewards[j][t] + discount * critic_values[t + 1] - critic_values[t];
		}

		/* Handle the last time step (without discounting future value). */
		temporal_difference[j][num_time_steps - 1] = rewards[j][num_time_steps - 1] - critic_values[num_time_steps - 1];
	}

	return temporal_difference;
}

torch::Tensor  ComputeGeneralAdvantageEstimation(const torch::Tensor & temporal_differences, double discount, double gae_parameter)
{
	int32_t num_agents = temporal_differences.size(0);
	int32_t num_time_steps = temporal_differences.size(1);

	torch::Tensor gae = torch::zeros({num_agents, num_time_steps});
	for (int32_t j = 0; j < num_agents; j++)
	{
		for(int32_t t = 0; t < num_time_steps; t++)
		{
			for(int32_t m = 0; m <= num_time_steps - t - 1; m++)
			{
				gae[j][t] += pow(discount * gae_parameter, m) * temporal_differences[j][t + m];
			}
		}
	}

	return gae;
}

torch::Tensor  ComputeProbabilityRatio(const torch::Tensor & current_probabilities, const torch::Tensor & previous_probabilities)
{
	return current_probabilities.divide(previous_probabilities);
}

torch::Tensor ComputePolicyLoss(const torch::Tensor & general_advantage_estimation, const torch::Tensor & probability_ratio, float clip_value, const torch::Tensor & policy_entropy)
{
	/* Clip the probability ratio. */
	torch::Tensor  probability_ratio_clipped = probability_ratio.clamp(1 - clip_value, 1 + clip_value);

	int32_t mini_batch_size = general_advantage_estimation.size(0);
	int32_t num_agents = general_advantage_estimation.size(1);
	int32_t num_time_steps = general_advantage_estimation.size(2);

	torch::Tensor loss = torch::zeros(1);

	/* Calculate the loss. */
	for (int32_t t = 0; t < num_time_steps; t++)
	{
		for(int32_t j = 0; j < num_agents; j++)
		{
			for(int32_t i = 0; i < mini_batch_size; i++)
			{
				loss += torch::min(probability_ratio[i][j][t] * general_advantage_estimation[i][j][t], probability_ratio_clipped[i][j][t] * general_advantage_estimation[i][j][t]);
			}
		}
	}

	return loss.div(mini_batch_size * num_agents * num_time_steps) + policy_entropy;
}

torch::Tensor ComputeCriticLoss(const torch::Tensor & current_values, const torch::Tensor & previous_values, const torch::Tensor & reward_to_go, float clip_value)
{

	/* Get the shape of the tensors. */
	int32_t num_mini_batches = current_values.size(0);
	int32_t num_time_steps = current_values.size(1);
	int32_t num_agents = reward_to_go.size(1);

	/* Clip the current values. */
	torch::Tensor  clipping_min = previous_values - clip_value;
	torch::Tensor  clipping_max = previous_values + clip_value;
	torch::Tensor  current_values_clipped = torch::clamp(current_values, clipping_min, clipping_max);

	torch::Tensor loss = torch::zeros(1);

	/* Calculate the loss. */
	for (int32_t i = 0; i < num_mini_batches; i++)
	{
		for (int32_t j = 0; j < num_agents; j++)
		{
			for (int32_t t = 0; t < num_time_steps; t++)
			{
				torch::Tensor current_values_loss = torch::huber_loss(current_values[i][t], reward_to_go[i][j][t], at::Reduction::None, 10);
				torch::Tensor current_values_clipped_loss = torch::huber_loss(current_values_clipped[i][t], reward_to_go[i][j][t], at::Reduction::None, 10);
				loss += torch::max(current_values_loss, current_values_clipped_loss);
			}
		}
	}

	return loss.div(num_mini_batches * num_agents * num_time_steps);
}

torch::Tensor ComputePolicyEntropy(const torch::Tensor & actions_probabilities, float entropy_coefficient)
{
	/* Ensure that the action probabilities are in the range (0, 1.0] in order to avoid log(0). */
	torch::Tensor  clipped_probabilities = torch::clamp(actions_probabilities, 1e-10, 1.0);

	int32_t num_mini_batch = actions_probabilities.size(0);
	int32_t num_agents = actions_probabilities.size(1);
	int32_t num_time_steps = actions_probabilities.size(2);
	int32_t num_actions = actions_probabilities.size(3);

	torch::Tensor entropy = torch::zeros(1);

	for (int32_t i = 0; i < num_mini_batch; i++)
	{
		/* Compute the entropy over all the time steps for each agent. */
		for (int32_t k = 0; k < num_agents; k++)
		{
			for (int32_t t = 0; t < num_time_steps; t++)
			{
				torch::Tensor probabitilies = actions_probabilities[i][k][t];

				entropy += -torch::sum(probabitilies.log2().mul(probabitilies));
			}
		}
	}

	/* Calculate the average entropy over the chunks. */
	return entropy_coefficient * entropy.div(num_mini_batch * num_agents * num_time_steps);
}

}
}