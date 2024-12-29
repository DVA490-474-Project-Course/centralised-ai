/* ==============================================================================
 * Author: Jacob Johansson, Viktor Eriksson
 * Creation date: 2024-10-07
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Source functions for utils.cc.
 * License: See LICENSE file for license details.
 * ==============================================================================
 */

#include "utils.h"
#include "cmath"
#include "iostream"
#include "stdint.h"
#include "torch/nn.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor ComputeRewardToGo(const torch::Tensor& kRewards,
                                double discount) {
  int32_t num_time_steps = kRewards.size(0);

  /* Calculate the finite-horizon undiscounted reward-to-go. */
  torch::Tensor output = torch::zeros(num_time_steps);

  output[num_time_steps - 1] =
      pow(discount, num_time_steps - 1) * kRewards[num_time_steps - 1];
  for (int32_t t = num_time_steps - 2; t >= 0; t--) {
    output[t] = pow(discount, t) * kRewards[t] + output[t + 1];
  }

  return output;
}

torch::Tensor NormalizeRewardToGo(const torch::Tensor& kRewardToGo) {

  torch::Tensor output = kRewardToGo.clone();

  /* Normalize the discounted reward-to-go. */
  torch::Tensor mean = output.mean();
  torch::Tensor std = output.std();

  /* Handle zero standard deviation. */
  if (torch::allclose(std, torch::tensor(0.0f), 1e-5)) {
    return torch::zeros_like(output);
  }

  return (output - mean) / std;
}

torch::Tensor ComputeTemporalDifference(const torch::Tensor& kCriticValues,
                                        const torch::Tensor& kRewards,
                                        double discount) {
  int32_t num_agents = kRewards.size(0);
  int32_t num_time_steps = kRewards.size(1);

  /* Calculate the temporal differences from t=0 up to t=T-2 */
  torch::Tensor temporal_difference =
      torch::zeros({num_agents, num_time_steps});
  for (int32_t j = 0; j < num_agents; j++) {
    for (int32_t t = 0; t < num_time_steps - 1; t++) {
      temporal_difference[j][t] =
          kRewards[j][t] + discount * kCriticValues[t + 1] - kCriticValues[t];
    }

    /* Handle the last time step (without discounting future value). */
    temporal_difference[j][num_time_steps - 1] =
        kRewards[j][num_time_steps - 1] - kCriticValues[num_time_steps - 1];
  }

  return temporal_difference;
}

torch::Tensor
ComputeGeneralAdvantageEstimation(const torch::Tensor& kTemporalDifferences,
                                  double discount, double gae_parameter) {
  int32_t num_agents = kTemporalDifferences.size(0);
  int32_t num_time_steps = kTemporalDifferences.size(1);

  /* Calculate the Generalized Advantage Estimation (GAE) for each time step and
   * agent. */
  torch::Tensor gae = torch::zeros({num_agents, num_time_steps});
  for (int32_t j = 0; j < num_agents; j++) {
    for (int32_t t = 0; t < num_time_steps; t++) {
      for (int32_t m = 0; m <= num_time_steps - t - 1; m++) {
        gae[j][t] +=
            pow(discount * gae_parameter, m) * kTemporalDifferences[j][t + m];
      }
    }
  }

  return gae;
}

torch::Tensor
ComputeProbabilityRatio(const torch::Tensor& kCurrentProbabilities,
                        const torch::Tensor& kPreviousProbabilities) {
  return kCurrentProbabilities.divide(kPreviousProbabilities);
}

torch::Tensor
ComputePolicyLoss(const torch::Tensor& kGeneralAdvantageEstimation,
                  const torch::Tensor& kProbabilityRatio, float clip_value,
                  const torch::Tensor& kPolicyEntropy) {

  /* Clip the probability ratio. */
  torch::Tensor probability_ratio_clipped =
      kProbabilityRatio.clamp(1 - clip_value, 1 + clip_value);

  int32_t mini_batch_size = kGeneralAdvantageEstimation.size(0);
  int32_t num_agents = kGeneralAdvantageEstimation.size(1);
  int32_t num_time_steps = kGeneralAdvantageEstimation.size(2);

  torch::Tensor loss = torch::zeros(1);

  /* Calculate the policy loss. */
  for (int32_t t = 0; t < num_time_steps; t++) {
    for (int32_t j = 0; j < num_agents; j++) {
      for (int32_t i = 0; i < mini_batch_size; i++) {
        loss += torch::min(kProbabilityRatio[i][j][t] *
                               kGeneralAdvantageEstimation[i][j][t],
                           probability_ratio_clipped[i][j][t] *
                               kGeneralAdvantageEstimation[i][j][t]);
      }
    }
  }

  return loss.div(mini_batch_size * num_agents * num_time_steps) +
         kPolicyEntropy;
}

torch::Tensor ComputeCriticLoss(const torch::Tensor& kCurrentValues,
                                const torch::Tensor& kPreviousValues,
                                const torch::Tensor& kRewardToGo,
                                float clip_value) {
  /* Get the shape of the tensors. */
  int32_t num_mini_batches = kCurrentValues.size(0);
  int32_t num_time_steps = kCurrentValues.size(1);
  int32_t num_agents = kRewardToGo.size(1);

  /* Clip the current values. */
  torch::Tensor clipping_min = kPreviousValues - clip_value;
  torch::Tensor clipping_max = kPreviousValues + clip_value;
  torch::Tensor current_values_clipped =
      torch::clamp(kCurrentValues, clipping_min, clipping_max);

  torch::Tensor loss = torch::zeros(1);

  /* Calculate the loss. */
  for (int32_t i = 0; i < num_mini_batches; i++) {
    for (int32_t j = 0; j < num_agents; j++) {
      for (int32_t t = 0; t < num_time_steps; t++) {
        torch::Tensor current_values_loss =
            torch::huber_loss(kCurrentValues[i][t], kRewardToGo[i][j][t],
                              at::Reduction::None, 10);
        torch::Tensor current_values_clipped_loss =
            torch::huber_loss(current_values_clipped[i][t],
                              kRewardToGo[i][j][t], at::Reduction::None, 10);
        loss += torch::max(current_values_loss, current_values_clipped_loss);
      }
    }
  }

  return loss.div(num_mini_batches * num_agents * num_time_steps);
}

torch::Tensor ComputePolicyEntropy(const torch::Tensor& kActionsProbabilities,
                                   float entropy_coefficient) {
  /* Ensure that the action probabilities are in the range (0, 1.0] in order to
   * avoid log(0).
   */
  torch::Tensor clipped_probabilities =
      torch::clamp(kActionsProbabilities, 1e-10, 1.0);

  int32_t num_mini_batch = kActionsProbabilities.size(0);
  int32_t num_agents = kActionsProbabilities.size(1);
  int32_t num_time_steps = kActionsProbabilities.size(2);
  int32_t num_actions = kActionsProbabilities.size(3);

  torch::Tensor entropy = torch::zeros(1);

  /* Compute the entropy over all the time steps in the chunks for each agent.
   */
  for (int32_t i = 0; i < num_mini_batch; i++) {
    for (int32_t k = 0; k < num_agents; k++) {
      for (int32_t t = 0; t < num_time_steps; t++) {
        torch::Tensor probabitilies = kActionsProbabilities[i][k][t];

        entropy += -torch::sum(probabitilies.log2().mul(probabitilies));
      }
    }
  }

  /* Calculate the average entropy over the chunks. */
  return entropy_coefficient *
         entropy.div(num_mini_batch * num_agents * num_time_steps);
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */