//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-10-25 by Jacob Johansson
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

namespace centralised_ai{
namespace collective_robot_behaviour{

    torch::Tensor compute_reward_to_go(const torch::Tensor & rewards, double discount){

        int32_t num_time_steps = rewards.size(0);

        // Calculate the finite-horizon undiscounted reward-to-go.
        torch::Tensor output = torch::zeros(num_time_steps);
        
        output[num_time_steps - 1] = pow(discount, num_time_steps-1) * rewards[num_time_steps - 1];
        for (int32_t t = num_time_steps - 2; t >= 0; t--){
            output[t] = pow(discount, t) * rewards[t] + output[t + 1];
        }

        return output;
    }

    torch::Tensor normalize_reward_to_go(const torch::Tensor & reward_to_go){

        torch::Tensor output = reward_to_go.clone();

        // Normalize the discounted reward-to-go.
        torch::Tensor  mean = output.mean();
        torch::Tensor  std  = output.std();

        // Handle zero standard deviation.
        if (torch::allclose(std, torch::tensor(0.0f), 1e-5)){
            return torch::zeros_like(output);
        }

        return (output - mean) / std;
    }

    torch::Tensor  compute_temporal_difference(const torch::Tensor & critic_values, const torch::Tensor & rewards, double discount){
        uint32_t num_time_steps = rewards.size(0);
        uint32_t num_agents = rewards.size(1);

        torch::Tensor  critic_values_expanded = critic_values.expand({-1, num_agents});

        // Calculate the temporal differences for all but the last time step.
        torch::Tensor  output = torch::empty((num_time_steps, num_agents));
        for (uint32_t t = 0; t < num_time_steps - 1; t++){
            output[t] = rewards[t] + discount * critic_values_expanded[t + 1] - critic_values_expanded[t];
        }

        // Handle the last time step (without discounting future value).
        output[num_time_steps - 1] = rewards[num_time_steps - 1] - critic_values_expanded[num_time_steps - 1];

        return output;
    }

    torch::Tensor  compute_general_advantage_estimation(const torch::Tensor & temporal_differences, double discount, double gae_parameter){
        
        uint32_t num_time_steps = temporal_differences.size(0);
        uint32_t num_agents = temporal_differences.size(1);

        // Calculate the factors for each time step.
        torch::Tensor  factors = torch::empty(num_time_steps);
        double discount_times_gae_parameter = discount * gae_parameter;
        for (int32_t t = 0; t < num_time_steps; t++){
            factors[t] = pow(discount_times_gae_parameter, t);
        }
        
        // Calculate the GAE for each time step.
        torch::Tensor  output = torch::zeros({num_time_steps, num_agents});
        for (int32_t t = 0; t < num_time_steps; t++){
            torch::Tensor  remaining_factors = factors.slice(0, 0, num_time_steps - t);
            torch::Tensor  remaining_temporal_differences = temporal_differences.index({torch::indexing::Slice(t, num_time_steps)});
            output[t] = remaining_factors.matmul(remaining_temporal_differences);
        }

        return output;
    }

    torch::Tensor  compute_probability_ratio(const torch::Tensor & current_probabilities, const torch::Tensor & previous_probabilities){
        return current_probabilities.divide(previous_probabilities);
    }

    torch::Tensor  clip_probability_ratio(const torch::Tensor & probability_ratio, float clip_value){
        return probability_ratio.clamp(1-clip_value, 1+clip_value);
    }

    torch::Tensor compute_policy_loss(const torch::Tensor & general_advantage_estimation, const torch::Tensor & probability_ratio, float clip_value, const torch::Tensor & policy_entropy){
        torch::Tensor  probability_ratio_clipped = clip_probability_ratio(probability_ratio, clip_value);
        torch::Tensor  probability_ratio_gae_product = torch::min(probability_ratio * general_advantage_estimation, probability_ratio_clipped * general_advantage_estimation);

        // Calculate the loss.
        int32_t num_time_steps = general_advantage_estimation.size(0);
        int32_t num_agents = general_advantage_estimation.size(1);
        
        return probability_ratio_gae_product.sum().div(num_time_steps * num_agents) + policy_entropy;
    }

    torch::Tensor compute_critic_loss(const torch::Tensor & current_values, const torch::Tensor & previous_values, const torch::Tensor & reward_to_go, float clip_value){
        
        // Get the shape of the tensors.
        int32_t num_time_steps = current_values.size(0);
        int32_t num_agents = current_values.size(1);

        // Clip the current values.
        torch::Tensor  clipping_min = previous_values - clip_value;
        torch::Tensor  clipping_max = previous_values + clip_value;
        torch::Tensor  current_values_clipped = torch::clamp(current_values, clipping_min, clipping_max);

        // Calculate Mean Squared Error.
        torch::Tensor  reward_to_go_expanded = reward_to_go.expand({num_time_steps, num_agents});
        torch::Tensor  current_values_mse = torch::pow(current_values - reward_to_go_expanded, 2);

        torch::Tensor  current_values_clipped_mse = torch::pow(current_values_clipped - reward_to_go_expanded, 2);

        // Calculate the loss.
        torch::Tensor  max_values = torch::max(current_values_mse, current_values_clipped_mse);

        return max_values.sum().div(num_time_steps * num_agents).unsqueeze(0);
    }

    torch::Tensor compute_policy_entropy(const torch::Tensor & actions_probabilities, float entropy_coefficient){
        // Ensure that the action probabilities are in the range (0, 1.0] in order to avoid log(0).
        torch::Tensor  clipped_probabilities = torch::clamp(actions_probabilities, 1e-10, 1.0);

        int32_t num_time_steps = actions_probabilities.size(0);
        int32_t num_agents = actions_probabilities.size(1);
        int32_t num_actions = actions_probabilities.size(2);

        torch::Tensor entropy = torch::empty({num_time_steps, num_agents});

        for(int32_t t = 0; t < num_time_steps; t++)
        {
            for(int32_t k = 0; k < num_agents; k++)
            {
                torch::Tensor probabitilies = actions_probabilities[t][k];

                entropy[t][k] = -torch::sum(probabitilies.log());
            }
        }

        // Calculate the average entropy over the time steps and agents.
        return entropy_coefficient * entropy.sum().div(num_time_steps * num_agents);
    }

    } /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */