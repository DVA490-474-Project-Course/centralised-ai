//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-10-09 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include <stdint.h>
#include "utils.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

    Tensor compute_reward_to_go(const Tensor& rewards, float discount){

        int32_t num_time_steps = rewards.size(0);

        // Calculate the discounted reward-to-go.
        Tensor output = rewards.clone();
        for (int32_t t = num_time_steps - 2; t >= 0; t--){
            output[t] += discount * output[t + 1];
        }

        // Normalize the discounted reward-to-go.
        Tensor mean = output.mean();
        Tensor std  = output.std();

        // Handle zero standard deviation.
        if (torch::allclose(std, torch::tensor(0.0f), 1e-5)){
            return torch::zeros_like(output);
        }

        return (output - mean) / std;
    }

    Tensor compute_general_advantage_estimation(const Tensor& temporal_differences, double discount, double gae_parameter){
        
        uint32_t num_time_steps = temporal_differences.size(0);

        // Calculate the factors for each time step.
        Tensor factors = torch.zeros(num_time_steps);
        double discount_times_gae_parameter = discount * gae_parameter;
        for (int32_t t = 0; t < num_time_steps; t++){
            factors[i] = torch::pow(discount_times_gae_parameter, t);
        }
        
        // Calculate the GAE for each time step.
        Tensor output = torch::zeros_like(temporal_differences)
        for (int32_t t = 0; t < num_time_steps; t++){
            Tensor remaining_factors = factors.slize(1, 0, num_time_steps - t);
            Tensor remaining_temporal_differences = temporal_differences.slize(0, t, num_time_steps);
            output[t] = remaining_factors.matmul(remaining_temporal_differences);
        }

        return output;
    }

    Tensor compute_probability_ratio(const Tensor& current_probabilities, const Tensor& previous_probabilities){
        return current_probabilities.divide(previous_probabilities);
    }

    Tensor clip_probability_ratio(const Tensor& probability_ratio, float clip_value){
        return probability_ratio.clamp(1-clip_value, 1+clip_value);
    }

    double compute_policy_loss(const Tensor& general_advantage_estimation, const Tensor& probability_ratio, float clip_value, double policy_entropy){
        Tensor probability_ratio_clipped = clip_probability_ratio(probability_ratio, clip_value);
        Tensor probability_ratio_gae_product = torch::min(probability_ratio * general_advantage_estimation, probability_ratio_clipped * general_advantage_estimation);

        // Calculate the loss.
        int32_t num_time_steps = general_advantage_estimation.size(0);
        int32_t num_agents = general_advantage_estimation.size(1);

        Tensor probability_ratio_gae_product_agent_sum = torch::sum(probability_ratio_gae_product, 1);
        Tensor probability_ratio_gae_product_batch_sum = torch::sum(probability_ratio_gae_product_agent_sum, 0);
        
        return (1/(num_time_steps * num_agents)) * (probability_ratio_gae_product_batch_sum) + policy_entropy;
    }

    double compute_critic_loss(const Tensor& current_values, const Tensor& previous_values, const Tensor& reward_to_go, float clip_value){
        
        // Get the shape of the tensors.
        int32_t num_time_steps = current_values.size(0);
        int32_t num_agents = current_values.size(1);

        // Clip the current values.
        Tensor clipping_min = previous_values - clip_value;
        Tensor clipping_max = previous_values + clip_value;
        Tensor current_values_clipped = torch::clamp(current_values, clipping_min, clipping_max);

        // Calculate Mean Squared Error.
        Tensor reward_to_go_expanded = reward_to_go.expand({-1, num_agents});
        Tensor current_values_mse = torch::pow(current_values - reward_to_go, 2);
        Tensor current_values_clipped_mse = torch::pow(current_values_clipped - reward_to_go, 2);

        // Calculate the loss.
        Tensor max_values = torch::max(current_values_mse, current_values_clipped_mse);
        Tensor max_values_agent_sum = torch::sum(max_values, 1);
        Tensor max_values_batch_sum = torch::sum(max_values_agent_sum, 0);
        return (1/(num_time_steps * num_agents)) * max_values_batch_sum;
    }

    double compute_policy_entropy(const Tensor& actions_probabilities, float entropy_coefficient){
        // Ensure that the action probabilities are in the range (0, 1.0] in order to avoid log(0).
        Tensor clipped_probabilities = torch::clamp(actions_probabilities, 1e-10, 1.0);

        // Calculate the enropy for each time step and agent.
        Tensor entropy = -torch::sum(clipped_probabilities * clipped_probabilities.log(), 2);

        // Calculate the average entropy over the time steps and agents.
        int32_t num_time_steps = actions_probabilities.size(0);
        int32_t num_agents = actions_probabilities.size(1);
        return (1/(num_time_steps * num_agents)) * entropy.sum() * entropy_coefficient;
    }

    } /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */