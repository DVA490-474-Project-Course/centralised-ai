//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-10-08 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include <stdint.h>
#include "utils.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

    Tensor compute_reward_to_go(const Tensor& rewards, uint32_t num_time_steps){
        
        output = torch.zeros(num_time_steps);

        for (int32_t t = 0; t < num_time_steps; t++){
            output[t] = rewards.slice(0, t, num_time_steps).sum();
        }
        
        return output;
    }

    Tensor compute_general_advantage_estimation(const Tensor& temporal_differences, double discount, double gae_parameter){
        
        uint32_t num_time_steps = temporal_differences.size(0);

        // Precompute the factors for each time step
        Tensor factors = torch.zeros(num_time_steps);
        double discount_times_gae_parameter = discount * gae_parameter;
        for (int32_t t = 0; t < num_time_steps; t++){
            factors[i] = torch::pow(discount_times_gae_parameter, t);
        }
        
        // Compute the GAE for each time step
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

    double compute_policy_loss(const Tensor& general_advantage_estimation, const Tensor& probability_ratio, float clip_value, double policy_entropy, float entropy_coefficient){
        Tensor probability_ratio_clipped = clip_probability_ratio(probability_ratio, clip_value);
        Tensor probability_ratio_gae_product = torch::min(probability_ratio * general_advantage_estimation, probability_ratio_clipped * general_advantage_estimation);

        // Compute the loss.
        int32_t num_time_steps = general_advantage_estimation.size(0);
        int32_t num_agents = general_advantage_estimation.size(1);

        Tensor probability_ratio_gae_product_agent_sum = torch::sum(probability_ratio_gae_product, 1);
        Tensor probability_ratio_gae_product_batch_sum = torch::sum(probability_ratio_gae_product_agent_sum, 0);

        Tensor policy_entropy_agent_sum = torch::sum(policy_entropy, 1);
        Tensor policy_entropy_batch_sum = torch::sum(policy_entropy_agent_sum, 0);
        
        return (1/(num_time_steps * num_agents))*(probability_ratio_gae_product_batch_sum + entropy_coefficient * policy_entropy_batch_sum);
    }

    } /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */