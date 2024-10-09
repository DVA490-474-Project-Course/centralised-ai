//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-08
// Last modified: 2024-10-09 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <stdint.h>

namespace centralised_ai{
namespace collective_robot_behaviour{

    /* Returns the reward-to-go values, where
        - discounts: The discount for each time step in a [1, num_time_steps] tensor, i.e. discount¹, discount², discount³, ...
        - rewards: The accumulated reward for each time step in a tensor of shape [num_time_steps, 1].
        - num_time_steps: The number of time steps to compute the reward-to-go values over.
    */
    Tensor compute_reward_to_go(const Tensor& rewards, uint32_t num_time_steps);

    /* Returns the general advantage estimation represented by a tensor of shape [num_time_steps, num_agents], where
        - temporal_differences: Tensor of shape [num_time_steps, num_agents]
        - discount: Discount factor.
        - gae_parameter: GAE parameter.
    */
    Tensor compute_general_advantage_estimation(const Tensor& temporal_differences, double discount, double gae_parameter);

    /* Returns the probability ratio for all agents for each time step with the shape [num_time_steps, num_agents], where
        - current_probabilities: Probability of choosing the action for each agent for each time step with current policy, with shape [num_time_steps, num_agents]
        - previous_probabilities: Probability of choosing the same action for each agent for each time step with previous policy, with shape [num_time_steps, num_agents]
    */
    Tensor compute_probability_ratio(const Tensor& current_probabilities, const Tensor& previous_probabilities);

    /* Returns the probability ratio clipped depending on the clip_value, where
        - probability_ratio: Probability ratio for each agent for each time step of shape [num_time_steps, num_agents]
        - clip_value: The parameter used to clip the probability ratio
    */
    Tensor clip_probability_ratio(const Tensor& probability_ratio, float clip_value);

    /* Returns the policy loss over a number of time steps, where
        - general_advantage_estimation: GAE for each agent for each time step with the shape [num_time_steps, num_agents]
        - probability_ratio: Probability ratio for each agent for each time step with the shape [num_time_steps, num_agents]
        - clip_value: The parameter used to clip the probability ratio
        - policy_entropy: Policy entropy for each agent for each time step with the shape [num_time_steps, num_agents]
        - entropy_coefficient: The parameter used to determine the weight of the entropies
    */
    double compute_policy_loss(const Tensor& general_advantage_estimation, const Tensor& probability_ratio, float clip_value, double policy_entropy, float entropy_coefficient);

    /* Returns the critic loss over a number of time steps, where
        - current_values: Values from the Critic network with current parameters for each agent and time step, with shape [num_time_steps, num_agents]
        - previous_values: Values from the Critic network with previous parameters for each agent and time step, with shape [num_time_steps, num_agents]
        - reward_to_go: The discounted reward-to-go values for each time step, with shape [num_time_steps, 0]
        - clip_value: The parameter used to clip the critic network values
    */
    double compute_critic_loss(const Tensor& current_values, const Tensor& previous_values, const Tensor& reward_to_go, float clip_value);
} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */


#endif /* UTILS_H */

