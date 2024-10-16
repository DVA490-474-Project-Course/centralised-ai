//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-08
// Last modified: 2024-10-16 by Jacob Johansson
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
        - rewards: The accumulated reward for each time step, with the shape [num_time_steps, 1]
        - discount: The discount factor
    */
    torch::Tensor  compute_reward_to_go(const torch::Tensor & rewards, float discount);

    /* Returns the temporal differences with the shape [num_time_steps, num_agents], where
        - critic_values: Values from the critic network per time step with the shape [num_time_steps, 1]
        - rewards: Reward per time step per actor with the shape [num_time_steps, num_actors]
        - discount: Discount factor
    */
    torch::Tensor  compute_temporal_difference(const torch::Tensor & critic_values, const torch::Tensor & rewards, double discount);

    /* Returns the general advantage estimation represented by a tensor of shape [num_time_steps, num_agents], where
        - temporal_differences: Tensor of shape [num_time_steps, num_agents]
        - discount: Discount factor.
        - gae_parameter: GAE parameter.
    */
    torch::Tensor  compute_general_advantage_estimation(const torch::Tensor & temporal_differences, double discount, double gae_parameter);

    /* Returns the probability ratio for all agents for each time step with the shape [num_time_steps, num_agents], where
        - current_probabilities: Probability of choosing the action for each agent for each time step with current policy, with shape [num_time_steps, num_agents]
        - previous_probabilities: Probability of choosing the same action for each agent for each time step with previous policy, with shape [num_time_steps, num_agents]
    */
    torch::Tensor  compute_probability_ratio(const torch::Tensor & current_probabilities, const torch::Tensor & previous_probabilities);

    /* Returns the probability ratio clipped depending on the clip_value, where
        - probability_ratio: Probability ratio for each agent for each time step of shape [num_time_steps, num_agents]
        - clip_value: The parameter used to clip the probability ratio
    */
    torch::Tensor  clip_probability_ratio(const torch::Tensor & probability_ratio, float clip_value);

    /* Returns the policy loss over a number of time steps, where
        - general_advantage_estimation: GAE for each agent for each time step with the shape [num_time_steps, num_agents]
        - probability_ratio: Probability ratio for each agent for each time step with the shape [num_time_steps, num_agents]
        - clip_value: The parameter used to clip the probability ratio
        - policy_entropy: Average Policy entropy over the time steps and agents
        - entropy_coefficient: The parameter used to determine the weight of the entropies
    */
    torch::Tensor compute_policy_loss(const torch::Tensor & general_advantage_estimation, const torch::Tensor & probability_ratio, float clip_value, const torch::Tensor & policy_entropy);

    /* Returns the critic loss over a number of time steps, where
        - current_values: Values from the Critic network with current parameters for each agent and time step, with shape [num_time_steps, num_agents]
        - previous_values: Values from the Critic network with previous parameters for each agent and time step, with shape [num_time_steps, num_agents]
        - reward_to_go: The discounted reward-to-go values for each time step, with shape [num_time_steps, 1]
        - clip_value: The parameter used to clip the critic network values
    */
    torch::Tensor compute_critic_loss(const torch::Tensor & current_values, const torch::Tensor & previous_values, const torch::Tensor & reward_to_go, float clip_value);

    /* Returns the policy entropy, where
        - actions_probabilities: Probabilities of choosing actions for each agent and time step, with the shape [num_time_steps, num_agents, num_actions]
        - entropy_coefficient: The parameter used to determine the weight of the entropy
    */
    torch::Tensor compute_policy_entropy(const torch::Tensor & actions_probabilities, float entropy_coefficient);
} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */


#endif /* UTILS_H */

