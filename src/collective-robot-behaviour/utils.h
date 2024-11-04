//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-08
// Last modified: 2024-11-04 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <stdint.h>

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    /*!
    @returns the reward-to-go values with the shape [num_time_steps, 1].
    @param[In] rewards: The accumulated reward for each time step, with the shape [num_time_steps, 1].
    @param[In] discount: Discount factor.
    */
    torch::Tensor  ComputeRewardToGo(const torch::Tensor & rewards, double discount);

    /*!
    @returns the normalized reward-to-go values with the shape [num_time_steps, 1].
    @param[In] reward_to_go: Unnormalized reward-to-go values, with the shape [num_time_steps, 1].
    */
    torch::Tensor NormalizeRewardToGo(const torch::Tensor & reward_to_go);

    /*!
    @returns the temporal differences with the shape [num_time_steps, num_agents].
    @param[In] critic_values: Values from the critic network per time step with the shape [num_time_steps, 1].
    @param[In] rewards: Reward per time step per actor with the shape [num_time_steps, num_actors].
    @param[In] discount: Discount factor.
    */
    torch::Tensor  ComputeTemporalDifference(const torch::Tensor & critic_values, const torch::Tensor & rewards, double discount);

    /*!
    @returns the general advantage estimation represented by a tensor of shape [num_time_steps, num_agents].
    @param[In] temporal_differences: Tensor of shape [num_time_steps, num_agents].
    @param[In] discount: Discount factor.
    @param[In] gae_parameter: GAE parameter.
    */
    torch::Tensor  ComputeGeneralAdvantageEstimation(const torch::Tensor & temporal_differences, double discount, double gae_parameter);

    /*!
    @returns the probability ratio for all agents for each time step with the shape [num_time_steps, num_agents].
    @param[In] current_probabilities: Probability of choosing the action for each agent for each time step with current policy, with shape [num_time_steps, num_agents].
    @param[In] previous_probabilities: Probability of choosing the same action for each agent for each time step with previous policy, with shape [num_time_steps, num_agents].
    */
    torch::Tensor  ComputeProbabilityRatio(const torch::Tensor & current_probabilities, const torch::Tensor & previous_probabilities);

    /*!
    @returns the probability ratio clipped depending on the clip_value with the shape [num_time_steps, num_agents].
    @param[In] probability_ratio: Probability ratio for each agent for each time step of shape [num_time_steps, num_agents].
    @param[In] clip_value: The parameter used to clip the probability ratio.
    */
    torch::Tensor  ClipProbabilityRatio(const torch::Tensor & probability_ratio, float clip_value);

    /*!
    @returns the policy loss over a number of time steps.
    @param[In] GAE for each agent for each time step with the shape [num_time_steps, num_agents].
    @param[In] probability_ratio: Probability ratio for each agent for each time step with the shape [num_time_steps, num_agents].
    @param[In] clip_value: The parameter used to clip the probability ratio.
    @param[In] policy_entropy: Average Policy entropy over the time steps and agents.
    @param[In] entropy_coefficient: The parameter used to determine the weight of the entropies.
    */
    torch::Tensor ComputePolicyLoss(const torch::Tensor & general_advantage_estimation, const torch::Tensor & probability_ratio, float clip_value, const torch::Tensor & policy_entropy);

    /*!
    @returns the critic loss over a number of time steps.
    @param[In] current_values: Values from the Critic network with current parameters for each agent and time step, with shape [num_time_steps, num_agents].
    @param[In] previous_values: Values from the Critic network with previous parameters for each agent and time step, with shape [num_time_steps, num_agents].
    @param[In] reward_to_go: The discounted reward-to-go values for each time step, with shape [num_time_steps, 1].
    @param[In] clip_value: The parameter used to clip the critic network values.
    */
    torch::Tensor ComputeCriticLoss(const torch::Tensor & current_values, const torch::Tensor & previous_values, const torch::Tensor & reward_to_go, float clip_value);

    /*!
    @returns the policy entropy.
    @param[In] actions_probabilities: Probabilities of choosing actions for each agent and time step, with the shape [num_time_steps, num_agents, num_actions].
    @param[In] entropy_coefficient: The parameter used to determine the weight of the entropy.
    */
    torch::Tensor ComputePolicyEntropy(const torch::Tensor & actions_probabilities, float entropy_coefficient);
}
}

#endif

