/* ==============================================================================
 * Author: Jacob Johansson, Viktor Eriksson
 * Creation date: 2024-10-08
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Headers for utils.h.
 * License: See LICENSE file for license details.
 * ==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_UTILS_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_UTILS_H_

#include "cmath"
#include "iostream"
#include "stdint.h"
#include "torch/nn.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Computes the reward-to-go for each time step.
 * @returns The reward-to-go values with the shape [num_time_steps, 1].
 * @param[in] rewards: The accumulated reward for each time step, with the shape
 * [num_time_steps, 1].
 *
 * @param[in] discount: Discount factor.
 */
torch::Tensor ComputeRewardToGo(const torch::Tensor& kRewards, double discount);

/*!
 * @brief Normalizes the reward-to-go values.
 * @returns The normalized reward-to-go values with the shape [num_time_steps,
 * 1].
 *
 * @param[in] reward_to_go: Unnormalized reward-to-go values, with the shape
 * [num_time_steps, 1].
 */
torch::Tensor NormalizeRewardToGo(const torch::Tensor& kRewardToGo);

/*!
 * @brief Computes the temporal difference residuals, which will be used in
 * computing the general advantage estimation.
 *
 * @returns The temporal differences with the shape [num_agents,
 * num_time_steps].
 *
 * @param[in] critic_values: Values from the critic network per time step with
 * the shape [num_time_steps].
 *
 * @param[in] rewards: Reward per time step per actor with the shape
 * [num_agents, num_time_steps].
 *
 * @param[in] discount: Discount factor.
 */
torch::Tensor ComputeTemporalDifference(const torch::Tensor& kCriticValues,
                                        const torch::Tensor& kRewards,
                                        double discount);

/*!
 * @brief Computes the general advantage estimation for each agent.
 *
 * @returns The general advantage estimation represented by a tensor of shape
 * [num_agents, num_time_steps].
 *
 * @param[in] temporal_differences: Tensor of shape [num_agents,
 * num_time_steps].
 *
 * @param[in] discount: Discount factor.
 * @param[in] gae_parameter: GAE parameter.
 */
torch::Tensor
ComputeGeneralAdvantageEstimation(const torch::Tensor& kTemporalDifferences,
                                  double discount, double gae_parameter);

/*!
 * @brief Computes the probability ratio for all agents for each time step.
 *
 * @returns The probability ratio for all agents for each time step with the
 * shape [mini_batch_size, num_agents, num_time_steps].
 *
 * @param[in] current_probabilities: Probability of choosing the action for each
 * agent for each time step with current policy, with shape [mini_batch_size,
 * num_agents, num_time_steps].
 *
 * @param[in] previous_probabilities: Probability of choosing the same action
 * for each agent for each time step with previous policy, with shape
 * [mini_batch_size, num_agents, num_time_steps].
 */
torch::Tensor
ComputeProbabilityRatio(const torch::Tensor& kCurrentProbabilities,
                        const torch::Tensor& kPreviousProbabilities);

/*!
 * @brief Computes the policy loss over the specified number of chunks and time
 * steps.
 *
 * @returns The policy loss over a number of time steps as a single tensor
 * value.
 *
 * @param[in] general_advantage_estimation for each agent for each chunk in the
 * mini batch with the shape [mini_batch_size, num_agents, num_time_steps].
 *
 * @param[in] probability_ratio: Probability ratio for each agent for each chunk
 * in the mini batch with the shape [mini_batch_size, num_agents,
 * num_time_steps].
 *
 * @param[in] clip_value: The parameter used to clip the probability ratio.
 *
 * @param[in] policy_entropy: Average Policy entropy over the time steps and
 * agents.
 *
 * @param[in] entropy_coefficient: The parameter used to determine the weight of
 * the entropies.
 */
torch::Tensor
ComputePolicyLoss(const torch::Tensor& kGeneralAdvantageEstimation,
                  const torch::Tensor& kProbabilityRatio, float clip_value,
                  const torch::Tensor& kPolicyEntropy);

/*!
 * @brief Computes the critic loss over the specified number of chunks and time
 * steps.
 *
 * @returns The critic loss over a number of time steps.
 *
 * @param[in] current_values: Values from the Critic network with current
 * parameters for each agent and chunk in the mini batch, with shape
 * [mini_batch_size, num_time_steps].
 *
 * @param[in] previous_values: Values from the Critic network with previous
 * parameters for each agent and chunk in the mini batch, with shape
 * [mini_batch_size, num_time_steps].
 *
 * @param[in] reward_to_go: The discounted reward-to-go values for chunk in the
 * mini batch, with shape [mini_batch_size, num_agents, num_time_steps].
 *
 * @param[in] clip_value: The parameter used to clip the critic network values.
 */
torch::Tensor ComputeCriticLoss(const torch::Tensor& kCurrentValues,
                                const torch::Tensor& kPreviousValues,
                                const torch::Tensor& kRewardToGo,
                                float clip_value);

/*!
 * @brief Computes the policy entropy over the specified number of chunks and
 * time steps.
 *
 * @returns The policy entropy.
 *
 * @param[in] actions_probabilities: Probabilities of all the actions for each
 * agent and time step, with the shape [mini_batch_size, num_agents,
 * num_time_steps, num_actions].
 *
 * @param[in] entropy_coefficient: The parameter used to determine the weight of
 * the entropy.
 */
torch::Tensor ComputePolicyEntropy(const torch::Tensor& kActionsProbabilities,
                                   float entropy_coefficient);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_UTILS_H_ */