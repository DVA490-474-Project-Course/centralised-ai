/* ==============================================================================
 * Author: Jacob Johansson, Viktor Eriksson
 * Creation date: 2024-11-6
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Header for run_state.cc.
 * License: See LICENSE file for license details.
 * ==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_RUNSTATE_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_RUNSTATE_H_

#include "../../src/collective-robot-behaviour/communication.h"
#include "../../src/collective-robot-behaviour/game_state_base.h"
#include "../../src/collective-robot-behaviour/reward.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Class that represents the run state which is responsible for
 * calculating the legal actions and reward per agent when running in "normal"
 * play state.
 *
 * @note Copyable, moveable.
 */
class RunState : public GameStateBase
{

 public:
  /*!
   * @brief Calculates the rewards for the run state.
   * @returns A tensor representing the rewards for the run state, with the
   * shape [num_agents].
   * @param[in] states: The states of the world, with the shape
   * [num_global_states].
   * @param[in] reward_configuration: The configuration of the rewards.
   */
  torch::Tensor
  ComputeRewards(const torch::Tensor& kStates,
                 struct RewardConfiguration reward_configuration) override;

  /*!
   * @brief Constructor for the run state.
   */
  RunState() = default;
};

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_RUNSTATE_H_ */