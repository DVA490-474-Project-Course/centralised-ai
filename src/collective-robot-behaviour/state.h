//==============================================================================
// Author: Jacob Johansson, Viktor Eriksson
// Creation date: 2024-10-01
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Header for the base of the game states.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_STATE_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_STATE_H_

#include "../../src/collective-robot-behaviour/communication.h"
#include "../common_types.h"
#include "torch/torch.h"
#include "vector"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Base class for all game states which are responsible for calculating
 * the legal actions and reward per agent for the given state.
 */
class GameStateBase {
public:
  /* Virtual destructor intended for this polymorphic base class.*/
  virtual ~GameStateBase() = default;

  /*!
   * @brief Calculates the action masks for the given state.
   * @returns A tensor representing the action mask for the given state, with
   * the shape [num_agents, num_actions].
   */
  virtual torch::Tensor ComputeActionMasks(const torch::Tensor& states) {
    return torch::ones({num_actions, amount_of_players_in_team});
  }

  /*!
   * @brief Calculates the rewards for the given state.
   * @returns A tensor representing the reward for the given state, with the
   * shape [num_agents].
   * @param[in] states: The states of the world, with the shape [num_states].
   * @param[in] reward_configuration: The configuration of the rewards.
   */
  virtual torch::Tensor
  ComputeRewards(const torch::Tensor& states,
                 struct RewardConfiguration reward_configuration) = 0;
};

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_STATE_H_ */