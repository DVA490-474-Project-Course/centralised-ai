//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-11-06 by Jacob Johansson
// Description: Headers for world.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <torch/torch.h>
#include "communication.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{
/*!
* @brief Base class for all game states which are responsible for calculating the legal actions and reward per agent for the given state.
*/
class GameStateBase
{
  public:
    /* Virtual destructor intended for this polymorphic base class.*/
    virtual ~GameStateBase() = default;

  /*!
  * @brief Calculates the action masks for the given state.
  * @returns A tensor representing the action mask for the given state, with the shape [num_agents, num_actions].
  */
    virtual torch::Tensor ComputeActionMasks(const torch::Tensor & states) = 0;
    /*!
    * @brief Calculates the rewards for the given state.
    * @returns A tensor representing the reward for the given state, with the shape [num_agents].
    * @param[In] states: The states of the world, with the shape [num_states].
    * @param[In] reward_configuration: The configuration of the rewards.
    */
    virtual torch::Tensor ComputeRewards(const torch::Tensor & states, struct RewardConfiguration reward_configuration) = 0;
};

/*!
  @returns a tensor representing the reward given by the average distance between all robots, with the shape [num_agents].
  @param[In] positions: A tensor of all the positions of all the robots, with the shape[2, num_agents].
  @param[In] max_distance: The maximum distance from the average position of all the robots when no reward will be given anymore. @note max_distance cannot be 0!
  @param[In] max_reward: The maximum reward that will be given when a robot is within the range [0, max_distance].
*/
torch::Tensor ComputeAverageDistanceReward(torch::Tensor & positions, float max_distance, float max_reward);

/*!
* @returns a tensor representing the reward given by the distance between the robots and the ball, with the shape [num_agents].
* @param[In] positions: A tensor of all the positions of all the robots, with the shape[2, num_agents].
* @param[In] ball_position: A tensor of the position of the ball, with the shape[2].
* @param[In] reward: The reward given when the robot is close to the ball.
*/
torch::Tensor ComputeDistanceToBallReward(torch::Tensor & positions, torch::Tensor & ball_position, float reward);

/*!
  @returns a tensor representing the reward given by when the robot either has the ball or not, with the shape [num_agents].
  @param[In] reward: The reward given when the robot has the ball.
*/
torch::Tensor ComputeHaveBallReward(torch::Tensor & have_ball_flags, float reward);


}
}

#endif
