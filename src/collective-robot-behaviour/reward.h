//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-11-06 by Jacob Johansson
// Description: Headers for reward.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef REWARD_H
#define REWARD_H

#include <torch/torch.h>

namespace centralised_ai
{
namespace collective_robot_behaviour
{

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

/*!
* @brief Computes the reward given by the angle between the robots and the ball, where the angle is in range [-1, 1] (-1 when the robot is looking away from the ball, 1 when the robot is looking towards the ball).
* @returns a tensor representing the reward given by the angle between the robots and the ball, with the shape [num_agents].
* @param[In] orientations: A tensor of all the orientations of all the robots, with the shape [num_agents].
* @param[In] positions: A tensor of all the positions of all the robots, with the shape [2, num_agents].
* @param[In] ball_position: A tensor of the position of the ball, with the shape [2, 1].
*/
torch::Tensor ComputeAngleToBallReward(const torch::Tensor & orientations, const torch::Tensor & positions, const torch::Tensor & ball_position);

}
}

#endif