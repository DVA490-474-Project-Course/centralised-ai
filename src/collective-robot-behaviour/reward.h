/* reward.h
 * ==============================================================================
 * Author: Jacob Johansson, Viktor Eriksson
 * Creation date: 2024-10-01
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Headers for reward.cc.
 * License: See LICENSE file for license details.
 * ==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_REWARD_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_REWARD_H_

#include "torch/torch.h"

namespace centralised_ai {
namespace collective_robot_behaviour {

/*!
 * @brief Computes the reward given by the distance between all the robots and
 * the center position of all the kPositions of the robots.
 * @returns A tensor representing the reward given by the average distance
 * between all robots, with the shape [num_agents].
 * @param[in] kPositions: A tensor of all the kPositions of all the robots, with
 * the shape[2, num_agents].
 * @param[in] max_distance: The maximum distance from the average position of
 * all the robots when no reward will be given anymore. @note max_distance
 * cannot be 0!
 * @param[in] max_reward: The maximum reward that will be given when a robot is
 * within the range [0, max_distance].
 */
torch::Tensor ComputeAverageDistanceReward(torch::Tensor& kPositions,
                                           float max_distance,
                                           float max_reward);

/*!
 * @brief Computes the reward given by the distance between the robots and the
 * ball.
 * @returns A tensor representing the reward given by the distance between the
 * robots and the ball, with the shape [num_agents].
 * @param[in] kPositions: A tensor of all the kPositions of all the robots, with
 * the shape[2, num_agents].
 * @param[in] kBallPosition: A tensor of the position of the ball, with the
 * shape[2].
 * @param[in] reward: The reward given when the robot is close to the ball.
 */
torch::Tensor ComputeDistanceToBallReward(torch::Tensor& kPositions,
                                          torch::Tensor& kBallPosition,
                                          float reward);

/*!
 * @brief Computes the reward given by whether the robots have the ball or not.
 * @returns A tensor representing the reward given by when the robot either has
 * the ball or not, with the shape [num_agents].
 * @param[in] reward: The reward given when the robot has the ball.
 */
torch::Tensor ComputeHaveBallReward(torch::Tensor& have_ball_flags,
                                    float reward);

/*!
 * @brief Computes the reward given by the angle between the robots and the
 * ball, where the angle is in range [-1, 1] (-1 when the robot is looking away
 * from the ball, 1 when the robot is looking towards the ball).
 * @returns a tensor representing the reward given by the angle between the
 * robots and the ball, with the shape [num_agents].
 * @param[in] kOrientations: A tensor of all the kOrientations of all the
 * robots, with the shape [num_agents].
 * @param[in] kPositions: A tensor of all the kPositions of all the robots, with
 * the shape [2, num_agents].
 * @param[in] kBallPosition: A tensor of the position of the ball, with the
 * shape [2, 1].
 */
torch::Tensor ComputeAngleToBallReward(const torch::Tensor& kOrientations,
                                       const torch::Tensor& kPositions,
                                       const torch::Tensor& kBallPosition);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_REWARD_H_ */