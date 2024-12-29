/* reward.cc
 *==============================================================================
 * Author: Jacob Johansson
 * Creation date: 2024-10-01
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Source file for all code related to the reward functions.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include "reward.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor ComputeAngleToBallReward(const torch::Tensor& kOrientations,
                                       const torch::Tensor& kPositions,
                                       const torch::Tensor& kBallPosition) {
  torch::Tensor angles_to_ball = torch::empty(kOrientations.size(0));
  torch::Tensor distances_between_robots_and_ball = kBallPosition - kPositions;

  /* Declare the word forward vector and the vector between each robot to the
   * ball. */
  torch::Tensor world_forward = torch::zeros(2);
  torch::Tensor robot_to_ball = torch::zeros(2);

  for (int32_t i = 0; i < kOrientations.size(0); i++) {
    world_forward[0] = kOrientations[i].cos();
    world_forward[1] = kOrientations[i].sin();

    robot_to_ball[0] = distances_between_robots_and_ball[0][i];
    robot_to_ball[1] = distances_between_robots_and_ball[1][i];

    torch::Tensor robot_to_ball_normalized =
        robot_to_ball.div(robot_to_ball.norm());
    torch::Tensor ball_product = robot_to_ball_normalized.dot(world_forward);

    angles_to_ball[i] = ball_product;
  }

  return angles_to_ball;
}

torch::Tensor ComputeAverageDistanceReward(torch::Tensor& kPositions,
                                           float max_distance,
                                           float max_reward) {
  /* Calculate the average position of all the kPositions. */
  torch::Tensor average_position = kPositions.mean(1, true);

  /* [num_agents]. */
  torch::Tensor distances = (kPositions - average_position).pow(2).sum(0);

  torch::Tensor rewards = (-1 / pow(max_distance, 2)) * distances + 1;

  return torch::clamp(rewards, 0, 1) * max_reward;
}

torch::Tensor ComputeDistanceToBallReward(torch::Tensor& positions,
                                          torch::Tensor& kBallPosition,
                                          float reward) {
  /* [num_agents]. */
  torch::Tensor distances = (positions - kBallPosition).pow(2).sum(0);

  return -torch::sqrt(distances) * reward;
}

torch::Tensor ComputeHaveBallReward(torch::Tensor& have_ball_flags,
                                    float reward) {
  torch::Tensor rewards = torch::zeros(have_ball_flags.size(0));

  for (int32_t i = 0; i < have_ball_flags.size(0); i++) {
    if (have_ball_flags[i].item<int>() > 0) {
      rewards[i] += reward;
    } else {
      rewards[i] -= reward;
    }
  }

  return rewards;
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */