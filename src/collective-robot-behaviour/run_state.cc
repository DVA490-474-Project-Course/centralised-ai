//==============================================================================
// Author: Jacob Johansson, Viktor Eriksson
// Creation date: 2024-11-6
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Source file for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#include "run_state.h"
#include "communication.h"
#include "reward.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor
RunState::ComputeRewards(const torch::Tensor& kStates,
                         struct RewardConfiguration reward_configuration) {

  torch::Tensor positions = torch::zeros({2, amount_of_players_in_team});
  positions[0][0] = kStates[3];
  positions[1][0] = kStates[4];
  positions[0][1] = kStates[5];
  positions[1][1] = kStates[6];
  positions[0][2] = kStates[7];
  positions[1][2] = kStates[8];
  positions[0][3] = kStates[9];
  positions[1][3] = kStates[10];
  positions[0][4] = kStates[11];
  positions[1][4] = kStates[12];
  positions[0][5] = kStates[13];
  positions[1][5] = kStates[14];

  torch::Tensor orientations = torch::zeros(amount_of_players_in_team);
  orientations[0] = kStates[15];
  orientations[1] = kStates[16];
  orientations[2] = kStates[17];
  orientations[3] = kStates[18];
  orientations[4] = kStates[19];
  orientations[5] = kStates[20];

  /* Distance to ball */
  torch::Tensor ball_position = torch::empty({2, 1});
  ball_position[0] = kStates[1];
  ball_position[1] = kStates[2];
  torch::Tensor distance_to_ball_reward = ComputeDistanceToBallReward(
      positions, ball_position, reward_configuration.distance_to_ball_reward);
  torch::Tensor angle_to_ball_reward =
      ComputeAngleToBallReward(orientations, positions, ball_position);
  torch::Tensor total_reward = distance_to_ball_reward + angle_to_ball_reward;

  return total_reward;
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */