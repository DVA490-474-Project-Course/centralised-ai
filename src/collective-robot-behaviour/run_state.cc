//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-11-61
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Source file for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#include "reward.h"
#include "run_state.h"
#include <torch/torch.h>
#include "communication.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    torch::Tensor RunState::ComputeActionMasks(const torch::Tensor & states)
    {
        return torch::ones({6, 10});
    }

    torch::Tensor RunState::ComputeRewards(const torch::Tensor & states, struct RewardConfiguration reward_configuration)
    {
        torch::Tensor positions = torch::zeros({2, amount_of_players_in_team});
        positions[0][0] = states[3];
        positions[1][0] = states[4];
        positions[0][1] = states[5];
        positions[1][1] = states[6];
        positions[0][2] = states[7];
        positions[1][2] = states[8];
        positions[0][3] = states[9];
        positions[1][3] = states[10];
        positions[0][4] = states[11];
        positions[1][4] = states[12];
        positions[0][5] = states[13];
        positions[1][5] = states[14];
        
        torch::Tensor orientations = torch::zeros(amount_of_players_in_team);
        orientations[0] = states[15];
        orientations[1] = states[16];
        orientations[2] = states[17];
        orientations[3] = states[18];
        orientations[4] = states[19];
        orientations[5] = states[20];

        /* Distance to ball */
        torch::Tensor ball_position = torch::empty({2, 1});
        ball_position[0] = states[1];
        ball_position[1] = states[2];
        torch::Tensor distance_to_ball_reward = ComputeDistanceToBallReward(positions, ball_position, reward_configuration.distance_to_ball_reward);
        torch::Tensor angle_to_ball_reward = ComputeAngleToBallReward(orientations, positions, ball_position);
        torch::Tensor total_reward = distance_to_ball_reward + angle_to_ball_reward;

        return total_reward;
    }
}
}