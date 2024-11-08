//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-11-61
// Last modified: 2024-11-06 by Jacob Johansson
// Description: Source file for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#include "world.h"
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
        torch::Tensor positions = torch::empty({2, 6});
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

        torch::Tensor average_distance_reward = ComputeAverageDistanceReward(positions, reward_configuration.max_distance_from_center, reward_configuration.average_distance_reward);
        torch::Tensor have_ball = states.slice(0, 28, 34);
        torch::Tensor have_ball_reward = ComputeHaveBallReward(have_ball, reward_configuration.have_ball_reward);
        

        /* Distance to ball */
        torch::Tensor ball_position = torch::empty({2, 1});
        ball_position[0] = states[1];
        ball_position[1] = states[2];
        torch::Tensor distance_to_ball_reward = ComputeDistanceToBallReward(positions, ball_position, reward_configuration.distance_to_ball_reward);

        torch::Tensor total_reward = average_distance_reward + have_ball_reward + distance_to_ball_reward;

        std::cout << "Total reward: " << distance_to_ball_reward << std::endl;
        return distance_to_ball_reward;
    }
}
}