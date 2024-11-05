//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-11-05 by Jacob Johansson
// Description: Stores all tests for the communication.cc and communication.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/communication.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

TEST(ComputeRewardsTest, TestShape)
{
    torch::Tensor states = torch::ones(40);
    RewardConfiguration reward_configuration = {1, 1, 1};
    Team own_team = Team::kBlue;

    torch::Tensor output = ComputeRewards(states, reward_configuration, own_team);

    EXPECT_EQ(output.size(0), 6);
}

}
}