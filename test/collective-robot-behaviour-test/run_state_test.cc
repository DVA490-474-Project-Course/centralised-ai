//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-11-06 by Jacob Johansson
// Description: Stores all tests for the run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/run_state.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

TEST(RunStateTest, ShapeTest)
{
    RunState state;

    torch::Tensor action_masks = state.ComputeActionMasks(torch::randn({4, 4}));
    torch::Tensor rewards = state.ComputeRewards(torch::randn({4, 4}));

    EXPECT_EQ(action_masks.size(0), 6);
    EXPECT_EQ(action_masks.size(1), 10);
    EXPECT_EQ(rewards.size(0), 6);
}

}
}