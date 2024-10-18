//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-10-16 by Jacob Johansson
// Description: Stores all tests for the utils.cc and utils.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <torch/torch.h>

namespace centralised_ai{
namespace collective_robot_behaviour{

  TEST(ComputeRewardToGoTest, Test_1)
  {
        torch::Tensor input = torch::ones(6, 1);

        torch::Tensor output = compute_reward_to_go(input, 1);
        for(int32_t i = 0; i < 6; i++)
        {
            EXPECT_FLOAT_EQ(output[i], 1);
        }
  }

  TEST(ComputeRewardToGoTest, Test_2)
  {
    torch::Tensor input = torch::zeros(6, 1);

    torch::Tensor output = compute_reward_to_go(input, 1);
    for (int32_t i = 0; i < 6; i++)
    {
        EXPECT_FLOAT_EQ(output[i], 0);
    }
  }

  TEST(ComputeRewardToGoTest, Test_3)
  {
    torch::Tensor input = torch::zeros(6, 1);
    for(int32_t i = 0; i < 6; i++)
    {
        input[i] = i;
    }

    torch::Tensor output = compute_reward_to_go(input, 2);
    for(int32_t i = 0; i < 6; i++)
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 10);
    EXPECT_EQ(output[2], 34);
    EXPECT_EQ(output[3], 98);
    EXPECT_EQ(output[4], 258);
    EXPECT_EQ(output[5], 642);
  }


} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */