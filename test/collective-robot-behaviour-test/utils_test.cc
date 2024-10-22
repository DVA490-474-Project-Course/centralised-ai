//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-10-21 by Jacob Johansson
// Description: Stores all tests for the utils.cc and utils.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/utils.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

  TEST(ComputeRewardToGoTest, Test_1)
  {
        torch::Tensor input = torch::ones(6);

        torch::Tensor output = compute_reward_to_go(input, 1);
        EXPECT_EQ(input.size(0), 6);
        EXPECT_EQ(output.size(0), 6);

        EXPECT_FLOAT_EQ(output[0].item<float>(), 6);
        EXPECT_FLOAT_EQ(output[1].item<float>(), 5);
        EXPECT_FLOAT_EQ(output[2].item<float>(), 4);
        EXPECT_FLOAT_EQ(output[3].item<float>(), 3);
        EXPECT_FLOAT_EQ(output[4].item<float>(), 2);
        EXPECT_FLOAT_EQ(output[5].item<float>(), 1);
  }

  TEST(ComputeRewardToGoTest, Test_2)
  {
    torch::Tensor input = torch::zeros(6);

    torch::Tensor output = compute_reward_to_go(input, 1);
    for (int32_t i = 0; i < 6; i++)
    {
        EXPECT_FLOAT_EQ(output[i].item<float>(), 0);
    }
  }

  TEST(ComputeRewardToGoTest, Test_3)
  {
    torch::Tensor input = torch::zeros(6);
    for(int32_t i = 0; i < 6; i++)
    {
        input[i] = i;
    }

    EXPECT_EQ(input[0].item<float>(), 0);
    EXPECT_EQ(input[1].item<float>(), 1);
    EXPECT_EQ(input[2].item<float>(), 2);
    EXPECT_EQ(input[3].item<float>(), 3);
    EXPECT_EQ(input[4].item<float>(), 4);
    EXPECT_EQ(input[5].item<float>(), 5);

    torch::Tensor output = compute_reward_to_go(input, 2);
    EXPECT_EQ(output[0].item<float>(), 258);
    EXPECT_EQ(output[1].item<float>(), 258);
    EXPECT_EQ(output[2].item<float>(), 256);
    EXPECT_EQ(output[3].item<float>(), 248);
    EXPECT_EQ(output[4].item<float>(), 224);
    EXPECT_EQ(output[5].item<float>(), 160);
  }

  TEST(ComputeGAETest, Test_1)
  {
    // Arrange
    torch::Tensor temporaldiffs = torch::zeros((1, 4));
    double discount = 2;
    double gae_parameter = 2;

    // Execute
    torch::Tensor output = compute_general_advantage_estimation(temporaldiffs, discount, gae_parameter);

    // Assert
    EXPECT_EQ(temporaldiffs.size(0), 4);
    EXPECT_EQ(output.size(0), 4);
    EXPECT_FLOAT_EQ(output[0].item<float>(), 0);
    EXPECT_FLOAT_EQ(output[1].item<float>(), 0);
    EXPECT_FLOAT_EQ(output[2].item<float>(), 0);
    EXPECT_FLOAT_EQ(output[3].item<float>(), 0);
  }

  TEST(ComputeGAETest, Test_2)
  {
    // Arrange
    torch::Tensor temporaldiffs = torch::zeros((1, 4));
    temporaldiffs[0] = 1;
    temporaldiffs[1] = 2;
    temporaldiffs[2] = 3;
    temporaldiffs[3] = 4;

    double discount = 2;
    double gae_parameter = 2;

    // Execute
    torch::Tensor output = compute_general_advantage_estimation(temporaldiffs, discount, gae_parameter);

    // Assert
    EXPECT_EQ(temporaldiffs.size(0), 4);
    EXPECT_EQ(output.size(0), 4);
    EXPECT_FLOAT_EQ(output[0].item<float>(), 313);
    EXPECT_FLOAT_EQ(output[1].item<float>(), 78);
    EXPECT_FLOAT_EQ(output[2].item<float>(), 19);
    EXPECT_FLOAT_EQ(output[3].item<float>(), 4);
  }


} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */