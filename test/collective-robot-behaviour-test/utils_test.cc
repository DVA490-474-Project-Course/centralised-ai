//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-10-29 by Jacob Johansson
// Description: Stores all tests for the utils.cc and utils.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/utils.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

TEST(ComputeRewardToGoTest, Test_1)
{
  torch::Tensor input = torch::ones(6);

  torch::Tensor output = ComputeRewardToGo(input, 1);
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

  torch::Tensor output = ComputeRewardToGo(input, 1);
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

  torch::Tensor output = ComputeRewardToGo(input, 2);
  EXPECT_EQ(output[0].item<float>(), 258);
  EXPECT_EQ(output[1].item<float>(), 258);
  EXPECT_EQ(output[2].item<float>(), 256);
  EXPECT_EQ(output[3].item<float>(), 248);
  EXPECT_EQ(output[4].item<float>(), 224);
  EXPECT_EQ(output[5].item<float>(), 160);
}

TEST(ComputeTemporalDifferenceTest, Test_1)
{
  torch::Tensor critic_values = torch::zeros(4);
  torch::Tensor rewards = torch::zeros({6, 4});
  double discount = 0;

  torch::Tensor output = ComputeTemporalDifference(critic_values, rewards, discount);

  EXPECT_EQ(output.size(0), 6);
  EXPECT_EQ(output.size(1), 4);
  EXPECT_FLOAT_EQ(output[0][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][3].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[1][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[1][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[1][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[1][3].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[2][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[2][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[2][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[2][3].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[3][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[3][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[3][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[3][3].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[4][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[4][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[4][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[4][3].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[5][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[5][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[5][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[5][3].item<float>(), 0);
}

TEST(ComputeTemporalDifferenceTest, Test_2)
{
  torch::Tensor critic_values = torch::zeros(4);
  torch::Tensor rewards = torch::ones({6, 4});
  double discount = 0.1;

  critic_values[0] = 0.1;
  critic_values[1] = 0.2;
  critic_values[2] = 0.3;
  critic_values[3] = 0.4;

  torch::Tensor output = ComputeTemporalDifference(critic_values, rewards, discount);

  EXPECT_EQ(output.size(0), 6);
  EXPECT_EQ(output.size(1), 4);
  EXPECT_NEAR(output[0][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[0][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[0][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[0][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[0][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[0][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[1][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[1][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[1][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[1][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[1][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[1][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[2][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[2][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[2][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[2][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[2][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[2][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[3][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[3][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[3][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[3][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[3][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[3][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[4][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[4][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[4][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[4][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[4][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[4][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[5][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[5][3].item<float>(), 0.60, 0.00001);
  EXPECT_NEAR(output[5][0].item<float>(), 0.92, 0.00001);
  EXPECT_NEAR(output[5][1].item<float>(), 0.83, 0.00001);
  EXPECT_NEAR(output[5][2].item<float>(), 0.74, 0.00001);
  EXPECT_NEAR(output[5][3].item<float>(), 0.60, 0.00001);
  
}

TEST(ComputeGAETest, Test_1)
{
  torch::Tensor temporaldiffs = torch::zeros({1, 4});
  double discount = 0.1;
  double gae_parameter = 0.1;

  torch::Tensor output = ComputeGeneralAdvantageEstimation(temporaldiffs, discount, gae_parameter);

  EXPECT_EQ(temporaldiffs.size(0), 1);
  EXPECT_EQ(temporaldiffs.size(1), 4);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 4);
  EXPECT_FLOAT_EQ(output[0][0].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][1].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][2].item<float>(), 0);
  EXPECT_FLOAT_EQ(output[0][3].item<float>(), 0);
}

TEST(ComputeGAETest, Test_2)
{
  torch::Tensor temporaldiffs = torch::ones({1, 3});

  double discount = 0.1;
  double gae_parameter = 0.1;

  torch::Tensor output = ComputeGeneralAdvantageEstimation(temporaldiffs, discount, gae_parameter);

  EXPECT_EQ(temporaldiffs.size(0), 1);
  EXPECT_EQ(temporaldiffs.size(1), 3);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 3);
  EXPECT_FLOAT_EQ(output[0][0].item<float>(), 1.0101);
  EXPECT_FLOAT_EQ(output[0][1].item<float>(), 1.01);
  EXPECT_FLOAT_EQ(output[0][2].item<float>(), 1);
}

TEST(ComputeGAETest, Test_3)
{
  torch::Tensor temporaldiffs = torch::ones({1, 3});
  temporaldiffs[0][0] = 0.1;
  temporaldiffs[0][1] = 0.2;
  temporaldiffs[0][2] = 0.3;

  double discount = 0.1;
  double gae_parameter = 0.1;

  torch::Tensor output = ComputeGeneralAdvantageEstimation(temporaldiffs, discount, gae_parameter);

  EXPECT_EQ(temporaldiffs.size(0), 1);
  EXPECT_EQ(temporaldiffs.size(1), 3);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 3);
  EXPECT_NEAR(output[0][0].item<float>(), 0.10203, 0.00001);
  EXPECT_NEAR(output[0][1].item<float>(), 0.203, 0.00001);
  EXPECT_NEAR(output[0][2].item<float>(), 0.3, 0.00001);
}

TEST(ComputeProbabilityRatio, Test_1)
{
  torch::Tensor currrentprobs = torch::ones({1, 6, 2});
  torch::Tensor previousprobs = torch::ones({1, 6, 2});

  torch::Tensor output = ComputeProbabilityRatio(currrentprobs, previousprobs);

  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 6);
  EXPECT_EQ(output.size(2), 2);
  EXPECT_FLOAT_EQ(output[0][0][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][1][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][2][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][3][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][4][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][5][0].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][0][1].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][1][1].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][2][1].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][3][1].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][4][1].item<float>(), 1);
  EXPECT_FLOAT_EQ(output[0][5][1].item<float>(), 1);
}

TEST(ComputeProbabilityRatio, Test_2)
{
  torch::Tensor currrentprobs = torch::ones({1, 6, 2});
  torch::Tensor previousprobs = torch::ones({1, 6, 2}) * 2;

  torch::Tensor output = ComputeProbabilityRatio(currrentprobs, previousprobs);

  EXPECT_EQ(output.size(0), 1);
  EXPECT_EQ(output.size(1), 6);
  EXPECT_EQ(output.size(2), 2);
  EXPECT_FLOAT_EQ(output[0][0][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][1][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][2][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][3][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][4][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][5][0].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][0][1].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][1][1].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][2][1].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][3][1].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][4][1].item<float>(), 0.5);
  EXPECT_FLOAT_EQ(output[0][5][1].item<float>(), 0.5);
}

TEST(ComputePolicyEntropy, Test_1)
{
  torch::Tensor actions_probabilities = torch::ones({1, 6, 2, 3});
  float entropy_coefficient = 1;

  torch::Tensor output = ComputePolicyEntropy(actions_probabilities, entropy_coefficient);

  EXPECT_EQ(actions_probabilities.size(0), 1);
  EXPECT_EQ(actions_probabilities.size(1), 6);
  EXPECT_EQ(actions_probabilities.size(2), 2);
  EXPECT_EQ(actions_probabilities.size(3), 3);
  EXPECT_NEAR(output.item<float>(), 0, 0.00001);
}

TEST(ComputePolicyEntropy, Test_2)
{
  torch::Tensor actions_probabilities = torch::ones({1, 6, 2, 3}) * 0.5;
  float entropy_coefficient = 1;

  torch::Tensor output = ComputePolicyEntropy(actions_probabilities, entropy_coefficient);

  EXPECT_EQ(actions_probabilities.size(0), 1);
  EXPECT_EQ(actions_probabilities.size(1), 6);
  EXPECT_EQ(actions_probabilities.size(2), 2);
  EXPECT_EQ(actions_probabilities.size(3), 3);
  EXPECT_NEAR(output.item<float>(), 1.5, 0.00001);
}

TEST(ComputePolicyEntropy, Test_3)
{
  torch::Tensor actions_probabilities = torch::ones({1, 6, 2, 3}) * 0.25;
  float entropy_coefficient = 0.1;

  torch::Tensor output = ComputePolicyEntropy(actions_probabilities, entropy_coefficient);

  EXPECT_EQ(actions_probabilities.size(0), 1);
  EXPECT_EQ(actions_probabilities.size(1), 6);
  EXPECT_EQ(actions_probabilities.size(2), 2);
  EXPECT_EQ(actions_probabilities.size(3), 3);
  EXPECT_NEAR(output.item<float>(), 0.15, 0.00001);
}

TEST(ComputePolicyLoss, Test_1)
{
  torch::Tensor gae = torch::ones({1, 6, 2});
  torch::Tensor probability_ratios = torch::ones({1, 6, 2});
  float clip_value = 1;
  torch::Tensor entropy = torch::zeros(1);

  torch::Tensor output = ComputePolicyLoss(gae, probability_ratios, clip_value, entropy);

  EXPECT_EQ(gae.size(0), 1);
  EXPECT_EQ(gae.size(1), 6);
  EXPECT_EQ(gae.size(2), 2);
  EXPECT_EQ(probability_ratios.size(0), 1);
  EXPECT_EQ(probability_ratios.size(1), 6);
  EXPECT_EQ(probability_ratios.size(2), 2);
  EXPECT_EQ(entropy.size(0), 1);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 1);
}

TEST(ComputePolicyLoss, Test_2)
{
  torch::Tensor gae = torch::ones({1, 6, 2});
  torch::Tensor probability_ratios = torch::ones({1, 6, 2}) * 0.5;
  float clip_value = 0;
  torch::Tensor entropy = torch::zeros(1);

  torch::Tensor output = ComputePolicyLoss(gae, probability_ratios, clip_value, entropy);

  EXPECT_EQ(gae.size(0), 1);
  EXPECT_EQ(gae.size(1), 6);
  EXPECT_EQ(gae.size(2), 2);
  EXPECT_EQ(probability_ratios.size(0), 1);
  EXPECT_EQ(probability_ratios.size(1), 6);
  EXPECT_EQ(probability_ratios.size(2), 2);
  EXPECT_EQ(entropy.size(0), 1);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 0.5);
}

TEST(ComputePolicyLoss, Test_3)
{
  torch::Tensor gae = torch::ones({1, 6, 2});
  torch::Tensor probability_ratios = torch::ones({1, 6, 2}) * 0.5;
  float clip_value = 1;
  torch::Tensor entropy = torch::ones(1);

  torch::Tensor output = ComputePolicyLoss(gae, probability_ratios, clip_value, entropy);

  EXPECT_EQ(gae.size(0), 1);
  EXPECT_EQ(gae.size(1), 6);
  EXPECT_EQ(gae.size(2), 2);
  EXPECT_EQ(probability_ratios.size(0), 1);
  EXPECT_EQ(probability_ratios.size(1), 6);
  EXPECT_EQ(probability_ratios.size(2), 2);
  EXPECT_EQ(entropy.size(0), 1);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 1.5);
}

TEST(ComputePolicyLoss, Test_4)
{
  torch::Tensor gae = torch::ones({1, 6, 2});
  gae[0][0][0] = 1;
  gae[0][1][0] = 2;
  gae[0][2][0] = 3;
  gae[0][3][0] = 4;
  gae[0][4][0] = 5;
  gae[0][5][0] = 6;
  gae[0][0][1] = 1;
  gae[0][1][1] = 2;
  gae[0][2][1] = 3;
  gae[0][3][1] = 4;
  gae[0][4][1] = 5;
  gae[0][5][1] = 6;
  
  torch::Tensor probability_ratios = torch::ones({1, 6, 2}) * 0.25;
  float clip_value = 0.2;
  torch::Tensor entropy = torch::ones(1) * 0.1;

  torch::Tensor output = ComputePolicyLoss(gae, probability_ratios, clip_value, entropy);

  EXPECT_EQ(gae.size(0), 1);
  EXPECT_EQ(gae.size(1), 6);
  EXPECT_EQ(gae.size(2), 2);
  EXPECT_EQ(probability_ratios.size(0), 1);
  EXPECT_EQ(probability_ratios.size(1), 6);
  EXPECT_EQ(probability_ratios.size(2), 2);
  EXPECT_EQ(entropy.size(0), 1);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 0.975);
}

TEST(ComputeCriticLoss, Test_1)
{
  torch::Tensor current_values = torch::ones({1, 2});
  torch::Tensor previous_values = torch::zeros({1, 2});
  torch::Tensor rewards_to_go = torch::ones({1, 6, 2}) * 0.1;
  float clip_value = 0;

  torch::Tensor output = ComputeCriticLoss(current_values, previous_values, rewards_to_go, clip_value);

  EXPECT_EQ(current_values.size(0), 1);
  EXPECT_EQ(current_values.size(1), 2);
  EXPECT_EQ(previous_values.size(0), 1);
  EXPECT_EQ(previous_values.size(1), 2);
  EXPECT_EQ(rewards_to_go.size(0), 1);
  EXPECT_EQ(rewards_to_go.size(1), 6);
  EXPECT_EQ(rewards_to_go.size(2), 2);
  EXPECT_FLOAT_EQ(output.item<float>(), 0.405);
}

TEST(ComputeCriticLoss, Test_2)
{
  torch::Tensor current_values = torch::ones({1, 2});
  torch::Tensor previous_values = torch::zeros({1, 2});
  torch::Tensor rewards_to_go = torch::ones({1, 6, 2}) * 0.1;
  float clip_value = 0.2;

  torch::Tensor output = ComputeCriticLoss(current_values, previous_values, rewards_to_go, clip_value);

  EXPECT_EQ(current_values.size(0), 1);
  EXPECT_EQ(current_values.size(1), 2);
  EXPECT_EQ(previous_values.size(0), 1);
  EXPECT_EQ(previous_values.size(1), 2);
  EXPECT_EQ(rewards_to_go.size(0), 1);
  EXPECT_EQ(rewards_to_go.size(1), 6);
  EXPECT_EQ(rewards_to_go.size(2), 2);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 0.405);
}

TEST(ComputeCriticLoss, Test_3)
{
  torch::Tensor current_values = torch::ones({1, 2}) * 0.5;
  torch::Tensor previous_values = torch::ones({1, 2}) * 0.2;
  torch::Tensor rewards_to_go = torch::ones({1, 6, 2}) * 0.1;
  float clip_value = 0.2;

  torch::Tensor output = ComputeCriticLoss(current_values, previous_values, rewards_to_go, clip_value);

  EXPECT_EQ(current_values.size(0), 1);
  EXPECT_EQ(current_values.size(1), 2);
  EXPECT_EQ(previous_values.size(0), 1);
  EXPECT_EQ(previous_values.size(1), 2);
  EXPECT_EQ(rewards_to_go.size(0), 1);
  EXPECT_EQ(rewards_to_go.size(1), 6);
  EXPECT_EQ(rewards_to_go.size(2), 2);
  EXPECT_EQ(output.size(0), 1);
  EXPECT_FLOAT_EQ(output[0].item<float>(), 0.08);
}

}
}