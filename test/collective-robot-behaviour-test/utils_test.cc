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

  TEST(ComputeRewardToGoTest, SingleValue){
  EXPECT_EQ(relu(5.0), 5.0);
  EXPECT_EQ(relu(-3.0), 0.0);
  EXPECT_EQ(relu(0.0), 0.0);
}


} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */