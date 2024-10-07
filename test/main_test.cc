//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Main test file which initiates and runs all tests.
// License: See LICENSE file for license details.
//==============================================================================

#include "gtest/gtest.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

/* Main function for calling all tests.*/
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */