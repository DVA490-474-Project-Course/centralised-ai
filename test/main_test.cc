// main_test.cc
//==============================================================================
// Author: Aaiza Aziz Khan
// Creation date: 2024-10-03
// Last modified: 2024-10-03 by Aaiza Aziz Khan
// Description: Main test file which initiates and runs all tests.
// License: See LICENSE file for license details.
//==============================================================================


// Other .h files
#include "gtest/gtest.h"


// Main
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
