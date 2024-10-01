//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Stores all test functions for the mlp.cc and mlp.h files.
// License: See LICENSE file for license details.
//==============================================================================

#include "../../src/collective-robot-behaviour/mlp.h"
#include <gtest/gtest.h>
#include <vector>

// Test ReLU for a single value
TEST(ReLUTest, SingleValue){
	EXPECT_EQ(relu(5.0), 5.0);
	EXPECT_EQ(relu(-3.0), 0.0);
	EXPECT_EQ(relu(0.0), 0.0);
}

// Test ReLU for a vector of values
TEST(ReLUTest, VectorValue){
	std::vector<double> input = {-3.5, 2.7, -1.2, 0.0, 5.4};
	std::vector<double> expected = {0.0, 2.7, 0.0, 0.0, 5.4};

	std::vector<double> result = relu(input);

	EXPECT_EQ(result.size(), expected.size());
	for(size_t i = 0; i < result.size(); ++i){
		EXPECT_EQ(result[i], expected[i]);
	}
}