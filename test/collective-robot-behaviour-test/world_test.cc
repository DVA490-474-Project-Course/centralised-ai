//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-11-01 by Jacob Johansson
// Description: Stores all tests for the world.cc and world.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include "../../src/collective-robot-behaviour/world.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

class WorldTest : public ::testing::Test 
{
	protected:World world;

	void SetUp() override 
	{
		world.field.width = 100.0;
		world.field.height = 50.0;

		world.ball.position_x = 10.0;
		world.ball.position_y = 20.0;
		world.ball.velocity_x = 5.0;
		world.ball.velocity_y = 3.0;

		world.state = GameState::kPlaying;
	}
};

TEST_F(WorldTest, InitialWorldState)
{
	EXPECT_EQ(world.field.width, 100.0);
	EXPECT_EQ(world.field.height, 50.0);
	EXPECT_EQ(world.ball.position_x, 10.0);
	EXPECT_EQ(world.ball.position_y, 20.0);
	EXPECT_EQ(world.ball.velocity_x, 5.0);
	EXPECT_EQ(world.ball.velocity_y, 3.0);
	EXPECT_EQ(world.state, GameState::kPlaying);
}

TEST_F(WorldTest, AddRobot)
{
	Robot robot;
	robot.position_x = 30.0;
	robot.position_y = 40.0;
	robot.velocity_x = 2.0;
	robot.velocity_y = -1.0;
	robot.orientation = 0.785;

	world.robots.push_back(robot);

	ASSERT_EQ(world.robots.size(), 1);
	EXPECT_FLOAT_EQ(world.robots[0].position_x, 30.0);
	EXPECT_FLOAT_EQ(world.robots[0].position_y, 40.0);
	EXPECT_FLOAT_EQ(world.robots[0].velocity_x, 2.0);
	EXPECT_FLOAT_EQ(world.robots[0].velocity_y, -1.0);
	EXPECT_FLOAT_EQ(world.robots[0].orientation, 0.785);
}

TEST_F(WorldTest, ChangeGameState)
{
	world.state = GameState::kHalted;
	EXPECT_EQ(world.state, GameState::kHalted);
}

TEST_F(WorldTest, BallMovement)
{
	world.ball.position_x += world.ball.velocity_x;
	world.ball.position_y += world.ball.velocity_y;

	EXPECT_EQ(world.ball.position_x, 15.0);
	EXPECT_EQ(world.ball.position_y, 23.0);
}

TEST(ComputeAverageDistanceReward, Test_1)
{
	torch::Tensor positions = torch::zeros({2, 6});
	float max_distance = 1;
	float max_reward = -0.001;

	torch::Tensor output = ComputeAverageDistanceReward(positions, max_distance, max_reward);

	EXPECT_EQ(output.size(0), 6);
	EXPECT_FLOAT_EQ(output[0].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[1].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[2].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[3].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[4].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[5].item<float>(), -0.001);
}

TEST(ComputeAverageDistanceReward, Test_2)
{
	torch::Tensor positions = torch::ones({2, 6});
	float max_distance = 1;
	float max_reward = -0.001;

	torch::Tensor output = ComputeAverageDistanceReward(positions, max_distance, max_reward);

	EXPECT_EQ(output.size(0), 6);
	EXPECT_FLOAT_EQ(output[0].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[1].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[2].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[3].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[4].item<float>(), -0.001);
	EXPECT_FLOAT_EQ(output[5].item<float>(), -0.001);
}

TEST(ComputeAverageDistanceReward, Test_3)
{
	torch::Tensor positions = torch::ones({2, 6});
	positions[0][0] = 1; positions[1][0] = 0;
	positions[0][1] = 3; positions[1][1] = 1;
	positions[0][2] = 5; positions[1][2] = 2;
	positions[0][3] = 7; positions[1][3] = 1;
	positions[0][4] = 2; positions[1][4] = 1;
	positions[0][5] = 4; positions[1][5] = 3;

	float max_distance = 2;
	float max_reward = -0.001;

	torch::Tensor output = ComputeAverageDistanceReward(positions, max_distance, max_reward);

	EXPECT_EQ(output.size(0), 6);
	EXPECT_FLOAT_EQ(output[0].item<float>(), -0.001 * 0);
	EXPECT_FLOAT_EQ(output[1].item<float>(), -0.001 * 0.8611111111);
	EXPECT_FLOAT_EQ(output[2].item<float>(), -0.001 * 0.4444444444);
	EXPECT_FLOAT_EQ(output[3].item<float>(), -0.001 * 0);
	EXPECT_FLOAT_EQ(output[4].item<float>(), -0.001 * 0.2777777778);
	EXPECT_FLOAT_EQ(output[5].item<float>(), -0.001 * 0.2777777778);
}

TEST(ComputeHaveBallReward, Test_1)
{
	torch::Tensor have_ball_flags = torch::zeros(6);
	float reward = 1;

	torch::Tensor output = ComputeHaveBallReward(have_ball_flags, reward);

	EXPECT_EQ(output.size(0), 6);
	EXPECT_EQ(have_ball_flags.size(0), 6);
	EXPECT_FLOAT_EQ(output[0].item<float>(), 0);
	EXPECT_FLOAT_EQ(output[1].item<float>(), 0);
	EXPECT_FLOAT_EQ(output[2].item<float>(), 0);
	EXPECT_FLOAT_EQ(output[3].item<float>(), 0);
	EXPECT_FLOAT_EQ(output[4].item<float>(), 0);
	EXPECT_FLOAT_EQ(output[5].item<float>(), 0);
}

TEST(ComputeHaveBallReward, Test_2)
{
	torch::Tensor have_ball_flags = torch::ones(6);
	float reward = 1;

	torch::Tensor output = ComputeHaveBallReward(have_ball_flags, reward);

	EXPECT_EQ(output.size(0), 6);
	EXPECT_EQ(have_ball_flags.size(0), 6);
	EXPECT_FLOAT_EQ(output[0].item<float>(), 1);
	EXPECT_FLOAT_EQ(output[1].item<float>(), 1);
	EXPECT_FLOAT_EQ(output[2].item<float>(), 1);
	EXPECT_FLOAT_EQ(output[3].item<float>(), 1);
	EXPECT_FLOAT_EQ(output[4].item<float>(), 1);
	EXPECT_FLOAT_EQ(output[5].item<float>(), 1);
}

}
}