//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Stores all tests for the world.cc and world.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include "../../src/collective-robot-behaviour/world.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

// Test Fixture for the World structure
class WorldTest : public ::testing::Test {
protected:
    World world;

    void SetUp() override {
        // Set up initial conditions for each test
        world.field.width = 100.0;
        world.field.height = 50.0;

        // Initialize the ball
        world.ball.position_x = 10.0;
        world.ball.position_y = 20.0;
        world.ball.velocity_x = 5.0;
        world.ball.velocity_y = 3.0;

        // Set game state
        world.state = GameState::kPlaying;
    }
};

// Test to check initial world state
TEST_F(WorldTest, InitialWorldState) {
    EXPECT_EQ(world.field.width, 100.0);
    EXPECT_EQ(world.field.height, 50.0);
    EXPECT_EQ(world.ball.position_x, 10.0);
    EXPECT_EQ(world.ball.position_y, 20.0);
    EXPECT_EQ(world.ball.velocity_x, 5.0);
    EXPECT_EQ(world.ball.velocity_y, 3.0);
    EXPECT_EQ(world.state, GameState::kPlaying);
}

// Test adding a robot to the world
TEST_F(WorldTest, AddRobot) {
    Robot robot;
    robot.position_x = 30.0;
    robot.position_y = 40.0;
    robot.velocity_x = 2.0;
    robot.velocity_y = -1.0;
    robot.orientation = 0.785; // 45 degrees

    world.robots.push_back(robot);

    // Check if the robot is added correctly
    ASSERT_EQ(world.robots.size(), 1);
    EXPECT_FLOAT_EQ(world.robots[0].position_x, 30.0);
    EXPECT_FLOAT_EQ(world.robots[0].position_y, 40.0);
    EXPECT_FLOAT_EQ(world.robots[0].velocity_x, 2.0);
    EXPECT_FLOAT_EQ(world.robots[0].velocity_y, -1.0);
    EXPECT_FLOAT_EQ(world.robots[0].orientation, 0.785);
}

// Test changing game state
TEST_F(WorldTest, ChangeGameState) {
    world.state = GameState::kHalted;
    EXPECT_EQ(world.state, GameState::kHalted);
}

// Test the ball's movement
TEST_F(WorldTest, BallMovement) {
    // Simulate ball movement
    world.ball.position_x += world.ball.velocity_x;
    world.ball.position_y += world.ball.velocity_y;

    EXPECT_EQ(world.ball.position_x, 15.0); // 10 + 5
    EXPECT_EQ(world.ball.position_y, 23.0); // 20 + 3
}

} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */