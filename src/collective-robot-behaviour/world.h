//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-29 by Jacob Johansson
// Description: Headers for world.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <torch/torch.h>

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
  @brief Width of the playing field, measured in meters.
*/
const int kFieldWidth = 9;
/*!
  @brief Height of the playing field, measured in meters.
*/
const int kFieldHeight = 6;

/*!
torch::Tensor ce area, measured in meters.
*/
const int kDefenseAreaHeight = 1;

/*!
  @brief Distance from the opponent's goal center from which a team executes a penalty kick against the opponent goal, measured in meters.
*/
const int kPenaltyMarkDistance = 6;

/*!
  @brief Width of the goal, measured in meters.
*/
const int kGoalWidth = 1;

/*!
  @brief Depth of the goal, measured in meters.
*/
const float kGoalDepth = 0.18;

//-------------------

/*!
  @brief Definition of the representation of the state of each individual robot.
*/
struct Robot {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
  float orientation;
};

/*!
  @brief Definition of the representation of the state of the ball.
*/
struct Ball {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
};

/*!
  @brief Definition of the representation of the physical field being played on.
*/
struct Field {
  float width;
  float height;
};

/*!
  @brief Definition of the representation of the state of the game, given by the game controller.
*/
enum class GameState {
  kHalted,
  kPlaying,
};

/*!
  @brief Definition of the representation of the state of the world at any given time.
*/
struct World {
  std::vector<Robot> robots;
  Ball ball;
  Field field;
  GameState state;

  /*!
    @brief Id of the robot that has the ball. -1 if no robot have it.
  */
  int have_ball_id;
};

/*!
  @returns a tensor representing the reward given by the average distance between all robots, with the shape [num_agents, 1].
  @param positions: A tensor of all the positions of all the robots, with the shape[2, num_agents]
  @param max_distance: The maximum distance from the average position of all the robots when no reward will be given anymore. @note max_distance cannot be 0!
  @param max_reward: The maximum reward that will be given when a robot is within the range [0, max_distance].
*/
torch::Tensor ComputeAverageDistanceReward(torch::Tensor positions, float max_distance, float max_reward);

}
}

#endif
