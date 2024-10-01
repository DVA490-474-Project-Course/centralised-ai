//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Jacob Johansson
// Description: Headers for world.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef WORLD_H
#define WORLD_H

#include <vector>

namespace centralised_ai{
namespace collective_robot_behaviour{

// Definition of the representation of the state of each individual robot.
struct Robot {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
  float orientation;
};

// Definition of the representation of the state of the ball.
struct Ball {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
};

// Definition of the representation of the physical field being played on.
struct Field {
  float width;
  float height;
};

// Definition of the representation of the state of the game, given by the game controller.
enum class GameState {
  kHalted,
  kPlaying,
};

// Definition of the representation of the state of the world at any given time.
struct World {
  std::vector<Robot> robots;
  Ball ball;
  Field field;
  GameState state;
};

} // namespace collective_robot_behaviour
} // namespace centralised_ai

#endif  // WORLD_H
