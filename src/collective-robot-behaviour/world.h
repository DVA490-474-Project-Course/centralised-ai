//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-11 by Jacob Johansson
// Description: Headers for world.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef WORLD_H
#define WORLD_H

#include <vector>

namespace centralised_ai{
namespace collective_robot_behaviour{

/* Global constants*/

/* Width of the playing field, measured in meters.*/
const int kFieldWidth = 9;
/* Height of the playing field, measured in meters.*/
const int kFieldHeight = 6;

/* Width of the defense area, measured in meters.*/
const int kDefenseAreaWidth = 2;
/* Height of the defense area, measured in meters.*/
const int kDefenseAreaHeight = 1;

/* Distance from the opponent's goal center from which a team executes a penalty kick against the opponent goal, measured in meters.*/
const int kPenaltyMarkDistance = 6;

/* Width of the goal, measured in meters.*/
const int kGoalWidth = 1;
/* Depth of the goal, measured in meters.*/
const float kGoalDepth = 0.18;

//-------------------

/* Definition of the representation of the state of each individual robot. */
struct Robot {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
  float orientation;
};

/* Definition of the representation of the state of the ball. */
struct Ball {
  float position_x;
  float position_y;
  float velocity_x;
  float velocity_y;
};

/* Definition of the representation of the physical field being played on. */
struct Field {
  float width;
  float height;
};

/* Definition of the representation of the state of the game, given by the game controller. */
enum class GameState {
  kHalted,
  kPlaying,
};

/* Definition of the representation of the state of the world at any given time. */
struct World {
  std::vector<Robot> robots;
  Ball ball;
  Field field;
  GameState state;

  /* Id of the robot that has the ball. -1 if no robot have it.*/
  int have_ball_id;
};

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif  /* WORLD_H */
