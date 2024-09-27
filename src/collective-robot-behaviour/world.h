#ifndef WORLD_H
#define WORLD_H

#include <vector>

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

#endif  // WORLD_H
