/* common_types.h
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-09-24
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Common types used by the centralized ai program.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_COMMONTYPES_H_
#define CENTRALISEDAI_COMMONTYPES_H_

/* Related .h files */

/* C++ standard library headers */

/* Other .h files */

/* Project .h files */

namespace centralised_ai
{


/*! 
 * @brief The maximum number of timesteps for the simulation or training.
 */
const int max_timesteps = 201;

/*!
 * @brief Initial step count, starting from zero.
 */
const int steps = 0;

/*!
 * @brief Maximum steps per episode or iteration.
 */
const int step_max = 0;

/*!
 * @brief Length of the experience replay buffer.
 */
const int buffer_length = 2;

/*!
 * @brief Number of players per team in the simulation or game.
 */
const int amount_of_players_in_team = 2;

/*!
 * @brief Batch size for training, calculated as buffer length multiplied by the number of players in the team.
 */
const int batch_size = buffer_length * amount_of_players_in_team;

/*!
 * @brief Input size for neural networks, typically representing the number of features in the state or observation.
 */
const int input_size = 9;

/*!
 * @brief Number of possible actions each agent can take.
 */
const int num_actions = 3;

/*!
 * @brief Size of the hidden layer in LSTM or RNN-based neural networks.
 */
const int hidden_size = 64;


/*! 
 * @brief The robot radius in mm.
 */
const float robot_radius = 85;

/*! 
 * @brief The ball radius in mm.
 */
const float ball_radius = 21.5;

/*! 
 * @brief The number of robots in each team.
 */
const int team_size = 2;

/*! 
 * @brief Enum representing player team selection.
 */
enum class Team
{
  kBlue = 0,
  kYellow = 1,
  kUnknown = -1
};

/*! 
 * @brief Enum with the possible referee commands that can be received from the
 * game controller.
 */
enum class RefereeCommand
{
  /* All robots should completely stop moving. */
  HALT = 0,
  /* Robots must keep 50 cm from the ball. */
  STOP = 1,
  /* A prepared kickoff or penalty may now be taken. */
  NORMAL_START = 2,
  /* The ball is dropped and free for either team. */
  FORCE_START = 3,
  /* The yellow team may move into kickoff position. Followed by Normal Start. */
  PREPARE_KICKOFF_YELLOW = 4,
  /* The blue team may move into kickoff position. Followed by Normal Start. */
  PREPARE_KICKOFF_BLUE = 5,
  /* The yellow team may move into penalty position. Followed by Normal Start. */
  PREPARE_PENALTY_YELLOW = 6,
  /* The blue team may move into penalty position. Followed by Normal Start. */
  PREPARE_PENALTY_BLUE = 7,
  /* The yellow team may take a direct free kick. */
  DIRECT_FREE_YELLOW = 8,
  /* The blue team may take a direct free kick. */
  DIRECT_FREE_BLUE = 9,
  /* The yellow team is currently in a timeout. */
  TIMEOUT_YELLOW = 12,
  /* The blue team is currently in a timeout. */
  TIMEOUT_BLUE = 13,
  /* Equivalent to STOP, but the yellow team must pick up the ball and
     drop it in the Designated Position. */
  BALL_PLACEMENT_YELLOW = 16,
  /* Equivalent to STOP, but the blue team must pick up the ball and drop
     it in the Designated Position. */
  BALL_PLACEMENT_BLUE = 17,
  /* Unknown Command */
  UNKNOWN_COMMAND = -1
};

} /* namespace centralised_ai */

namespace robot_controller_interface
{
  /*! 
 * @brief Enum representing player team selection.
 */
  enum class Team
  {
    kBlue = 0,
    kYellow = 1,
    kUnknown = -1
  };
}

#endif /* CENTRALISEDAI_COMMONTYPES_H_ */
