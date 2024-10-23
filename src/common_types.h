/* common_types.h
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-09-24
 * Last modified: 2024-10-21 by Emil Åberg
 * Description: Common types used by the centralized ai program.
 * License: See LICENSE file for license details.
 *==============================================================================
 */


#ifndef CENTRALIZEDAI_COMMONTYPES_H_
#define CENTRALIZEDAI_COMMONTYPES_H_


/* Related .h files */

/* C++ standard library headers */

/* Other .h files */

/* Project .h files */

namespace centralized_ai
{

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
const int team_size = 6;

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

} /* namespace centralized_ai */

#endif /* CENTRALIZEDAI_COMMONTYPES_H_ */
