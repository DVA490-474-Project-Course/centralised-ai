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
  kHalt = 0,
  /* Robots must keep 50 cm from the ball. */
  kStop = 1,
  /* A prepared kickoff or penalty may now be taken. */
  kNormalStart = 2,
  /* The ball is dropped and free for either team. */
  kForceStart = 3,
  /* The yellow team may move into kickoff position. Followed by Normal Start. */
  kPrepareKickoffYellow = 4,
  /* The blue team may move into kickoff position. Followed by Normal Start. */
  kPrepareKickoffBlue = 5,
  /* The yellow team may move into penalty position. Followed by Normal Start. */
  kPreparePenaltyYellow = 6,
  /* The blue team may move into penalty position. Followed by Normal Start. */
  kPreparePenaltyBlue = 7,
  /* The yellow team may take a direct free kick. */
  kDirectFreeYellow = 8,
  /* The blue team may take a direct free kick. */
  kDirectFreeBlue = 9,
  /* The yellow team is currently in a timeout. */
  kTimeoutYellow = 12,
  /* The blue team is currently in a timeout. */
  kTimeoutBlue = 13,
  /* Equivalent to STOP, but the yellow team must pick up the ball and
     drop it in the Designated Position. */
  kBallPlacementYellow = 16,
  /* Equivalent to STOP, but the blue team must pick up the ball and drop
     it in the Designated Position. */
  kBallPlacementBlue = 17,
  /* Unknown Command */
  kUnknownCommand = -1
};

} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COMMONTYPES_H_ */
