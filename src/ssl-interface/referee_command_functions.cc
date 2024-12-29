/* referee_command_functions.cc
 *==============================================================================
 * Author: Aaiza A. Khan, Emil Åberg
 * Creation date: 2024-11-26
 * Last modified: 2024-11-26 by Emil Åberg
 * Description: Contains functions to handle RefereeCommand enumerator.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "../ssl-interface/referee_command_functions.h"

/* C++ standard library headers */
#include "string"

/* Project .h files */
#include "../common_types.h"
#include "../ssl-interface/generated/ssl_gc_referee_message.pb.h"

namespace centralised_ai
{
namespace ssl_interface
{

/* Convert from protobuf enum definition to project enum definition.
 * The Referee_Command enum is defined by protobuf generated code. */
enum RefereeCommand ConvertRefereeCommand(enum Referee_Command command)
{
  switch (command)
  {
    case Referee::HALT:
      return RefereeCommand::kHalt;
    case Referee::STOP:
      return RefereeCommand::kStop;
    case Referee::NORMAL_START:
      return RefereeCommand::kNormalStart;
    case Referee::FORCE_START:
      return RefereeCommand::kForceStart;
    case Referee::PREPARE_KICKOFF_YELLOW:
      return RefereeCommand::kPrepareKickoffYellow;
    case Referee::PREPARE_KICKOFF_BLUE:
      return RefereeCommand::kPrepareKickoffBlue;
    case Referee::PREPARE_PENALTY_YELLOW:
      return RefereeCommand::kPreparePenaltyYellow;
    case Referee::PREPARE_PENALTY_BLUE:
      return RefereeCommand::kPreparePenaltyBlue;
    case Referee::DIRECT_FREE_YELLOW:
      return RefereeCommand::kDirectFreeYellow;
    case Referee::DIRECT_FREE_BLUE:
      return RefereeCommand::kDirectFreeBlue;
    case Referee::TIMEOUT_YELLOW:
      return RefereeCommand::kTimeoutYellow;
    case Referee::TIMEOUT_BLUE:
      return RefereeCommand::kTimeoutBlue;
    case Referee::BALL_PLACEMENT_YELLOW:
      return RefereeCommand::kBallPlacementYellow;
    case Referee::BALL_PLACEMENT_BLUE:
      return RefereeCommand::kBallPlacementBlue;
    default:
      return RefereeCommand::kUnknownCommand;
  }
}

/* Translate RefereeCommand enumerator to string */
std::string RefereeCommandToString(RefereeCommand referee_command)
{
  switch (referee_command)
  {
    case RefereeCommand::kHalt: return "kHalt";
    case RefereeCommand::kStop: return "kStop";
    case RefereeCommand::kNormalStart: return "kNormalStart";
    case RefereeCommand::kForceStart: return "kForceStart";
    case RefereeCommand::kPrepareKickoffYellow: return "kPrepareKickoffYellow";
    case RefereeCommand::kPrepareKickoffBlue: return "kPrepareKickoffBlue";
    case RefereeCommand::kPreparePenaltyYellow: return "kPreparePenaltyYellow";
    case RefereeCommand::kPreparePenaltyBlue: return "kPreparePenaltyBlue";
    case RefereeCommand::kDirectFreeYellow: return "kDirectFreeYellow";
    case RefereeCommand::kDirectFreeBlue: return "kDirectFreeBlue";
    case RefereeCommand::kTimeoutYellow: return "kTimeoutYellow";
    case RefereeCommand::kTimeoutBlue: return "kTimeoutBlue";
    case RefereeCommand::kBallPlacementYellow: return "kBallPlacementYellow";
    case RefereeCommand::kBallPlacementBlue: return "kBallPlacementBlue";
    default: return "kUnknownCommand";
  }
}

} /* namespace ssl_interface */
} /* namespace centralised_ai */