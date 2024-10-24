/* ssl_automated_referee.h
 *==============================================================================
 * Author: Aaiza A. Khan, Shruthi P. Kunnon, Emil Åberg
 * Creation date: 2024-10-10
 * Last modified: 2024-10-23 by Emil Åberg
 * Description: Automates referee commands based on robot and ball positions.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "ssl_automated_referee.h"

/* C++ standard library headers */
#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>

/* Project .h files */
#include "ssl_vision_client.h"
#include "../common_types.h"
#include "simulation_reset.h"

namespace centralized_ai {
namespace ssl_interface {

/* Constructor */
AutomatedReferee::AutomatedReferee(VisionClient &vision_client,
  std::string grsim_ip, uint16_t grsim_port)
    : vision_client_(vision_client), referee_command(RefereeCommand::STOP),
      blue_team_score(0), yellow_team_score(0), last_kicker_team(Team::kUnknown),
      designated_position({0.0F, 0.0F}),
      game_running(false), grsim_ip(grsim_ip), grsim_port(grsim_port) {}

/* Analyze the game state by using VisionClient to access robot and ball
   positions */
void AutomatedReferee::AnalyzeGameState() {
  enum Team touching_ball;

  if (game_running) {
    /* Keep track of which team touched ball last */
    touching_ball = CheckForCollision();
    if (touching_ball != Team::kUnknown)
    {
      last_kicker_team = CheckForCollision();
    }

    /* Update current referee command according to ssl game rules */
    RefereeStateHandler();

    /* Update stage time left */
    stage_time_left = stage_time + (int)std::round(time_at_game_start -
      vision_client_.GetTimestamp());
  }
}

/* Updates the current referee command, designated position, score and
 * resets ball and robot position when a goal is scored. */
void AutomatedReferee::RefereeStateHandler() {
  float current_time = vision_client_.GetTimestamp();

  switch (referee_command) {
    case RefereeCommand::PREPARE_KICKOFF_YELLOW:
    case RefereeCommand::PREPARE_KICKOFF_BLUE:
      if (current_time - prepare_kickoff_start_time >= prepare_kickoff_duration) {
        referee_command = RefereeCommand::NORMAL_START;
      }
      break;
    case RefereeCommand::BALL_PLACEMENT_BLUE:
      if (BallSuccessfullyPlaced()) {
        referee_command = RefereeCommand::DIRECT_FREE_BLUE;
      }
      break;
    case RefereeCommand::BALL_PLACEMENT_YELLOW:
      if (BallSuccessfullyPlaced()) {
        referee_command = RefereeCommand::DIRECT_FREE_YELLOW;
      }
      break;
    case RefereeCommand::DIRECT_FREE_BLUE:
    case RefereeCommand::DIRECT_FREE_YELLOW:
    case RefereeCommand::NORMAL_START:
      if (IsBallInBlueGoal(vision_client_.GetBallPositionX(),
        vision_client_.GetBallPositionY())) {
        yellow_team_score++;
        referee_command = RefereeCommand::PREPARE_KICKOFF_BLUE;
        prepare_kickoff_start_time = current_time;
        ResetRobotsAndBall(grsim_ip, grsim_port);
      }
      else if (IsBallInYellowGoal(vision_client_.GetBallPositionX(),
        vision_client_.GetBallPositionY())) {
        blue_team_score++;
        referee_command = RefereeCommand::PREPARE_KICKOFF_YELLOW;
        prepare_kickoff_start_time = current_time;
        ResetRobotsAndBall(grsim_ip, grsim_port);
      }
      else if (IsBallOutOfField(vision_client_.GetBallPositionX(),
        vision_client_.GetBallPositionY())) {
        designated_position = CalcBallDesignatedPosition();
        if (last_kicker_team == Team::kYellow) {
          /* Free kick for blue team */
          referee_command = RefereeCommand::BALL_PLACEMENT_BLUE;
        } else if (last_kicker_team == Team::kBlue) {
          /* Free kick for yellow team */
          referee_command = RefereeCommand::BALL_PLACEMENT_YELLOW;
        }
      }
      break;
    default:
      /* unhandled state encountered */
      break;
  }
}

/* Starts the automated referee, reset score, referee command,
 * robot and ball positions. */
void AutomatedReferee::StartGame(enum Team starting_team,
  enum Team team_on_positive_half, double prepare_kickoff_duration,
  int64_t stage_time) {
  yellow_team_score = 0;
  blue_team_score = 0;
  designated_position.x = 0.0;
  designated_position.y = 0.0;
  prepare_kickoff_start_time = vision_client_.GetTimestamp();
  time_at_game_start = vision_client_.GetTimestamp();
  this->prepare_kickoff_duration = prepare_kickoff_duration;
  this->team_on_positive_half = team_on_positive_half;
  last_kicker_team = starting_team;
  game_running = true;
  this->stage_time = stage_time;
  stage_time_left = stage_time;

  if (starting_team == Team::kBlue) {
    referee_command = RefereeCommand::PREPARE_KICKOFF_BLUE;
  }
  else {
    referee_command = RefereeCommand::PREPARE_KICKOFF_YELLOW;
  }

  ResetRobotsAndBall(grsim_ip, grsim_port);
}

/* Stops the automated referee, outputs will no longer be updated. */
void AutomatedReferee::StopGame() {
  game_running = false;
}

/* Print the current command and score */
void AutomatedReferee::Print() {
  printf("referee command: <%s> score: <%i, %i> designated position <%f, %f> stage time left: <%li>\n",
    RefereeCommandToString(referee_command).c_str(),
    blue_team_score, yellow_team_score, designated_position.x, designated_position.y,
    stage_time_left);
}

/* Returns true if ball is in blue teams goal */
bool AutomatedReferee::IsBallInBlueGoal(float ball_x, float ball_y) {
  return (ball_x > 4500 && ball_y > -500 && ball_y < 500);
}

/* Returns true if ball is in yellow teams goal */
bool AutomatedReferee::IsBallInYellowGoal(float ball_x, float ball_y) {
  return (ball_x < -4500 && ball_y > -500 && ball_y < 500);
}

/* Check if the ball has gone out of field */
bool AutomatedReferee::IsBallOutOfField(float ball_x, float ball_y) {
  return (ball_x > 4500 || ball_x < -4500 || ball_y > 3000 || ball_y < -3000);
}

/* Returns which team is currently touching the ball, returns kUnknow if no
 * team is currently in contact with the ball. */
enum Team AutomatedReferee::CheckForCollision() {
  for (auto team : {Team::kYellow, Team::kBlue}) {
    for (int id = 0; id < team_size; id++) {
      if (DistanceToBall(id, team) <= ball_radius + collision_margin) {
        return team;
      }
    }
  }

  return Team::kUnknown;
}

/* Return distance to ball and specified robot */
float AutomatedReferee::DistanceToBall(int id, enum Team team) {
  return std::sqrt(
    std::pow((vision_client_.GetBallPositionX() -
      vision_client_.GetRobotPositionX(id, team)), 2) +
    std::pow((vision_client_.GetBallPositionY() -
      vision_client_.GetRobotPositionY(id, team)), 2)) -
    robot_radius;
}

/* Return distance to ball and specified point */
float AutomatedReferee::DistanceToBall(float x, float y) {
  return std::sqrt(
    std::pow((vision_client_.GetBallPositionX() - x), 2) +
    std::pow((vision_client_.GetBallPositionY() - y), 2));
}

bool AutomatedReferee::BallSuccessfullyPlaced() {
  /* there is no robot within 0.05 meters distance to the ball */
  for (auto team : {Team::kYellow, Team::kBlue}) {
    for (int id = 0; id < team_size; id++) {
      if (DistanceToBall(id, team) <= 50) {
        return false;
      }
    }
  }

  /* the ball is at a position within 0.15 meters radius from the requested position */
  if (DistanceToBall(designated_position.x, designated_position.y)
    > 150) {
    return false;
  }

  return true;
}

/* Assuming ball is out of field, returns the point of where ball should be
   placed for freekick/cornerkick. */
struct AutomatedReferee::Point AutomatedReferee::CalcBallDesignatedPosition() {
  struct Point local_designated_position;
  float ball_x = vision_client_.GetBallPositionX();
  float ball_y = vision_client_.GetBallPositionY();

  if ((ball_x > 4500 && ball_y > 500) || (ball_x >= 4300 && ball_y > 3000)) {
    local_designated_position.x = 4300;
    local_designated_position.y = 2800;
  } else if ((ball_x > 4500 && ball_y < -0500) ||
             (ball_x >= 4300 && ball_y < -3000)) {
    local_designated_position.x = 4300;
    local_designated_position.y = -2800;
  } else if ((ball_x < -4500 && ball_y > 0500) ||
             (ball_x <= -4300 && ball_y > 3000)) {
    local_designated_position.x = -4300;
    local_designated_position.y = 2800;
  } else if ((ball_x < -4500 && ball_y < -0500) ||
             (ball_x <= -4300 && ball_y < -3000)) {
    local_designated_position.x = -4300;
    local_designated_position.y = -2800;
  } else if (ball_x > -4300 && ball_x < 4300 && ball_y < -3000) {
    local_designated_position.x = ball_x;
    local_designated_position.y = -2800;
  } else if (ball_x > -4300 && ball_x < 4300 && ball_y > 3000) {
    local_designated_position.x = ball_x;
    local_designated_position.y = 2800;
  }

  return local_designated_position;
}

/* Translate RefereeCommand enumerator to string */
std::string
AutomatedReferee::RefereeCommandToString(RefereeCommand referee_command) {
  switch (referee_command) {
  case RefereeCommand::HALT: return "HALT";
  case RefereeCommand::STOP: return "STOP";
  case RefereeCommand::NORMAL_START: return "NORMAL_START";
  case RefereeCommand::FORCE_START: return "FORCE_START";
  case RefereeCommand::PREPARE_KICKOFF_YELLOW: return "PREPARE_KICKOFF_YELLOW";
  case RefereeCommand::PREPARE_KICKOFF_BLUE: return "PREPARE_KICKOFF_BLUE";
  case RefereeCommand::PREPARE_PENALTY_YELLOW: return "PREPARE_PENALTY_YELLOW";
  case RefereeCommand::PREPARE_PENALTY_BLUE: return "PREPARE_PENALTY_BLUE";
  case RefereeCommand::DIRECT_FREE_YELLOW: return "DIRECT_FREE_YELLOW";
  case RefereeCommand::DIRECT_FREE_BLUE: return "DIRECT_FREE_BLUE";
  case RefereeCommand::TIMEOUT_YELLOW: return "TIMEOUT_YELLOW";
  case RefereeCommand::TIMEOUT_BLUE: return "TIMEOUT_BLUE";
  case RefereeCommand::BALL_PLACEMENT_YELLOW: return "BALL_PLACEMENT_YELLOW";
  case RefereeCommand::BALL_PLACEMENT_BLUE: return "BALL_PLACEMENT_BLUE";
  default: return "UNKNOWN_COMMAND";
  }
}

/* Public getters */
enum RefereeCommand AutomatedReferee::GetRefereeCommand() {return referee_command;}
int AutomatedReferee::GetBlueTeamScore() {return blue_team_score;}
int AutomatedReferee::GetYellowTeamScore() {return yellow_team_score;}
float AutomatedReferee::GetBallDesignatedPositionX() {return designated_position.x;}
float AutomatedReferee::GetBallDesignatedPositionY() {return designated_position.y;}
enum Team AutomatedReferee::TeamOnPositiveHalf() {return team_on_positive_half;}
int64_t AutomatedReferee::GetStageTimeLeft() {return stage_time_left;};

} /* namespace ssl_interface */
} /* namespace centralized_ai */