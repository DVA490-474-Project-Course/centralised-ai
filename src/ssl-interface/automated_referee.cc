/* automated_referee.cc
 *==============================================================================
 * Author: Aaiza A. Khan, Shruthi P. Kunnon, Emil Åberg
 * Creation date: 2024-10-10
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Automates referee commands based on robot and ball positions.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "../ssl-interface/automated_referee.h"

/* C++ standard library headers */
#include "cmath"
#include <cstdlib>
#include "iostream"
#include "string"

/* Project .h files */
#include "../ssl-interface/referee_command_functions.h"
#include "../ssl-interface/ssl_vision_client.h"
#include "../ssl-interface/simulation_reset.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

 /* Constructor: Initializes the AutomatedReferee class.  */
AutomatedReferee::AutomatedReferee(VisionClient &vision_client,
    std::string grsim_ip, uint16_t grsim_port)
    : vision_client_(vision_client),
      referee_command_(RefereeCommand::kStop),
    blue_team_score_(0),
    yellow_team_score_(0),
    last_kicker_team_(Team::kUnknown),
    designated_position_({0.0F, 0.0F}),
    game_running_(false),
    grsim_ip_(grsim_ip),
    grsim_port_(grsim_port) {
}

/* Analyze the game state by using VisionClient to access robot and ball
 * positions */
void AutomatedReferee::AnalyzeGameState()
{
  enum Team touching_ball;

  if (game_running_)
  {
    /* Keep track of which team touched ball last */
    touching_ball = CheckForCollision();
    if (touching_ball != Team::kUnknown)
    {
      last_kicker_team_ = CheckForCollision();
    }

    /* Update current referee command according to ssl game rules */
    RefereeStateHandler();

    /* Update stage time left */
    stage_time_left_ = stage_time_ +
        std::lround(time_at_game_start_ - vision_client_.GetTimestamp());
  }
}

/* Updates the current referee command, designated position, score and
 * resets ball and robot position when a goal is scored. */
void AutomatedReferee::RefereeStateHandler()
{
  float current_time = vision_client_.GetTimestamp();

  switch (referee_command_)
  {
    case RefereeCommand::kPrepareKickoffYellow:
    case RefereeCommand::kPrepareKickoffBlue:
      /* Transition to normal start after kickoff preparation time. */
      if (current_time - prepare_kickoff_start_time_
          >= prepare_kickoff_duration_)
      {
        referee_command_ = RefereeCommand::kNormalStart;
      }
      break;
    case RefereeCommand::kBallPlacementBlue:
      /* Transition to direct free kick if the ball is successfully placed. */
      if (BallSuccessfullyPlaced())
      {
        referee_command_ = RefereeCommand::kDirectFreeBlue;
      }
      break;
    case RefereeCommand::kBallPlacementYellow:
      /* Transition to direct free kick if the ball is successfully placed. */
      if (BallSuccessfullyPlaced())
      {
        referee_command_ = RefereeCommand::kDirectFreeYellow;
      }
      break;
    case RefereeCommand::kDirectFreeBlue:
    case RefereeCommand::kDirectFreeYellow:
    case RefereeCommand::kNormalStart:
      /* Handle goals and ball out-of-field events. */
      if (IsBallInGoal(Team::kBlue))
      {
        yellow_team_score_++;
        referee_command_ = RefereeCommand::kPrepareKickoffBlue;
        prepare_kickoff_start_time_ = current_time;
        ResetRobotsAndBall(grsim_ip_, grsim_port_, team_on_positive_half_);
      }
      else if (IsBallInGoal(Team::kYellow))
      {
        blue_team_score_++;
        referee_command_ = RefereeCommand::kPrepareKickoffYellow;
        prepare_kickoff_start_time_ = current_time;
        ResetRobotsAndBall(grsim_ip_, grsim_port_, team_on_positive_half_);
      }
      else if (IsBallOutOfField(vision_client_.GetBallPositionX(),
          vision_client_.GetBallPositionY()))
      {
        designated_position_ = CalcBallDesignatedPosition();
        /* Assign free kicks to the appropriate team. */
        if (last_kicker_team_ == Team::kYellow)
        {
          /* Free kick for blue team */
          referee_command_ = RefereeCommand::kBallPlacementBlue;
        }
        else if (last_kicker_team_ == Team::kBlue)
        {
          /* Free kick for yellow team */
          referee_command_ = RefereeCommand::kBallPlacementYellow;
        }
      }
      break;
    default:
      /* Handle unexpected states by throwing a runtime error. */
      throw std::runtime_error("unhandled state encountered: " +
          std::to_string(static_cast<int>(referee_command_)) +
          ". Runtime exception occurred.");
  }
}

/* Starts the automated referee, reset score, referee command,
 * robot and ball positions. */
void AutomatedReferee::StartGame(enum Team starting_team,
  enum Team team_on_positive_half, double prepare_kickoff_duration,
      int64_t stage_time)
{
  yellow_team_score_ = 0;
  blue_team_score_ = 0;
  designated_position_.x = 0.0F;
  designated_position_.y = 0.0F;
  prepare_kickoff_start_time_ = vision_client_.GetTimestamp();
  time_at_game_start_ = vision_client_.GetTimestamp();
  prepare_kickoff_duration_ = prepare_kickoff_duration;
  team_on_positive_half_ = team_on_positive_half;
  last_kicker_team_ = starting_team;
  game_running_ = true;
  stage_time_ = stage_time;
  stage_time_left_ = stage_time;

  /* Set the initial referee command based on the starting team. */
  if (starting_team == Team::kBlue)
  {
    referee_command_ = RefereeCommand::kPrepareKickoffBlue;
  }
  else
  {
    referee_command_ = RefereeCommand::kPrepareKickoffYellow;
  }
  /* Reset robots and ball to initial positions. */
  ResetRobotsAndBall(grsim_ip_, grsim_port_, team_on_positive_half_);
}

/* Stops the automated referee, outputs will no longer be updated. */
void AutomatedReferee::StopGame()
{
  game_running_ = false;
}

/* Print the current command and score */
void AutomatedReferee::Print()
{
  printf("referee command: <%s> score: <%i, %i> designated position <%f, "
      "%f> stage time left: <%li>\n",
      RefereeCommandToString(referee_command_).c_str(),
      blue_team_score_, yellow_team_score_, designated_position_.x,
      designated_position_.y, stage_time_left_);
}

/* Returns true if ball is in the goal of the specified team */
bool AutomatedReferee::IsBallInGoal(enum Team team)
{
  float ball_x = vision_client_.GetBallPositionX();
  float ball_y = vision_client_.GetBallPositionY();

  if (team_on_positive_half_ == team)
  {
    return (ball_x > kGoalXPositiveHalf_ && ball_y > kGoalWidthMinY_
        && ball_y < kGoalWidthMaxY_);
  }
  else
  {
    return (ball_x < kGoalXNegativeHalf_ && ball_y > kGoalWidthMinY_
        && ball_y < kGoalWidthMaxY_);
  }
}

/* Check if the ball has gone out of field */
bool AutomatedReferee::IsBallOutOfField(float ball_x, float ball_y)
{
  return (ball_x > kGoalXPositiveHalf_ || ball_x < kGoalXNegativeHalf_
      || ball_y > kBallOutOfFieldMaxY_ || ball_y < kBallOutOfFieldMinY_);
}

/* Returns true if the specified robot is currently touching the ball */
bool AutomatedReferee::IsTouchingBall(int id, enum Team team)
{
  return (DistanceToBall(id, team) <= kBallRadius + kCollisionMargin_);
}

/* Returns which team is currently touching the ball, returns KUnknown if no
 * team is currently in contact with the ball. */
enum Team AutomatedReferee::CheckForCollision()
{
  for (auto team : {Team::kYellow, Team::kBlue})
  {
    for (int id = 0; id < amount_of_players_in_team; id++)
    {
      if (IsTouchingBall(id, team))
      {
        return team;
      }
    }
  }

  return Team::kUnknown;
}

/* Return distance to center of the ball and specified robot */
float AutomatedReferee::DistanceToBall(int id, enum Team team)
{
  return std::sqrt(
    std::pow((vision_client_.GetBallPositionX() -
        vision_client_.GetRobotPositionX(id, team)), 2) +
    std::pow((vision_client_.GetBallPositionY() -
        vision_client_.GetRobotPositionY(id, team)), 2)) -
    kRobotRadius;
}

/* Return distance to center of the ball and specified point */
float AutomatedReferee::DistanceToBall(float x, float y)
{
  return std::sqrt(
      std::pow((vision_client_.GetBallPositionX() - x), 2) +
      std::pow((vision_client_.GetBallPositionY() - y), 2));
}

/* Returns true if ball is considered sucessfully placed according to SSL
 * rules */
bool AutomatedReferee::BallSuccessfullyPlaced()
{
  /* there is no robot within 0.05 meters distance to the ball */
  for (auto team : {Team::kYellow, Team::kBlue})
  {
    for (int id = 0; id < amount_of_players_in_team; id++)
    {
      if (DistanceToBall(id, team) <= 50)
      {
        return false;
      }
    }
  }

  /* the ball is at a position within 0.15 meters radius from the requested
   * position */
  if (DistanceToBall(designated_position_.x, designated_position_.y)
      > 150)
  {
    return false;
  }

  return true;
}

/* Assuming ball is out of field, returns the point of where ball should be
 * placed for freekick/cornerkick. */
struct AutomatedReferee::Point AutomatedReferee::CalcBallDesignatedPosition()
{
  struct Point local_designated_position;
  float ball_x = vision_client_.GetBallPositionX();
  float ball_y = vision_client_.GetBallPositionY();

  if ((ball_x > kGoalXPositiveHalf_ && ball_y > kGoalWidthMaxY_) ||
      (ball_x >= 4300 && ball_y > kBallOutOfFieldMinY_))
  {
    local_designated_position.x = 4300;
    local_designated_position.y = 2800;
  }
  else if ((ball_x > kGoalXPositiveHalf_ && ball_y < -kGoalWidthMaxY_) ||
      (ball_x >= 4300 && ball_y < -kBallOutOfFieldMinY_))
  {
    local_designated_position.x = 4300;
    local_designated_position.y = -2800;
  }
  else if ((ball_x < kGoalXNegativeHalf_ && ball_y > kGoalWidthMaxY_) ||
      (ball_x <= -4300 && ball_y > kBallOutOfFieldMinY_))
  {
    local_designated_position.x = -4300;
    local_designated_position.y = 2800;
  }
  else if ((ball_x < kGoalXNegativeHalf_ && ball_y < -kGoalWidthMaxY_) ||
      (ball_x <= -4300 && ball_y < -kBallOutOfFieldMinY_))
  {
    local_designated_position.x = -4300;
    local_designated_position.y = -2800;
  }
  else if (ball_x > -4300 && ball_x < 4300 &&
      ball_y < -kBallOutOfFieldMinY_)
  {
    local_designated_position.x = ball_x;
    local_designated_position.y = -2800;
  }
  else if (ball_x > -4300 && ball_x < 4300 &&
      ball_y > kBallOutOfFieldMinY_)
  {
    local_designated_position.x = ball_x;
    local_designated_position.y = 2800;
  }

  return local_designated_position;
}

/* Public getters */
enum RefereeCommand AutomatedReferee::GetRefereeCommand() {
  return referee_command_;
}
int AutomatedReferee::GetBlueTeamScore() {
  return blue_team_score_;
}
int AutomatedReferee::GetYellowTeamScore() {
  return yellow_team_score_;
}
float AutomatedReferee::GetBallDesignatedPositionX() {
  return designated_position_.x;
}
float AutomatedReferee::GetBallDesignatedPositionY() {
  return designated_position_.y;
}
enum Team AutomatedReferee::TeamOnPositiveHalf() {
  return team_on_positive_half_;
}
int64_t AutomatedReferee::GetStageTimeLeft() {
  return stage_time_left_;
};

} /* namespace ssl_interface */
} /* namespace centralised_ai */