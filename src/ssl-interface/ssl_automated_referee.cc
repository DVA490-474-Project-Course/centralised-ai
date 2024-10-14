// ssl_automated_referee.cc
//==============================================================================
// Author: Aaiza A. Khan, Shruthi P. Kunnon
// Creation date: 2024-10-10
// Description: Automates referee commands based on robot and ball positions.
// License: See LICENSE file for license details.
//==============================================================================

#include "ssl_automated_referee.h"
#include <iostream>
#include <string>

namespace centralized_ai
{
namespace ssl_interface
{

AutomatedReferee::AutomatedReferee(VisionClient& vision_client)
    : vision_client_(vision_client), referee_command(Referee::STOP), blue_team_score(0), 
      yellow_team_score(0), last_kicker_team(Team::UNKNOWN), ball_designated_position_x(0.0), 
      ball_designated_position_y(0.0) {}

// Analyze the game state by using VisionClient to access robot and ball positions
void AutomatedReferee::AnalyzeGameState()
{
    float ball_x = vision_client_.GetBallPositionX();
    float ball_y = vision_client_.GetBallPositionY();
    std::cout << "Ball Position - X: " << ball_x << " Y: " << ball_y << std::endl;

    // Check if a goal has been scored
    if (IsGoalScored(ball_x, ball_y))
    {
        if (ball_x > 4500 && ball_y > -0500 && ball_y < 0500) // Ball in the blue team's goal
        {
            referee_command = Referee::GOAL_BLUE;
            blue_team_score++;
            std::cout << "Blue Team Scored!" << std::endl;
        }
        else if (ball_x < -4500 && ball_y > -0500 && ball_y < 0500) // Ball in the yellow team's goal
        {
            referee_command = Referee::GOAL_YELLOW;
            yellow_team_score++;
            std::cout << "Yellow Team Scored!" << std::endl;
        }
    }
    // Check for ball placement if out of field
    else if (IsBallOutOfField(ball_x, ball_y))
    {
        if (ball_x > 4.5 && ball_y > 0.5)
        {
            ball_designated_position_x = 4.3;
            ball_designated_position_y = 2.8;
        }
        else if (ball_x > 4.5 && ball_y < -0.5)
        {
            ball_designated_position_x = 4.3;
            ball_designated_position_y = -2.8;
        }
        else if (ball_x < -4.5 && ball_y > 0.5)
        {
            ball_designated_position_x = -4.3;
            ball_designated_position_y = 2.8;
        }
        else if (ball_x < -4.5 && ball_y < -0.5)
        {
            ball_designated_position_x = -4.3;
            ball_designated_position_y = -2.8;
        }

        if (last_kicker_team == Team::YELLOW)
        {
            referee_command = Referee::DIRECT_FREE_BLUE;  // Free kick for blue team
        }
        else if (last_kicker_team == Team::BLUE)
        {
            referee_command = Referee::DIRECT_FREE_YELLOW; // Free kick for yellow team
        }
    }
    else if (IsKickoffConditionMet(ball_x, ball_y))
    {
        referee_command = Referee::PREPARE_KICKOFF_BLUE; // Example kickoff for blue
    }
    else
    {
        referee_command = Referee::NORMAL_START;
    }
}

// Print the current command and score
void AutomatedReferee::PrintCommand()
{
    // Convert Protobuf enum (Referee::Command) to internal enum (RefereeCommand)
    RefereeCommand internal_command = ConvertRefereeCommand(referee_command);

    // Now use the internal RefereeCommand to convert to string
    std::string command_str = RefereeCommandToString(internal_command);
    
    std::cout << "Referee Command: " << command_str << std::endl;
    std::cout << "Score - Blue: " << blue_team_score << " | Yellow: " << yellow_team_score << std::endl;
}


enum RefereeCommand AutomatedReferee::GetRefereeCommand()
{
    return ConvertRefereeCommand(referee_command);  // Convert before returning
}

// New getters for blue and yellow team scores
int AutomatedReferee::GetBlueTeamScore()
{
    return blue_team_score;
}

int AutomatedReferee::GetYellowTeamScore()
{
    return yellow_team_score;
}

// New getters for ball designated position
float AutomatedReferee::GetBallDesignatedPositionX()
{
    return ball_designated_position_x;
}

float AutomatedReferee::GetBallDesignatedPositionY()
{
    return ball_designated_position_y;
}


// Check if a goal has been scored based on ball position
bool AutomatedReferee::IsGoalScored(float ball_x, float ball_y)
{
    std::cout << "Checking Goal: X: " << ball_x << " Y: " << ball_y << std::endl;
    bool inside_goal = (ball_x > 4500 || ball_x < -4500);
    bool goal_scored = inside_goal && !(inside_goal_prev);
    inside_goal_prev = inside_goal;
    return goal_scored;
}

// Check if a kickoff condition is met (e.g., ball in center of field after a goal)
bool AutomatedReferee::IsKickoffConditionMet(float ball_x, float ball_y)
{
    return (ball_x > -10 && ball_x < 10 && ball_y > -10 && ball_y < 10);
}

// Check if the ball has gone out of field
bool AutomatedReferee::IsBallOutOfField(float ball_x, float ball_y)
{
    return (ball_x > 4.5 || ball_x < -4.5 || ball_y > 3.0 || ball_y < -3.0);
}

// Empty function to get the last robot to kick the ball
void AutomatedReferee::GetLastKicker(int& robot_id, Team& team)
{
    // Logic to determine which robot last kicked the ball will be implemented later
    // The function will update the robot_id and team variables
}
/* Convert from protobuf enum definition to project enum definition */
enum RefereeCommand AutomatedReferee::ConvertRefereeCommand(enum Referee_Command command)
{
  switch (command)
  {
    case Referee::HALT: return RefereeCommand::HALT;
    case Referee::STOP: return RefereeCommand::STOP;
    case Referee::NORMAL_START: return RefereeCommand::NORMAL_START;
    case Referee::FORCE_START: return RefereeCommand::FORCE_START;
    case Referee::PREPARE_KICKOFF_YELLOW: return RefereeCommand::PREPARE_KICKOFF_YELLOW;
    case Referee::PREPARE_KICKOFF_BLUE: return RefereeCommand::PREPARE_KICKOFF_BLUE;
    case Referee::PREPARE_PENALTY_YELLOW: return RefereeCommand::PREPARE_PENALTY_YELLOW;
    case Referee::PREPARE_PENALTY_BLUE: return RefereeCommand::PREPARE_PENALTY_BLUE;
    case Referee::DIRECT_FREE_YELLOW: return RefereeCommand::DIRECT_FREE_YELLOW;
    case Referee::DIRECT_FREE_BLUE: return RefereeCommand::DIRECT_FREE_BLUE;
    case Referee::TIMEOUT_YELLOW: return RefereeCommand::TIMEOUT_YELLOW;
    case Referee::TIMEOUT_BLUE: return RefereeCommand::TIMEOUT_BLUE;
    case Referee::BALL_PLACEMENT_YELLOW: return RefereeCommand::BALL_PLACEMENT_YELLOW;
    case Referee::BALL_PLACEMENT_BLUE: return RefereeCommand::BALL_PLACEMENT_BLUE;
    default: return RefereeCommand::UNKNOWN_COMMAND;
  }
}
/* Translate RefereeCommand enumerator to string */
std::string AutomatedReferee::RefereeCommandToString(RefereeCommand referee_command)
{
  switch (referee_command)
  {
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
} // namespace ssl_interface
} // namespace centralized_ai