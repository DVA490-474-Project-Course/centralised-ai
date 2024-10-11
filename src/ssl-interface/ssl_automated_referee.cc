// ssl_automated_referee.cc
//==============================================================================
// Author: Aaiza A. Khan, Shruthi P. Kunnon
// Creation date: 2024-10-10
// Description: Automates referee commands based on robot and ball positions.
// License: See LICENSE file for license details.
//==============================================================================

#include "ssl_automated_referee.h"
#include <iostream>

namespace centralized_ai
{
namespace ssl_interface
{

AutomatedReferee::AutomatedReferee()
    : referee_command(Referee::STOP), blue_team_score(0), yellow_team_score(0) {}

// Analyze the position data and generate the appropriate referee command
void AutomatedReferee::AnalyzeGameState(PositionData* position_data)
{
    // Example: Check if a goal has been scored
    if (IsGoalScored(position_data))
    {
        if (position_data->ball_position.x > 0) // Example: Ball in the blue team's goal
        {
            referee_command = Referee::GOAL_YELLOW;
            yellow_team_score++;
        }
        else // Ball in the yellow team's goal
        {
            referee_command = Referee::GOAL_BLUE;
            blue_team_score++;
        }
    }
    else if (IsKickoffConditionMet(position_data))
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
    std::string command_str = CommandToString(referee_command);
    std::cout << "Referee Command: " << command_str << std::endl;
    std::cout << "Score - Blue: " << blue_team_score << " | Yellow: " << yellow_team_score << std::endl;
}

// Check if a goal has been scored
bool AutomatedReferee::IsGoalScored(const PositionData* position_data)
{
    // Example: Check if the ball is near the goal (use actual goal positions from your field dimensions)
    // Assume x > 1000 is blue goal, x < -1000 is yellow goal (example dimensions)
    return (position_data->ball_position.x > 1000 || position_data->ball_position.x < -1000);
}

// Check if a kickoff condition is met (e.g., ball in center of field after a goal)
bool AutomatedReferee::IsKickoffConditionMet(const PositionData* position_data)
{
    // Example: Detect if the ball is at the center of the field (x, y close to 0)
    return (position_data->ball_position.x > -10 && position_data->ball_position.x < 10 &&
            position_data->ball_position.y > -10 && position_data->ball_position.y < 10);
}

// Helper function to convert Command enum to a string for display
std::string AutomatedReferee::CommandToString(Referee::Command command)
{
    switch (command)
    {
        case Referee::HALT: return "HALT";
        case Referee::STOP: return "STOP";
        case Referee::NORMAL_START: return "NORMAL_START";
        case Referee::FORCE_START: return "FORCE_START";
        case Referee::PREPARE_KICKOFF_YELLOW: return "PREPARE_KICKOFF_YELLOW";
        case Referee::PREPARE_KICKOFF_BLUE: return "PREPARE_KICKOFF_BLUE";
        case Referee::PREPARE_PENALTY_YELLOW: return "PREPARE_PENALTY_YELLOW";
        case Referee::PREPARE_PENALTY_BLUE: return "PREPARE_PENALTY_BLUE";
        case Referee::DIRECT_FREE_YELLOW: return "DIRECT_FREE_YELLOW";
        case Referee::DIRECT_FREE_BLUE: return "DIRECT_FREE_BLUE";
        case Referee::BALL_PLACEMENT_YELLOW: return "BALL_PLACEMENT_YELLOW";
        case Referee::BALL_PLACEMENT_BLUE: return "BALL_PLACEMENT_BLUE";
        case Referee::GOAL_YELLOW: return "GOAL_YELLOW";
        case Referee::GOAL_BLUE: return "GOAL_BLUE";
        default: return "UNKNOWN_COMMAND";
    }
}

} // namespace ssl_interface
} // namespace centralized_ai
