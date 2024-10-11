// ssl_automated_referee.h
//==============================================================================
// Author: Aaiza A. Khan
// Creation date: 2024-10-10
// Description: Automates referee commands based on robot and ball positions.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef SSL_AUTOMATED_REFEREE_H
#define SSL_AUTOMATED_REFEREE_H

// C++ standard library headers
#include <string>

// Protobuf generated header
#include "ssl_gc_referee_message.pb.h"
#include "ssl_vision_client.h" // Include the VisionClient header
#include "../common_types.h"

namespace centralized_ai
{
namespace ssl_interface
{

class AutomatedReferee
{
public:
    AutomatedReferee();

    // Analyze the position data from VisionClient and generate commands
    void AnalyzeGameState(PositionData* position_data);

    // Method to print the current command and score
    void PrintCommand();

private:
    Referee::Command referee_command;
    int blue_team_score;
    int yellow_team_score;

    // Helper to translate command to string
    std::string CommandToString(Referee::Command command);

    // Game logic to determine if a goal has been scored
    bool IsGoalScored(const PositionData* position_data);

    // Example logic for detecting other conditions (e.g., kickoffs)
    bool IsKickoffConditionMet(const PositionData* position_data);
};

} // namespace ssl_interface
} // namespace centralized_ai

#endif // SSL_AUTOMATED_REFEREE_H
