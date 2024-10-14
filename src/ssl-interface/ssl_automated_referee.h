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

    AutomatedReferee(VisionClient& vision_client);
    // Analyze the game state using VisionClient and generate commands
    void AnalyzeGameState();

    // Print the current command and score
    void PrintCommand();

    // New getters for game state data
    enum RefereeCommand GetRefereeCommand();             // Same return type as ssl_game_controller_client.h
    int GetBlueTeamScore();                              // Same return type
    int GetYellowTeamScore();                            // Same return type
    float GetBallDesignatedPositionX();                  // Same return type
    float GetBallDesignatedPositionY();                  // Same return type


    // NEW: Function to get the robot ID and team of the last kicker
    void GetLastKicker(int& robot_id, Team& team);
      // New methods to convert referee commands
    enum RefereeCommand ConvertRefereeCommand(enum Referee_Command command); // Added declaration
    std::string RefereeCommandToString(enum RefereeCommand referee_command); // Added declaration


private:
    VisionClient& vision_client_;
    Referee::Command referee_command;
    int blue_team_score;
    int yellow_team_score;
    float ball_designated_position_x;
    float ball_designated_position_y;
    bool inside_goal_prev = false;
    

    // Track the team that last kicked the ball
    Team last_kicker_team;

    // Helper to translate command to string
    std::string CommandToString(Referee::Command command);

    // Game logic to determine if a goal has been scored
    bool IsGoalScored(float ball_x, float ball_y);

    // Logic for detecting other conditions (e.g., kickoffs)
    bool IsKickoffConditionMet(float ball_x, float ball_y);

    // Detect if the ball is out of field
    bool IsBallOutOfField(float ball_x, float ball_y);
};

} // namespace ssl_interface
} // namespace centralized_ai

#endif // SSL_AUTOMATED_REFEREE_H
