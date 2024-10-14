#include "ssl_automated_referee.h"
#include "ssl_vision_client.h"
#include <iostream>
#include <string>

using namespace centralized_ai::ssl_interface;

int main() {
    // Define the IP and port for the VisionClient
    std::string vision_ip = "127.0.0.1";  // Replace with actual IP if needed
    int vision_port = 10003;  // Replace with actual port if needed

    // Create the VisionClient instance with IP and port
    VisionClient vision_client(vision_ip, vision_port);

    // Create the AutomatedReferee instance with the VisionClient
   
    AutomatedReferee referee(vision_client);

    while (true)
    {

        vision_client.ReceivePacket();

        // Manually set the last kicker team (Blue or Yellow)
        Team team = Team::BLUE;  // Change this to Team::YELLOW to test yellow's free kick
        int robot_id = 1;
        // Simulate the last kicker being set manually
        referee.GetLastKicker(robot_id, team);  // The function can be called, but team is set manually
        
        // Call AnalyzeGameState to check the goal logic
        referee.AnalyzeGameState();
        referee.PrintCommand();

        // Print the current ball designated position if it's set
        std::cout << "Ball Designated Position X: " << referee.GetBallDesignatedPositionX() << std::endl;
        std::cout << "Ball Designated Position Y: " << referee.GetBallDesignatedPositionY() << std::endl;

        // Update the last kicker team (Blue or Yellow)
        referee.GetLastKicker(robot_id, team);  // Change team as needed

        // Call AnalyzeGameState again to check ball placement logic
        referee.AnalyzeGameState();
        referee.PrintCommand();

        // Print the new ball designated position (should update based on logic)
        std::cout << "New Ball Designated Position X: " << referee.GetBallDesignatedPositionX() << std::endl;
        std::cout << "New Ball Designated Position Y: " << referee.GetBallDesignatedPositionY() << std::endl;

        // Final check on scores
        std::cout << "Final Blue Team Score: " << referee.GetBlueTeamScore() << std::endl;
        std::cout << "Final Yellow Team Score: " << referee.GetYellowTeamScore() << std::endl;
    }
  

    return 0;
}
