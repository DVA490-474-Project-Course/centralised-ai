#include "ssl_automated_referee.h"
#include "ssl_vision_client.h"
#include <iostream>
#include <string>

using namespace centralized_ai::ssl_interface;

int main() {
    // Define the IP and port for the VisionClient
    std::string vision_ip = "127.0.0.1";  // Replace with actual IP if needed
    int vision_port = 10006;  // Replace with actual port if needed

    // Create the VisionClient instance with IP and port
    VisionClient vision_client(vision_ip, vision_port);

    // Create the AutomatedReferee instance with the VisionClient
   
    AutomatedReferee referee(vision_client);

    while (true)
    {

        vision_client.ReceivePacket();

        // Manually set the last kicker team (Blue or Yellow)
        enum centralized_ai::Team team = centralized_ai::Team::kBlue;  // Change this to Team::YELLOW to test yellow's free kick
        int robot_id = 1;
        
        // Call AnalyzeGameState to check the goal logic
        referee.AnalyzeGameState();
        referee.PrintCommand();

        // Print the current ball designated position if it's set
        std::cout << "Ball Designated Position X: " << referee.GetBallDesignatedPositionX() << std::endl;
        std::cout << "Ball Designated Position Y: " << referee.GetBallDesignatedPositionY() << std::endl;

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
