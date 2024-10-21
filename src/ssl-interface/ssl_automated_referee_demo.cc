#include "ssl_automated_referee.h"
#include "ssl_vision_client.h"
#include <iostream>
#include <string>

#include "../common_types.h"

using namespace centralized_ai::ssl_interface;

int main()
{
  /* Define the IP and port for the VisionClient */
  std::string vision_ip = "127.0.0.1";
  int vision_port = 10006;

  /* Define the IP and command listen port for grSim */
  std::string grsim_ip = "127.0.0.1";
  int grsim_port = 20011;

  /* Create the VisionClient instance with IP and port */
  centralized_ai::ssl_interface::VisionClient vision_client(vision_ip, vision_port);
  vision_client.ReceivePacket();

  /* Create the AutomatedReferee instance with the VisionClient */
  AutomatedReferee referee(vision_client, grsim_ip, grsim_port);

  /* Start the automated referee */
  referee.StartGame(centralized_ai::Team::kBlue, centralized_ai::Team::kBlue, 3.0F);

  while (true)
  {
    vision_client.ReceivePacket();
    
    // Call AnalyzeGameState to check the goal logic
    referee.AnalyzeGameState();
    referee.PrintCommand();

    // Print the current ball designated position if it's set
    std::cout << "Ball Designated Position X: " << referee.GetBallDesignatedPositionX() << std::endl;
    std::cout << "Ball Designated Position Y: " << referee.GetBallDesignatedPositionY() << std::endl;
  }


  return 0;
}
