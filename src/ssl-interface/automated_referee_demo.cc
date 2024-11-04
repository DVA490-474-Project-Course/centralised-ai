/* automated_referee_demo.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-10-23
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: demo of the automated referee
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* C++ standard library headers */
#include <iostream>
#include <string>

/* Project .h files */
#include "automated_referee.h"
#include "ssl_vision_client.h"
#include "../common_types.h"

int main()
{
  /* Define the IP and port for the VisionClient */
  std::string vision_ip = "127.0.0.1";
  int vision_port = 10006;

  /* Define the IP and command listen port for grSim */
  std::string grsim_ip = "127.0.0.1";
  int grsim_port = 20011;

  /* Create the VisionClient instance with IP and port */
  centralised_ai::ssl_interface::VisionClient vision_client(vision_ip, vision_port);
  vision_client.ReceivePacket();

  /* Create the AutomatedReferee instance with the VisionClient */
  centralised_ai::ssl_interface::AutomatedReferee referee(vision_client, grsim_ip,
    grsim_port);

  /* Start the automated referee */
  referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow, 3.0F, 300);

  while (true)
  {
    vision_client.ReceivePacket();
    
    /* Call AnalyzeGameState to check the goal logic */
    referee.AnalyzeGameState();
    referee.Print();
  }

  return 0;
}
