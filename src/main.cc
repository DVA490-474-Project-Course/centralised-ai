/* main.cc
 *==============================================================================
 * Author: Jacob Johansson, Emil Åberg
 * Creation date: 2024-09-16
 * Last modified: 2024-10-01 by Jacob Johansson
 * Description: Main function.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* C++ standard library */
#include <vector>

/* Project .h files */
#include "collective-robot-behaviour/world.h"
#include "collective-robot-behaviour/utils.h"
#include "collective-robot-behaviour/network.h"
#include "collective-robot-behaviour/mappo.h"
#include "ssl-interface/ssl_vision_client.h"

/*Configuration values*/
int max_timesteps = 20;
int steps = 0; /*move into mappo------------------------*/
int step_max = 0;
int batch_size = 30;
int amount_of_players_in_team = 6;
int input_size = 43; // Number of input features
int num_actions = 4;
int hidden_size = 7;

int main() {
  std::vector<centralised_ai::collective_robot_behaviour::Agents> models; /*Create Models class for each robot.*/
  centralised_ai::collective_robot_behaviour::CriticNetwork critic; /*Create global critic network*/

  /*Comment out if want to create new agents, otherwise load in saved models*/
  models = centralised_ai::collective_robot_behaviour::CreateAgents(amount_of_players_in_team);
  //Models = LoadAgents(amount_of_players_in_team,critic); //Load in the trained model

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
  referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow,3.0F, 300);

  while (true) {
    /*run actions and save  to buffer*/
    auto databuffer = MappoRun(models,critic,referee,vision_client,centralised_ai::Team::kBlue);

    /*Run Mappo Agent algorithm by Policy Models and critic network*/
    centralised_ai::collective_robot_behaviour::Mappo_Update(models,critic,databuffer);
  }


  return 0;
}
