/* main.cc
 *==============================================================================
 * Author: Jacob Johansson, Emil Åberg, Viktor Eriksson
 * Creation date: 2024-09-16
 * Last modified: 2024-12-12 by Jacob Johansson & Viktor Eriksson
 * Description: Main function.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* C++ standard library */
#include <vector>

/* Project .h files */
#include "collective-robot-behaviour/utils.h"
#include "collective-robot-behaviour/network.h"
#include "collective-robot-behaviour/mappo.h"
#include "ssl-interface/ssl_vision_client.h"
#include "simulation-interface/simulation_interface.h"

#include "collective-robot-behaviour/communication.h"
#include "collective-robot-behaviour/evaluation.h"
#include "common_types.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <matplotlibcpp.h>
#include <iostream>
#include "common_types.h"
#include <ctime>

int main()
{
  /* Create the centralised critic network class */
  centralised_ai::collective_robot_behaviour::CriticNetwork critic;
  centralised_ai::collective_robot_behaviour::PolicyNetwork policy;

  /* Comment out if want to create new agents, otherwise load in saved models*/
  /*centralised_ai::collective_robot_behaviour::PolicyNetwork policy = centralised_ai::collective_robot_behaviour::CreatePolicy();*/
  LoadNetworks(policy, critic);

  /* Define the IP and port for the VisionClient */
  std::string vision_ip = "127.0.0.1";
  int vision_port = 10006;

  /* Define the IP and command listen port for grSim */
  std::string grsim_ip = "127.0.0.1";
  int grsim_port = 20011;

  /* Create the VisionClient instance with IP and port */
  centralised_ai::ssl_interface::VisionClient vision_client(vision_ip, vision_port);
  vision_client.ReceivePacketsUntilAllDataRead();

  /* Create the AutomatedReferee instance with the VisionClient */
  centralised_ai::ssl_interface::AutomatedReferee referee(vision_client, grsim_ip, grsim_port);

  /* Start the automated referee */
  referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow, 3.0F, 300);

  std::vector<centralised_ai::simulation_interface::SimulationInterface> simulation_interfaces;
  for (int32_t id = 0; id < centralised_ai::amount_of_players_in_team; id++)
  {
    simulation_interfaces.push_back(centralised_ai::simulation_interface::SimulationInterface(grsim_ip, grsim_port, id, centralised_ai::Team::kBlue));
  }

  /* Generate the file name from date. */
  /* Get current time */
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  /* Convert to local time and format */
  std::tm local_time = *std::localtime(&now_time);
  std::ostringstream oss;
  oss << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");  // e.g., "2024-09-12_14-30-00"

  /* Create the file name */
  std::string file_name = "../rewards/reward_" + oss.str() + ".csv";
  std::cout << "File name to save rewards: " << file_name << std::endl;

  /* Save the initial state of the networks. */
  centralised_ai::collective_robot_behaviour::SaveOldNetworks(policy, critic);

  int epochs = 0;
  std::cout << "Running" << std::endl;
  while (true)
  {
    referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow, 3.0F, 300);
    /*run actions and save  to buffer*/
    auto databuffer = centralised_ai::collective_robot_behaviour::MappoRun(policy, critic, referee, vision_client, centralised_ai::Team::kBlue, simulation_interfaces);

    /*Run Mappo Agent algorithm by Policy Models and critic network*/
    centralised_ai::collective_robot_behaviour::Mappo_Update(policy, critic, databuffer);

    /*Save the reward to go to a file*/
    int32_t num_batches = databuffer.size();
    int32_t num_time_steps = databuffer[0].t.size();
    torch::Tensor rewards = torch::zeros({num_batches, num_time_steps});
    
    for (int32_t b = 0; b < databuffer.size(); b++)
    {
      for (int32_t t = 0; t < databuffer[b].t.size(); t++)
      {
        rewards[b][t] = databuffer[b].t[t].rewards.mean();
      }
    }

    centralised_ai::collective_robot_behaviour::SaveRewardToFile(rewards.mean(), epochs, file_name);

    /* Update the epoch index */
    std::cout << "* Epochs: " << epochs << std::endl;
    epochs++;
  }

  return 0;
}
