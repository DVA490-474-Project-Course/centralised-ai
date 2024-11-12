/* main.cc
 *==============================================================================
 * Author: Jacob Johansson, Emil Åberg, Viktor Eriksson
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
#include "simulation-interface/simulation_interface.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <matplotlibcpp.h>
#include <iostream>

/*Configuration values*/
int max_timesteps = 300;
int steps = 0; /*move into mappo------------------------*/
int step_max = 0;
int batch_size = 5;
int amount_of_players_in_team = 6;
int input_size = 43; // Number of input features
int num_actions = 3;
int hidden_size = 64;

std::vector<double> critic_loss;

void PlotLoss()
{
  /* Set plot labels and title. */
  matplotlibcpp::figure();

  /* Plot the data in real-time. */
  while (true) {
  
      /* Plot the data. */
      matplotlibcpp::plot(critic_loss, "-k");

      matplotlibcpp::grid(true);

      matplotlibcpp::title("Critic Loss");
      matplotlibcpp::xlabel("Time Step");
      matplotlibcpp::ylabel("Critic Loss");
    
      //matplotlibcpp::xlim(0, static_cast<int32_t>(time_steps.size()));

      /* Pause for more efficient plotting. */
      matplotlibcpp::pause(0.1);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }  
}

int main() {
  std::vector<centralised_ai::collective_robot_behaviour::Agents> models; /*Create Models class for each robot.*/
  centralised_ai::collective_robot_behaviour::CriticNetwork critic; /*Create global critic network*/

  /*Comment out if want to create new agents, otherwise load in saved models*/
  models = centralised_ai::collective_robot_behaviour::CreateAgents(amount_of_players_in_team);
  //models = LoadAgents(amount_of_players_in_team,critic); //Load in the trained model

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

  std::vector<robot_controller_interface::simulation_interface::SimulationInterface> simulation_interfaces;
  for (int32_t id = 0; id < 6; id++)
  {
    simulation_interfaces.push_back(robot_controller_interface::simulation_interface::SimulationInterface(grsim_ip, grsim_port, id, robot_controller_interface::Team::kBlue));
    //simulation_interfaces[id].SetVelocity(5.0F, 0.0F, 0.0F);
  }

  // Launch the plotting in a separate thread
  std::thread plot_thread(PlotLoss);

  auto old_net = models;
  auto old_net_critic = critic;
  while (true) {
    std::cout << "Running" << std::endl;
    /*run actions and save  to buffer*/
    auto databuffer = MappoRun(models,critic,referee,vision_client,
      centralised_ai::Team::kBlue,simulation_interfaces);

      /* Push the mean reward for each time step in the returned epoch. */
      for (int32_t t = 0; t < databuffer[0].t.size(); t++)
      {
        /* Todo: Push critic loss to the list. */
      }

    /*Run Mappo Agent algorithm by Policy Models and critic network*/
    centralised_ai::collective_robot_behaviour::Mappo_Update(models,critic,databuffer,old_net, old_net_critic);

    referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow,3.0F, 300);
  }

  plot_thread.join();


  return 0;
}
