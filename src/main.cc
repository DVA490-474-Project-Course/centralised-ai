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
try {
        // Initialize the Python interpreter in this thread
        pybind11::scoped_interpreter guard{};

        // Import matplotlib in Python
        pybind11::module plt = pybind11::module::import("matplotlib.pyplot");

        // Create a plot figure
        pybind11::object figure = plt.attr("figure")();
        pybind11::object ax = figure.attr("add_subplot")(111);  // Add a subplot

        // Set plot labels and title
        ax.attr("set_title")("Real-Time Critic Loss Plot");
        ax.attr("set_xlabel")("Time");
        ax.attr("set_ylabel")("Critic Loss");


        plt.attr("show")();  // Show the plot

        // Plot the data in real-time
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Adjust the interval to your need

            // Read the latest data in the vector (non-blocking)
            pybind11::list py_data = pybind11::cast(critic_loss);  // Convert the data vector to a Python list

            // Redraw the plot
            plt.attr("plot")();  // Update the plot with new data
            plt.attr("pause")(0.01);  // Allow matplotlib to process the plot update
        }
    }
    catch (const pybind11::error_already_set& e) {
        std::cerr << "Python exception occurred: " << e.what() << std::endl;
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
  int32_t counter = 0;
  while (true) {
    std::cout << "Running" << std::endl;
    /*run actions and save  to buffer*/
    auto databuffer = MappoRun(models,critic,referee,vision_client,
      centralised_ai::Team::kBlue,simulation_interfaces);

      //critic_loss.push_back(databuffer[0].t[0].rewards[0].item<float>());
      critic_loss.push_back(counter);
      counter++;

    /*Run Mappo Agent algorithm by Policy Models and critic network*/
    centralised_ai::collective_robot_behaviour::Mappo_Update(models,critic,databuffer,old_net, old_net_critic);

    referee.StartGame(centralised_ai::Team::kBlue, centralised_ai::Team::kYellow,3.0F, 300);
  }

  plot_thread.join();


  return 0;
}
