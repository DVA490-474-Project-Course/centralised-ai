/* simulation_interface_demo.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-09-16
 * Last modified: 2024-10-15 by Emil Åberg
 * Description: Demo of the simulation interface
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "simulation_interface.h"

int main(int argc, char *argv[])
{
  robot_controller_interface::simulation_interface::SimulationInterface simulation_interface(
    "127.0.0.1", 20011, 3, robot_controller_interface::Team::kBlue);
  simulation_interface.SetVelocity(10.0F, 0.0F, 0.0F);

  /* Run a loop to send commands and then reset the robot after 600 iterations */
  int count = 0;
  while (true)
  {
    simulation_interface.SendPacket();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  return 0;
}