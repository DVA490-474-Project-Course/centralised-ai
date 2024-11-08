/* simulation_interface.cc
 *==============================================================================
 * Author: Emil Åberg, Shruthi Puthiya Kunnon, Aaiza Aziz Khan
 * Creation date: 2024-09-25
 * Last modified: 2024-10-07 by Emil Åberg
 * Description: Interface that can send UDP data to grSim to make robots move
 * and kick
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "simulation_interface.h"

/* C system headers */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

/* C++ standard library headers */
#include <memory>
#include <string>

/* Project .h files */
#include "grSim_Commands.pb.h"
#include "grSim_Packet.pb.h"
#include "../common_types.h"

namespace robot_controller_interface
{
namespace simulation_interface
{

/* Constructor */
SimulationInterface::SimulationInterface(std::string ip, uint16_t port, int id, enum Team team)
{
  /* Define destination address */
  destination.sin_family = AF_INET;
  destination.sin_port = htons(port);
  destination.sin_addr.s_addr = inet_addr(ip.c_str());

  /* Create the client socket */
  socket = ::socket(AF_INET, SOCK_DGRAM, 0);

  /* Set initial values for robot */
  SetRobot(id, team);
  SetVelocity(0.0F, 0.0F, 0.0F);
  SetKickerSpeed(0.0F);
  SetSpinnerOn(false);
}

void SimulationInterface::SetRobot(int id, enum Team team)
{
  this->id = id;
  this->team = team;
}

void SimulationInterface::SetVelocity(float x_speed, float y_speed, float angular_speed)
{
  this->using_wheel_speed = false;
  this->x_speed = x_speed;
  this->y_speed = y_speed;
  this->angular_speed = angular_speed;
}

void SimulationInterface::SetVelocity(float front_left_wheel_speed, float back_left_wheel_speed,
    float back_right_wheel_speed, float front_right_wheel_speed)
{
  this->using_wheel_speed = true;
  this->wheel1 = -front_left_wheel_speed;
  this->wheel2 = -back_left_wheel_speed;
  this->wheel3 = back_right_wheel_speed;
  this->wheel4 = front_right_wheel_speed;
}

void SimulationInterface::SetKickerSpeed(float kicker_speed)
{
  this->kicker_speed = kicker_speed;
}

void SimulationInterface::SetSpinnerOn(bool spinner_on)
{
  this->spinner_on = spinner_on;
}

/* Send a UDP packet with the robot command */
void SimulationInterface::SendPacket()
{
  grSim_Packet packet;
  grSim_Robot_Command *command;

  /* Write the data to the protobuf message */
  packet.mutable_commands()->set_isteamyellow(team == Team::kYellow);
  packet.mutable_commands()->set_timestamp(0.0L);
  command = packet.mutable_commands()->add_robot_commands();
  command->set_id(id);
  command->set_kickspeedx(kicker_speed);
  command->set_kickspeedz(0.0F);
  command->set_spinner(spinner_on);
  command->set_wheelsspeed(using_wheel_speed);
  command->set_wheel1(wheel1);
  command->set_wheel2(wheel2);
  command->set_wheel3(wheel3);
  command->set_wheel4(wheel4);
  command->set_veltangent(x_speed);
  command->set_velnormal(y_speed);
  command->set_velangular(angular_speed);

  /* Send the packet */
  SendPacket(packet);
}

/* Send a grSim packet with UDP */
void SimulationInterface::SendPacket(grSim_Packet packet)
{
  size_t size;
  void *buffer;

  /* Serialize the protobuf message before sending */
  size = packet.ByteSizeLong();
  buffer = (char *)malloc(size);
  packet.SerializeToArray(buffer, size);

  /* Send the UDP packet*/
  ::sendto(socket, buffer, size, 0, reinterpret_cast<sockaddr *>(&destination),
           sizeof(destination));

  free(buffer);
}

} /* namespace simulation_interface */
} /* namespace robot_controller_interface */