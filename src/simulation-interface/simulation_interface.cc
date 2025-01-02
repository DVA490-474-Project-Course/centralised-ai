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
#include "../simulation-interface/simulation_interface.h"

/* C system headers */
#include "arpa/inet.h"
#include "netinet/in.h"
#include "sys/socket.h"

/* C++ standard library headers */
#include "memory"
#include "string"

/* Project .h files */
#include "../simulation-interface/generated/grsim_commands.pb.h"
#include "../simulation-interface/generated/grsim_packet.pb.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace simulation_interface
{

/* Constructor */
SimulationInterface::SimulationInterface(std::string ip, uint16_t port,
    int id, enum Team team)
{
  /* Define destination address */
  destination_.sin_family = AF_INET;
  destination_.sin_port = htons(port);
  destination_.sin_addr.s_addr = inet_addr(ip.c_str());

  /* Create the client socket */
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);

  /* Set initial values for robot */
  SetRobot(id, team);
  SetVelocity(0.0F, 0.0F, 0.0F);
  SetKickerSpeed(0.0F);
  SetSpinnerOn(false);
}

 /* Function for setting robot through robot id and team */
void SimulationInterface::SetRobot(int id, enum Team team)
{
  id_ = id;
  team_ = team;
}

 /* set the robot velocity in terms of x,y and angular speed*/
void SimulationInterface::SetVelocity(float x_speed, float y_speed, 
    float angular_speed)
{
  using_wheel_speed_ = false;
  x_speed_ = x_speed;
  y_speed_ = y_speed;
  angular_speed_ = angular_speed;
}

/* set the robot velocity by setting the speed of robot wheels */
void SimulationInterface::SetVelocity(float front_left_wheel_speed, 
    float back_left_wheel_speed,
    float back_right_wheel_speed, 
    float front_right_wheel_speed)
{
  using_wheel_speed_ = true;
  wheel_1_ = -front_left_wheel_speed;
  wheel_2_ = -back_left_wheel_speed;
  wheel_3_ = back_right_wheel_speed;
  wheel_4_ = front_right_wheel_speed;
}

/* Function to set the velocity of the kicker */
void SimulationInterface::SetKickerSpeed(float kicker_speed)
{
  kicker_speed_ = kicker_speed;
}

/* Function to control spinner */
void SimulationInterface::SetSpinnerOn(bool spinner_on)
{
  spinner_on_ = spinner_on;
}

/* Send a UDP packet with the robot command */
void SimulationInterface::SendPacket()
{
  GrSimPacket packet;

  /* Write the data to the protobuf message */
  packet = CreateProtoPacket();

  /* Send the packet */
  SendPacket(packet);
}

GrSimPacket SimulationInterface::CreateProtoPacket()
{
  GrSimPacket packet;
  GrSimRobotCommand *command;

  /* Write the data to the protobuf message */
  packet.mutable_commands()->set_is_team_yellow(team_ == Team::kYellow);
  packet.mutable_commands()->set_timestamp(0.0L);
  command = packet.mutable_commands()->add_robot_commands();
  command->set_id(id_);
  command->set_kick_speed_x(kicker_speed_);
  command->set_kick_speed_z(0.0F);
  command->set_spinner(spinner_on_);
  command->set_wheels_speed(using_wheel_speed_);
  command->set_wheel_1(wheel_1_);
  command->set_wheel_2(wheel_2_);
  command->set_wheel_3(wheel_3_);
  command->set_wheel_4(wheel_4_);
  command->set_vel_tangent(x_speed_);
  command->set_vel_normal(y_speed_);
  command->set_vel_angular(angular_speed_);

  return packet;
}

/* Send a grSim packet with UDP */
void SimulationInterface::SendPacket(GrSimPacket packet)
{
  size_t size;
  void *buffer;

  /* Serialize the protobuf message before sending */
  size = packet.ByteSizeLong();
  buffer = malloc(size);
  packet.SerializeToArray(buffer, size);

  /* Send the UDP packet*/
  ::sendto(socket_, buffer, size, 0, 
      reinterpret_cast<sockaddr *>(&destination_), sizeof(destination_));

  free(buffer);
}

} /* namespace simulation_interface */
} /* namespace centralised_ai */