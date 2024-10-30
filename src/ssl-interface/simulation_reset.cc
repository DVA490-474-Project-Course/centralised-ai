/* simulation_reset.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-10-21
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Provides function to reset robots and ball in grSim
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "simulation_reset.h"

/* C system headers */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

/* C++ standard library headers */
#include <memory>
#include <string>

/* Project .h files */
#include "generated/grSim_Commands.pb.h"
#include "generated/grSim_Packet.pb.h"

namespace centralised_ai
{
namespace ssl_interface
{

/* Initial position of yellow robots, for blue x values will have opposite sign */
double initial_position_x[6] = {1.50, 1.50, 1.50, 0.55, 2.50, 3.60};
double initial_position_y[6] = {1.12, 0.0, -1.12, 0.00, 0.00, 0.00};

/* Send a grSim packet with UDP */
void SendPacket(grSim_Packet packet, std::string ip, uint16_t port)
{
  size_t size;
  void *buffer;
  int socket;
  sockaddr_in destination;

  /* Define destination address */
  destination.sin_family = AF_INET;
  destination.sin_port = htons(port);
  destination.sin_addr.s_addr = inet_addr(ip.c_str());

  /* Create the socket */
  socket = ::socket(AF_INET, SOCK_DGRAM, 0);

  /* Serialize the protobuf message before sending */
  size = packet.ByteSizeLong();
  buffer = (char *)malloc(size);
  packet.SerializeToArray(buffer, size);

  /* Send the UDP packet*/
  ::sendto(socket, buffer, size, 0, reinterpret_cast<sockaddr *>(&destination),
           sizeof(destination));

  free(buffer);
}

/* Reset ball and all robots position and other attributes */
void ResetRobotsAndBall(std::string ip, uint16_t port)
{
  grSim_Packet packet;
  grSim_Robot_Command *command;
  grSim_RobotReplacement *replacement;
  grSim_BallReplacement *ball_replacement;

  packet.mutable_commands()->set_isteamyellow(false); /* Set to false for blue team */
  packet.mutable_commands()->set_timestamp(0.0L);

  /* Loop through each robot index to reset the positions and other attributes
  of blue and yellow team*/
  for (int k = 0; k < 6; k++)
  {
    /* Reset blue team robots (yellowteam = false) */
    command = packet.mutable_commands()->add_robot_commands();
    command->set_id(k);
    command->set_wheelsspeed(false);
    command->set_veltangent(0.0F); /* Stop all movement */
    command->set_velnormal(0.0F);  /* Stop all movement */
    command->set_velangular(0.0F); /* Stop angular movement */
    command->set_kickspeedx(0.0F); /* No kick speed */
    command->set_kickspeedz(0.0F); /* No kick in Z direction */
    command->set_spinner(false);   /* Turn off the spinner */

    /* Set up the replacement packet for blue team */
    replacement = packet.mutable_replacement()->add_robots();
    replacement->set_id(k);
    replacement->set_x(-initial_position_x[k]);          /* Set new x position */
    replacement->set_y(initial_position_y[k]);           /* Set new y position */
    replacement->set_dir(0.0F);         /* Set direction (angle in radians) */
    replacement->set_yellowteam(false); /* Set to blue team (yellowteam = false) */

    /* Reset yellow team robots (yellowteam = true) */
    command = packet.mutable_commands()->add_robot_commands();
    command->set_id(k);
    command->set_wheelsspeed(false);
    command->set_veltangent(0.0F); /* Stop all movement */
    command->set_velnormal(0.0F);  /* Stop all movement */
    command->set_velangular(0.0F); /* Stop angular movement */
    command->set_kickspeedx(0.0F); /* No kick speed */
    command->set_kickspeedz(0.0F); /* No kick in Z direction */
    command->set_spinner(false);   /* Turn off the spinner */

    /* Set up the replacement packet for yellow team */
    replacement = packet.mutable_replacement()->add_robots();
    replacement->set_id(k);
    replacement->set_x(initial_position_x[k]);
    replacement->set_y(initial_position_y[k]);
    replacement->set_dir(0.0F);
    replacement->set_yellowteam(true);
  }

  /* Replacement packet for ball */
  ball_replacement = packet.mutable_replacement()->mutable_ball();
  ball_replacement->set_x(0.0F);
  ball_replacement->set_y(0.0F);
  ball_replacement->set_vx(0.0F);
  ball_replacement->set_vy(0.0F);

  /* Send the packet */
  SendPacket(packet, ip, port);
}

} /* namespace ssl_interface */
} /* namespace centralised_ai */