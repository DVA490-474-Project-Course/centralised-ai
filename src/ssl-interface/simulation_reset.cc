/* simulation_reset.cc
 *==============================================================================
 * Author: Shruthi P. Kunnon
 * Creation date: 2024-10-21
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Provides function to reset robots and ball in grSim
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "simulation_reset.h"

/* C system headers */
#include "arpa/inet.h"
#include "netinet/in.h"
#include "sys/socket.h"

/* C++ standard library headers */
#include "memory"
#include "string"

/* Project .h files */
#include "generated/grsim_commands.pb.h"
#include "generated/grsim_packet.pb.h"

namespace centralised_ai
{
namespace ssl_interface
{

/* Initial positions of robots for both teams.
 * These are defined globally to avoid redundancy and ensure consistent starting
 * positions for both yellow and blue teams. For the blue team, the x-values
 * have opposite sign to place them on the opposite side of the field.
 */
constexpr double initial_position_x[6] = {1.50, 1.50, 1.50, 0.55, 2.50, 3.60};
constexpr double initial_position_y[6] = {1.12, 0.0, -1.12, 0.00, 0.00, 0.00};

/* Send a grSim packet with UDP */
void SendPacket(GrSimPacket packet, std::string ip, uint16_t port)
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
void ResetRobotsAndBall(std::string ip, uint16_t port,
    enum Team team_on_positive_half)
{
  GrSimPacket packet;
  GrSimRobotCommand *command;
  GrSimRobotReplacement *replacement;
  GrSimBallReplacement *ball_replacement;

  /* Set to false for blue team */
  packet.mutable_commands()->set_is_team_yellow(false);
  packet.mutable_commands()->set_timestamp(0.0L);

  /* Loop through each robot index to reset the positions and other attributes
  of blue and yellow team*/
  for (int k = 0; k < team_size; k++)
  {
    /* Reset blue team robots (yellowteam = false) */
    command = packet.mutable_commands()->add_robot_commands();
    command->set_id(k);
    command->set_wheels_speed(false);
    command->set_vel_tangent(0.0F); /* Stop all movement */
    command->set_vel_normal(0.0F);  /* Stop all movement */
    command->set_vel_angular(0.0F); /* Stop angular movement */
    command->set_kick_speed_x(0.0F); /* No kick speed */
    command->set_kick_speed_z(0.0F); /* No kick in Z direction */
    command->set_spinner(false);   /* Turn off the spinner */

    /* Set up the replacement packet for blue team */
    replacement = packet.mutable_replacement()->add_robots();
    replacement->set_id(k);
    if (team_on_positive_half == Team::kYellow)
    {
      /* Set new x position */
      replacement->set_x(-initial_position_x[k]);
      replacement->set_dir(0.0F);
    }
    else
    {
      /* Set new x position */
      replacement->set_x(initial_position_x[k]);
      replacement->set_dir(180.0F);
    }
    /* Set new y position */
    replacement->set_y(initial_position_y[k]);
    /* Set to blue team (yellowteam = false) */
    replacement->set_yellow_team(false);

    /* Reset yellow team robots (yellowteam = true) */
    command = packet.mutable_commands()->add_robot_commands();
    command->set_id(k);
    command->set_wheels_speed(false);
    command->set_vel_tangent(0.0F); /* Stop all movement */
    command->set_vel_normal(0.0F);  /* Stop all movement */
    command->set_vel_angular(0.0F); /* Stop angular movement */
    command->set_kick_speed_x(0.0F); /* No kick speed */
    command->set_kick_speed_z(0.0F); /* No kick in Z direction */
    command->set_spinner(false);   /* Turn off the spinner */

    /* Set up the replacement packet for yellow team */
    replacement = packet.mutable_replacement()->add_robots();
    replacement->set_id(k);
    if (team_on_positive_half == Team::kYellow)
    {
      /* Set new x position */
      replacement->set_x(initial_position_x[k]);
      replacement->set_dir(180.0F);
    }
    else
    {
      /* Set new x position */
      replacement->set_x(-initial_position_x[k]);
      replacement->set_dir(0.0F);
    }
    replacement->set_y(initial_position_y[k]);
    replacement->set_yellow_team(true);
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