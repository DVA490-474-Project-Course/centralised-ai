/* ssl_vision_client.h
 *==============================================================================
 * Author: Aaiza A. Khan, Shruthi Puthiya Kunnon, Emil Åberg
 * Creation date: 2024-09-20
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: A simple client receiving ball and robots positions from ssl-vision
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "ssl_vision_client.h"

/* C system headers */
#include "arpa/inet.h" 
#include "netinet/in.h"
#include "stdio.h"
#include "sys/socket.h" 

/* C++ standard library headers */
#include "string" 

/* Project .h files */
#include "generated/ssl_vision_detection.pb.h"
#include "generated/ssl_vision_wrapper.pb.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

/* Constructor */
VisionClient::VisionClient(std::string ip, int port)
{
  /* Define client address */
  client_address_.sin_family = AF_INET;
  client_address_.sin_port = htons(port);
  client_address_.sin_addr.s_addr = inet_addr(ip.c_str());

  /* Create the client socket */
  socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
     
  /* Bind the socket with the client address */
  bind(socket_, (const struct sockaddr *)&client_address_, sizeof(client_address_));
}

/* Receive one UDP packet and write the data to the output parameter */
void VisionClient::ReceivePacket()
{
  SslWrapperPacket packet;
  int message_length;
  char buffer[kMaxDatagramSize];

  /* Receive raw packet */
  message_length = recv(socket_, (char *)buffer, kMaxDatagramSize, MSG_WAITALL);

  if (message_length > 0)
  {
    /* Decode packet */
    packet.ParseFromArray(buffer, message_length);

    /* Read data from packet */
    ReadVisionData(packet);
  }
}

/* Receive packets until all positions have been read at least once */
void VisionClient::ReceivePacketsUntilAllDataRead()
{
  bool all_data_has_been_read;

  /* Set flags indicating that ball and robot positions have not been read yet */
  ball_data_read_ = false;
  for (int id = 0; id < team_size; id++)
  {
    blue_robot_positions_read_[id] = false;
    yellow_robot_positions_read_[id] = false;
  }

  do
  {
    ReceivePacket();
    all_data_has_been_read = true;

    for (int id = 0; id < team_size; id++)
    {
      if (blue_robot_positions_read_[id] == false ||
          yellow_robot_positions_read_[id] == false ||
          ball_data_read_ == false)
      {
        all_data_has_been_read = false;
        break;
      }
    }
  }
  while (all_data_has_been_read == false);
}

void VisionClient::ReadVisionData(SslWrapperPacket packet)
{
  SslDetectionFrame detection;
  SslDetectionRobot robot;
  SslDetectionBall ball;
  int id;

  if (packet.has_detection())
  {
    detection = packet.detection();

    /* Read positions of blue robots */
    for (int i = 0; i < detection.robots_blue_size(); ++i)
    {
      robot = detection.robots_blue(i);
      id = robot.robot_id();

      if (id < team_size)
      {
        blue_robot_positions_x_[id] = robot.x();
        blue_robot_positions_y_[id] = robot.y();
        if (robot.has_orientation())
        {
          blue_robot_orientations_[id] = robot.orientation();
          blue_robot_positions_read_[id] = true;
        }
      }
    }

    /* Read positions of yellow robots */
    for (int i = 0; i < detection.robots_yellow_size(); ++i)
    {
      robot = detection.robots_yellow(i);
      id = robot.robot_id();

      if (id < team_size)
      {
        yellow_robot_positions_x_[id] = robot.x();
        yellow_robot_positions_y_[id] = robot.y();
        if (robot.has_orientation())
        {
          yellow_robot_orientations_[id] = robot.orientation();
          yellow_robot_positions_read_[id] = true;
        }
      }
    }

    /* Read ball position */
    if (detection.balls_size() > 0)
    {
      ball = detection.balls(0);    // Assume only one ball is in play
      ball_position_x_ = ball.x();
      ball_position_y_ = ball.y();
      ball_data_read_ = true;
    }
  }

  /* Get timestamp */
  timestamp_ = detection.t_capture();
}

/* Method to print position data, used for debugging/demo */
void VisionClient::Print()
{
  for (int id = 0; id < team_size; id++)
  {
    printf("BLUE ROBOT ID=<%d> POS=<%9.2f,%9.2f> ROT=<%9.2f>  ", id,
        blue_robot_positions_x_[id],
        blue_robot_positions_y_[id],
        blue_robot_orientations_[id]);
    printf("YELLOW ROBOT ID=<%d> POS=<%9.2f,%9.2f> ROT=<%9.2f>\n", id,
        yellow_robot_positions_x_[id],
        yellow_robot_positions_y_[id],
        yellow_robot_orientations_[id]);
  }
  
  printf("BALL POS=<%9.2f,%9.2f> TIME=<%f> \n\n",
      ball_position_x_,
      ball_position_y_,
      timestamp_);
}

double VisionClient::GetTimestamp()
{
  return timestamp_;
}

float VisionClient::GetRobotPositionX(int id, enum Team team)
{
  if (team == Team::kBlue)
  {
    return blue_robot_positions_x_[id];
  }
  else if (team == Team::kYellow)
  {
    return yellow_robot_positions_x_[id];
  }
  else
  {
    std::perror("GetRobotPositionX called with unknown team.");
    return 0.0F;
  }
}

float VisionClient::GetRobotPositionY(int id, enum Team team)
{
  if (team == Team::kBlue)
  {
    return blue_robot_positions_y_[id];
  }
  else if (team == Team::kYellow)
  {
    return yellow_robot_positions_y_[id];
  }
  else
  {
    std::perror("GetRobotPositionY called with unknown team.");
    return 0.0F;
  }
}

float VisionClient::GetRobotOrientation(int id, enum Team team)
{
  if (team == Team::kBlue)
  {
    return blue_robot_orientations_[id];
  }
  else if (team == Team::kYellow)
  {
    return yellow_robot_orientations_[id];
  }
  else
  {
    std::perror("GetRobotOrientation called with unknown team.");
    return 0.0F;
  }
}

float VisionClient::GetBallPositionX()
{
  return ball_position_x_;
}

float VisionClient::GetBallPositionY()
{
  return ball_position_y_;
}

} /* namespace ssl_interface */
} /* namesapce centralised_ai */
