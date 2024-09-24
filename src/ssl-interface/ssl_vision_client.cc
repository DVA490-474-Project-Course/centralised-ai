// ssl_vision_client.h
//==============================================================================
// Author: Aaiza A. Khan and Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 2024-09-24 by Emil Åberg
// Description: A simple client receiving ball and robots positions from ssl-vision
// License: See LICENSE file for license details.
//==============================================================================

// C system headers
#include "ssl_vision_client.h"
#include <arpa/inet.h> 
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h> 

// C++ standard library headers
#include <string> 

// Project .h files
#include "messages_robocup_ssl_detection.pb.h"
#include "messages_robocup_ssl_wrapper.pb.h"

// Constructor
VisionClient::VisionClient(std::string ip, int port)
{ 
    // Define client address
  client_address.sin_family = AF_INET;
  client_address.sin_port = htons(port);
  client_address.sin_addr.s_addr = inet_addr(ip.c_str());

  // Create the client socket
  socket = ::socket(AF_INET, SOCK_DGRAM, 0);
     
  // Bind the socket with the client address 
  bind(socket, (const struct sockaddr *)&client_address, sizeof(client_address));

  // Save length of client address
  address_length = sizeof(client_address);
}

// Receive one UDP packet and write the data to the output parameter
void VisionClient::ReceivePacket(struct PositionData* position_data)
{
  SSL_WrapperPacket packet;
  SSL_DetectionFrame detection;
  SSL_DetectionRobot robot;
  SSL_DetectionBall ball;
  int id;
    int n;
    char buffer[max_datagram_size];
  struct PositionData data;

  // Receive raw packet
    n = recvfrom(socket, (char *)buffer, max_datagram_size,  
                MSG_WAITALL, (struct sockaddr *) &client_address, 
                &address_length);

  if (n > 0)
  {
    // Decode packet
    packet.ParseFromArray(buffer, n);

    if (packet.has_detection())
    {
      detection = packet.detection();

      // Read positions of blue robots
      for (int i = 0; i < detection.robots_blue_size(); ++i)
      {
        robot = detection.robots_blue(i);
        id = robot.robot_id();

        if (id < TEAM_SIZE)
        {
          position_data->blue_robot_position[id].x = robot.x();
          position_data->blue_robot_position[id].y = robot.y();
          if (robot.has_orientation())
          {
            position_data->blue_robot_position[id].orientation = robot.orientation();
          }
        }
      }

      // Read positions of yellow robots
      for (int i = 0; i < detection.robots_yellow_size(); ++i)
      {
        robot = detection.robots_yellow(i);
        id = robot.robot_id();

        if (id < TEAM_SIZE)
        {
          position_data->yellow_robot_position[id].x = robot.x();
          position_data->yellow_robot_position[id].y = robot.y();
          if (robot.has_orientation())
          {
            position_data->yellow_robot_position[id].orientation = robot.orientation();
          }
        }
      }

      // Read ball position
      if (detection.balls_size() > 0)
      {
        ball = detection.balls(0);    // Assume only one ball is in play
        position_data->ball_position.x = ball.x();
        position_data->ball_position.y = ball.y();
      }
    }
  }
}

// Method to print position data, used for debugging/demo
void VisionClient::PrintPositionData(struct PositionData position_data)
{
  for (int id = 0; id < TEAM_SIZE; id++)
  {
    printf("BLUE ROBOT ID=<%d> POS=<%9.2f,%9.2f> ROT=<%9.2f>  ", id,
      position_data.blue_robot_position[id].x,
      position_data.blue_robot_position[id].y,
      position_data.blue_robot_position[id].orientation);
    printf("YELLOW ROBOT ID=<%d> POS=<%9.2f,%9.2f> ROT=<%9.2f>\n", id,
      position_data.yellow_robot_position[id].x,
      position_data.yellow_robot_position[id].y,
      position_data.yellow_robot_position[id].orientation);
  }
  
  printf("BALL POS=<%9.2f,%9.2f> \n\n",
    position_data.ball_position.x,
    position_data.ball_position.y);
}