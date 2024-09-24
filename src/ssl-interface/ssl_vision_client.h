// ssl_vision_client.cc
//==============================================================================
// Author: Aaiza A. Khan and Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 2024-09-24 by Emil Åberg
// Description: A simple client receiving ball and robots positions from ssl-vision
// License: See LICENSE file for license details.
//==============================================================================
#ifndef CENTRALIZEDAI_SSLVISIONCLIENT_H_
#define CENTRALIZEDAI_SSLVISIONCLIENT_H_

// C system headers
#include <arpa/inet.h>

// C++ standard library headers
#include <string> 

// Project .h files
#include "messages_robocup_ssl_detection.pb.h"
#include "messages_robocup_ssl_wrapper.pb.h"

#define TEAM_SIZE 6

struct PositionData
{
  struct RobotPosition
  {
    float x = 0.0F;
    float y = 0.0F;
    float orientation = 0.0F;
  };

  struct BallPosition
  {
    float x = 0.0F;
    float y = 0.0F;
  };

  struct RobotPosition blue_robot_position[TEAM_SIZE];
  struct RobotPosition yellow_robot_position[TEAM_SIZE];
  struct BallPosition ball_position;
};

class VisionClient
{
public:
  VisionClient(std::string ip, int port);
  void ReceivePacket(struct PositionData* position_data);
  void PrintPositionData(struct PositionData position_data);
private:
  sockaddr_in client_address;
  int socket;
  static const int max_datagram_size = 65536;
  socklen_t address_length;
};

#endif // CENTRALIZEDAI_SSLVISIONCLIENT_H_