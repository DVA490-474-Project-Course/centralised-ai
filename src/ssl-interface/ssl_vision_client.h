// ssl_vision_client.h
//==============================================================================
// Author: Aaiza A. Khan, Shruthi Puthiya Kunnon, Emil Åberg
// Creation date: 2024-09-20
// Last modified: 2024-10-01 by Shruthi Puthiya Kunnon
// Description: Added virtual to Receivepacket function for test suite no.3.
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
  virtual void ReceivePacket(struct PositionData* position_data);
  void PrintPositionData(struct PositionData position_data);

protected:
  sockaddr_in client_address;
  int socket;
  static const int max_datagram_size = 65536;
  socklen_t address_length;
};

#endif // CENTRALIZEDAI_SSLVISIONCLIENT_H_