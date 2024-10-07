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

/* C system headers */
#include <arpa/inet.h>

/* C++ standard library headers */
#include <string> 

/* Project .h files */
#include "messages_robocup_ssl_detection.pb.h"
#include "messages_robocup_ssl_wrapper.pb.h"
#include "../common_types.h"

namespace centralized_ai
{
namespace ssl_interface
{

const int max_datagram_size = 65536;

class VisionClient
{
public:
  VisionClient(std::string ip, int port);
  virtual void ReceivePacket();
  void Print();

  /* Getters for position data */
  float GetBlueRobotPositionX(int id);
  float GetBlueRobotPositionY(int id);
  float GetBlueRobotOrientation(int id);
  float GetYellowRobotPositionX(int id);
  float GetYellowRobotPositionY(int id);
  float GetYellowRobotOrientation(int id);
  float GetBallPositionX();
  float GetBallPositionY();
 
protected:
  /* Network variables */
  sockaddr_in client_address;
  int socket;

  /* Position data */
  float blue_robot_positions_x[team_size];
  float blue_robot_positions_y[team_size];
  float blue_robot_orientations[team_size];
  float yellow_robot_positions_x[team_size];
  float yellow_robot_positions_y[team_size];
  float yellow_robot_orientations[team_size];
  float ball_position_x;
  float ball_position_y;
};

} /* namespace ssl_interface */
} /* namesapce centralized_ai */

#endif /* CENTRALIZEDAI_SSLVISIONCLIENT_H_ */