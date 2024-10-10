// ssl_game_controller_client.h
//==============================================================================
// Author: Emil Åberg, Aaiza A. Khan
// Creation date: 2024-10-01
// Last modified: 2024-10-02 by Emil Åberg
// Description: A client for receiveing game state from ssl game controller
// License: See LICENSE file for license details.
//==============================================================================

#ifndef CENTRALIZEDAI_SSLGMAECONTROLLERCLIENT_H_
#define CENTRALIZEDAI_SSLGAMECONTROLLERCLIENT_H_

/* C system headers */
#include <arpa/inet.h>

/* C++ standard library headers */
#include <string> 

/* Project .h files */
#include "ssl_gc_referee_message.pb.h"
#include "../common_types.h"

namespace centralized_ai
{
namespace ssl_interface
{

const int max_datagram_size = 65536;

class GameControllerClient
{
public:
  GameControllerClient(std::string ip, int port);
  void ReceivePacket();
  void Print();

  /* Geters for game state data */
  enum RefereeCommand GetRefereeCommand();
  int GetBlueTeamScore();
  int GetYellowTeamScore();
  float GetBallDesignatedPositionX();
  float GetBallDesignatedPositionY();

protected:
  std::string RefereeCommandToString(enum RefereeCommand referee_command);
  void ReadGameStateData(Referee packet);
  enum RefereeCommand ConvertRefereeCommand(enum Referee_Command command);

  /* Network variables */
  sockaddr_in client_address;
  int socket;

  /* Game state data */
  enum RefereeCommand referee_command;
  int blue_team_score;
  int yellow_team_score;
  float ball_designated_position_x;
  float ball_designated_position_y;
};

} /* namespace ssl_interface */
} /* namesapce centralized_ai */

#endif /* CENTRALIZEDAI_SSLGAMECONTROLLERCLIENT_H_ */