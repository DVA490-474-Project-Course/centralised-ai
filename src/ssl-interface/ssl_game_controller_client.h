// ssl_game_controller_client.h
//==============================================================================
// Author: Emil Åberg
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Emil Åberg
// Description: A client for receiveing game state from ssl game controller
// License: See LICENSE file for license details.
//==============================================================================

#ifndef CENTRALIZEDAI_SSLGMAECONTROLLERCLIENT_H_
#define CENTRALIZEDAI_SSLGAMECONTROLLERCLIENT_H_

// C system headers
#include <arpa/inet.h>

// C++ standard library headers
#include <string> 

// Project .h files
#include "ssl_gc_referee_message.pb.h"

struct GameStateData
{
  enum Referee_Command referee_command;
  int blue_team_score;
  int yellow_team_score;
};

class GameControllerClient
{
public:
  GameControllerClient(std::string ip, int port);
  struct GameStateData ReceivePacket();
  void Print();

protected:
  void ReadGameStateData(Referee packet);
  sockaddr_in client_address;
  int socket;
  const int max_datagram_size = 65536;
  socklen_t address_length;
  struct GameStateData game_state_data_local;
};

#endif // CENTRALIZEDAI_SSLGAMECONTROLLERCLIENT_H_