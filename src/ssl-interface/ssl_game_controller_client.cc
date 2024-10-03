// ssl_game_controller_client.cc
//==============================================================================
// Author: Emil Åberg, Aaiza A. Khan
// Creation date: 2024-10-01
// Last modified: 2024-10-02 by Emil Åberg
// Description: A client for receiveing game state from ssl game controller
// License: See LICENSE file for license details.
//==============================================================================

// Related .h files
#include "ssl_game_controller_client.h"

// C system headers
#include <arpa/inet.h>

// C++ standard library headers
#include <string> 

// Project .h files
#include "ssl_gc_referee_message.pb.h"

namespace centralized_ai
{
namespace ssl_interface
{

// Constructor
GameControllerClient::GameControllerClient(std::string ip, int port)
{
  // Define client address
  client_address.sin_family = AF_INET;
  client_address.sin_port = htons(port);
  client_address.sin_addr.s_addr = inet_addr(ip.c_str());

  // Create the client socket
  socket = ::socket(AF_INET, SOCK_DGRAM, 0);
     
  // Bind the socket with the client address 
  bind(socket, (const struct sockaddr *)&client_address, sizeof(client_address));
}

// Read a UDP packet from game controller and return the game state
struct GameStateData GameControllerClient::ReceivePacket()
{
  Referee packet;
  int message_length;
  char buffer[max_datagram_size];

  // Receive raw packet
  message_length = recv(socket, (char *)buffer, max_datagram_size,  
              MSG_WAITALL);

  if (message_length > 0)
  {
    // Decode packet
    packet.ParseFromArray(buffer, message_length);

    // Read and store the game state data
    game_state_data = ReadGameStateData(packet);
  }

  return game_state_data;
}

// Return the data of interest in the protobuf packet
struct GameStateData GameControllerClient::ReadGameStateData(Referee packet)
{
  struct GameStateData game_state_data_local;

  game_state_data_local.referee_command = packet.command();
  game_state_data_local.blue_team_score = packet.blue().score();
  game_state_data_local.yellow_team_score = packet.yellow().score();

  return game_state_data_local;
}

// Translate command enumerator to string
std::string GameControllerClient::CommandToString(Referee::Command command)
{
  switch (command)
  {
    case Referee::HALT: return "HALT";
    case Referee::STOP: return "STOP";
    case Referee::NORMAL_START: return "NORMAL_START";
    case Referee::FORCE_START: return "FORCE_START";
    case Referee::PREPARE_KICKOFF_YELLOW: return "PREPARE_KICKOFF_YELLOW";
    case Referee::PREPARE_KICKOFF_BLUE: return "PREPARE_KICKOFF_BLUE";
    case Referee::PREPARE_PENALTY_YELLOW: return "PREPARE_PENALTY_YELLOW";
    case Referee::PREPARE_PENALTY_BLUE: return "PREPARE_PENALTY_BLUE";
    case Referee::DIRECT_FREE_YELLOW: return "DIRECT_FREE_YELLOW";
    case Referee::DIRECT_FREE_BLUE: return "DIRECT_FREE_BLUE";
    case Referee::INDIRECT_FREE_YELLOW: return "INDIRECT_FREE_YELLOW"; // Deprecated
    case Referee::INDIRECT_FREE_BLUE: return "INDIRECT_FREE_BLUE";     // Deprecated
    case Referee::TIMEOUT_YELLOW: return "TIMEOUT_YELLOW";
    case Referee::TIMEOUT_BLUE: return "TIMEOUT_BLUE";
    case Referee::GOAL_YELLOW: return "GOAL_YELLOW"; // Deprecated
    case Referee::GOAL_BLUE: return "GOAL_BLUE";     // Deprecated
    case Referee::BALL_PLACEMENT_YELLOW: return "BALL_PLACEMENT_YELLOW";
    case Referee::BALL_PLACEMENT_BLUE: return "BALL_PLACEMENT_BLUE";
    default: return "UNKNOWN_COMMAND";
  }
}

// Method to print the game state, used for debugging/demo
void GameControllerClient::Print()
{
  printf("referee command:<%s> score: <%i, %i>\n",
    CommandToString(game_state_data.referee_command).c_str(), game_state_data.blue_team_score,
    game_state_data.yellow_team_score);
}

} // namespace ssl_interface
} // namesapce centralized_ai