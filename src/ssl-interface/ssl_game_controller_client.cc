// ssl_game_controller_client.cc
//==============================================================================
// Author: Emil Åberg
// Creation date: 2024-10-01
// Last modified: 2024-10-01 by Emil Åberg
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

  // Save length of client address
  address_length = sizeof(client_address);
}

// Read an UDP packet from game controller and output the game state
struct GameStateData GameControllerClient::ReceivePacket()
{
  Referee packet;
  int message_length;
  char buffer[max_datagram_size];

  // Receive raw packet
  message_length = recvfrom(socket, (char *)buffer, max_datagram_size,  
              MSG_WAITALL, (struct sockaddr *) &client_address, 
              &address_length);

  if (message_length > 0)
  {
    // Decode packet
    packet.ParseFromArray(buffer, message_length);

    // Read game state data
    ReadGameStateData(packet);
  }

  return game_state_data_local;
}

// Read the data we want from the protobuf packet
void GameControllerClient::ReadGameStateData(Referee packet)
{
  game_state_data_local.referee_command = packet.command();
  game_state_data_local.blue_team_score = packet.blue().score();
  game_state_data_local.yellow_team_score = packet.yellow().score();
}

// Method to print the game state, used for debugging/demo
void GameControllerClient::Print()
{
  printf("referee command:<%i> score: <%i, %i>\n",
    (int)game_state_data_local.referee_command, game_state_data_local.blue_team_score,
    game_state_data_local.yellow_team_score);
}