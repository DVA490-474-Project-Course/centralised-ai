// ssl_game_controller_client.cc
//==============================================================================
// Author: Emil Åberg, Aaiza A. Khan
// Creation date: 2024-10-01
// Last modified: 2024-10-02 by Emil Åberg
// Description: A client for receiveing game state from ssl game controller
// License: See LICENSE file for license details.
//==============================================================================

/* Related .h files */
#include "ssl_game_controller_client.h"

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

/* Constructor */
GameControllerClient::GameControllerClient(std::string ip, int port)
{
  /* Define client address */
  client_address.sin_family = AF_INET;
  client_address.sin_port = htons(port);
  client_address.sin_addr.s_addr = inet_addr(ip.c_str());

  /* Create the client socket */
  socket = ::socket(AF_INET, SOCK_DGRAM, 0);
     
  /* Bind the socket with the client address */
  bind(socket, (const struct sockaddr *)&client_address, sizeof(client_address));
}

/* Read a UDP packet from game controller and return the game state */
void GameControllerClient::ReceivePacket()
{
  Referee packet;
  int message_length;
  char buffer[max_datagram_size];

  /* Receive raw packet */
  message_length = recv(socket, (char *)buffer, max_datagram_size,  
              MSG_WAITALL);

  if (message_length > 0)
  {
    /* Decode packet */
    packet.ParseFromArray(buffer, message_length);

    /* Read and store the game state data */
    ReadGameStateData(packet);
  }
}

/* Read and store the data we are interested in from the protobuf message */
void GameControllerClient::ReadGameStateData(Referee packet)
{
  referee_command = ConvertRefereeCommand(packet.command());
  blue_team_score = packet.blue().score();
  yellow_team_score = packet.yellow().score();
}

/* Getter for referee command */
enum RefereeCommand GameControllerClient::GetRefereeCommand()
{
  return referee_command;
}

/* Getter for blue team score */
int GameControllerClient::GetBlueTeamScore()
{
  return blue_team_score;
}

/* Getter for yellow team score */
int GameControllerClient::GetYellowTeamScore()
{
  return yellow_team_score;
}

/* Convert enum Referee::Command (protobuf definition) to enum RefereeCommand enum (our definition) */
enum centralized_ai::RefereeCommand ConvertRefereeCommand(Referee::Command command)
{
  switch (command)
  {
    case Referee::HALT: return centralized_ai::RefereeCommand::HALT;
    case Referee::STOP: return centralized_ai::RefereeCommand::STOP;
    case Referee::NORMAL_START: return centralized_ai::RefereeCommand::NORMAL_START;
    case Referee::FORCE_START: return centralized_ai::RefereeCommand::FORCE_START;
    case Referee::PREPARE_KICKOFF_YELLOW: return centralized_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW;
    case Referee::PREPARE_KICKOFF_BLUE: return centralized_ai::RefereeCommand::PREPARE_KICKOFF_BLUE;
    case Referee::PREPARE_PENALTY_YELLOW: return centralized_ai::RefereeCommand::PREPARE_PENALTY_YELLOW;
    case Referee::PREPARE_PENALTY_BLUE: return centralized_ai::RefereeCommand::PREPARE_PENALTY_BLUE;
    case Referee::DIRECT_FREE_YELLOW: return centralized_ai::RefereeCommand::DIRECT_FREE_YELLOW;
    case Referee::DIRECT_FREE_BLUE: return centralized_ai::RefereeCommand::DIRECT_FREE_BLUE;
    case Referee::TIMEOUT_YELLOW: return centralized_ai::RefereeCommand::TIMEOUT_YELLOW;
    case Referee::TIMEOUT_BLUE: return centralized_ai::RefereeCommand::TIMEOUT_BLUE;
    case Referee::BALL_PLACEMENT_YELLOW: return centralized_ai::RefereeCommand::BALL_PLACEMENT_YELLOW;
    case Referee::BALL_PLACEMENT_BLUE: return centralized_ai::RefereeCommand::BALL_PLACEMENT_BLUE;
    default: return centralized_ai::RefereeCommand::UNKNOWN_COMMAND;
  }
}

/* Translate RefereeCommand enumerator to string */
std::string GameControllerClient::RefereeCommandToString(centralized_ai::RefereeCommand referee_command)
{
  switch (referee_command)
  {
    case centralized_ai::RefereeCommand::HALT: return "HALT";
    case centralized_ai::RefereeCommand::STOP: return "STOP";
    case centralized_ai::RefereeCommand::NORMAL_START: return "NORMAL_START";
    case centralized_ai::RefereeCommand::FORCE_START: return "FORCE_START";
    case centralized_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW: return "PREPARE_KICKOFF_YELLOW";
    case centralized_ai::RefereeCommand::PREPARE_KICKOFF_BLUE: return "PREPARE_KICKOFF_BLUE";
    case centralized_ai::RefereeCommand::PREPARE_PENALTY_YELLOW: return "PREPARE_PENALTY_YELLOW";
    case centralized_ai::RefereeCommand::PREPARE_PENALTY_BLUE: return "PREPARE_PENALTY_BLUE";
    case centralized_ai::RefereeCommand::DIRECT_FREE_YELLOW: return "DIRECT_FREE_YELLOW";
    case centralized_ai::RefereeCommand::DIRECT_FREE_BLUE: return "DIRECT_FREE_BLUE";
    case centralized_ai::RefereeCommand::TIMEOUT_YELLOW: return "TIMEOUT_YELLOW";
    case centralized_ai::RefereeCommand::TIMEOUT_BLUE: return "TIMEOUT_BLUE";
    case centralized_ai::RefereeCommand::BALL_PLACEMENT_YELLOW: return "BALL_PLACEMENT_YELLOW";
    case centralized_ai::RefereeCommand::BALL_PLACEMENT_BLUE: return "BALL_PLACEMENT_BLUE";
    default: return "UNKNOWN_COMMAND";
  }
}

/* Method to print the game state, used for debugging/demo */
void GameControllerClient::Print()
{
  printf("referee command:<%s> score: <%i, %i>\n",
    RefereeCommandToString(referee_command).c_str(), blue_team_score, yellow_team_score);
}

} /* namespace ssl_interface */
} /* namesapce centralized_ai */