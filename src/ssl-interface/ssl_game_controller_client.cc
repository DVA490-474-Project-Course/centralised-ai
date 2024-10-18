/* ssl_game_controller_client.cc
 *==============================================================================
 * Author: Emil Åberg, Aaiza A. Khan
 * Creation date: 2024-10-01
 * Last modified: 2024-10-10 by Emil Åberg
 * Description: A client for receiveing game state from ssl game controller
 * License: See LICENSE file for license details.
 *=============================================================================
 */

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

  if (packet.has_designated_position())
  {
    ball_designated_position_x = packet.designated_position().x();
    ball_designated_position_y = packet.designated_position().y();
  }

  if (packet.has_blue_team_on_positive_half())
  {
    blue_team_on_positive_half = packet.blue_team_on_positive_half();
  }

  if (packet.has_next_command())
  {
    next_referee_command = ConvertRefereeCommand(packet.next_command());
  }
  else
  {
    next_referee_command = RefereeCommand::UNKNOWN_COMMAND;
  }

  if (packet.has_stage_time_left())
  {
    stage_time_left = packet.stage_time_left();
  }
}

/* Convert from protobuf enum definition to project enum definition */
enum RefereeCommand GameControllerClient::ConvertRefereeCommand(enum Referee_Command command)
{
  switch (command)
  {
    case Referee::HALT: return RefereeCommand::HALT;
    case Referee::STOP: return RefereeCommand::STOP;
    case Referee::NORMAL_START: return RefereeCommand::NORMAL_START;
    case Referee::FORCE_START: return RefereeCommand::FORCE_START;
    case Referee::PREPARE_KICKOFF_YELLOW: return RefereeCommand::PREPARE_KICKOFF_YELLOW;
    case Referee::PREPARE_KICKOFF_BLUE: return RefereeCommand::PREPARE_KICKOFF_BLUE;
    case Referee::PREPARE_PENALTY_YELLOW: return RefereeCommand::PREPARE_PENALTY_YELLOW;
    case Referee::PREPARE_PENALTY_BLUE: return RefereeCommand::PREPARE_PENALTY_BLUE;
    case Referee::DIRECT_FREE_YELLOW: return RefereeCommand::DIRECT_FREE_YELLOW;
    case Referee::DIRECT_FREE_BLUE: return RefereeCommand::DIRECT_FREE_BLUE;
    case Referee::TIMEOUT_YELLOW: return RefereeCommand::TIMEOUT_YELLOW;
    case Referee::TIMEOUT_BLUE: return RefereeCommand::TIMEOUT_BLUE;
    case Referee::BALL_PLACEMENT_YELLOW: return RefereeCommand::BALL_PLACEMENT_YELLOW;
    case Referee::BALL_PLACEMENT_BLUE: return RefereeCommand::BALL_PLACEMENT_BLUE;
    default: return RefereeCommand::UNKNOWN_COMMAND;
  }
}

/* Translate RefereeCommand enumerator to string */
std::string GameControllerClient::RefereeCommandToString(RefereeCommand referee_command)
{
  switch (referee_command)
  {
    case RefereeCommand::HALT: return "HALT";
    case RefereeCommand::STOP: return "STOP";
    case RefereeCommand::NORMAL_START: return "NORMAL_START";
    case RefereeCommand::FORCE_START: return "FORCE_START";
    case RefereeCommand::PREPARE_KICKOFF_YELLOW: return "PREPARE_KICKOFF_YELLOW";
    case RefereeCommand::PREPARE_KICKOFF_BLUE: return "PREPARE_KICKOFF_BLUE";
    case RefereeCommand::PREPARE_PENALTY_YELLOW: return "PREPARE_PENALTY_YELLOW";
    case RefereeCommand::PREPARE_PENALTY_BLUE: return "PREPARE_PENALTY_BLUE";
    case RefereeCommand::DIRECT_FREE_YELLOW: return "DIRECT_FREE_YELLOW";
    case RefereeCommand::DIRECT_FREE_BLUE: return "DIRECT_FREE_BLUE";
    case RefereeCommand::TIMEOUT_YELLOW: return "TIMEOUT_YELLOW";
    case RefereeCommand::TIMEOUT_BLUE: return "TIMEOUT_BLUE";
    case RefereeCommand::BALL_PLACEMENT_YELLOW: return "BALL_PLACEMENT_YELLOW";
    case RefereeCommand::BALL_PLACEMENT_BLUE: return "BALL_PLACEMENT_BLUE";
    default: return "UNKNOWN_COMMAND";
  }
}

/* Method to print the game state, used for debugging/demo */
void GameControllerClient::Print()
{
  printf("referee command: <%s> next command: <%s> score: <%i, %i> designated position <%f, %f> stage time left: <%li>\n",
    RefereeCommandToString(referee_command).c_str(),
    RefereeCommandToString(next_referee_command).c_str(),
    blue_team_score, yellow_team_score, ball_designated_position_x, ball_designated_position_y,
    stage_time_left);
}

/* Public getters */
enum RefereeCommand GameControllerClient::GetRefereeCommand() {return referee_command;}
enum RefereeCommand GameControllerClient::GetNextRefereeCommand() {return next_referee_command;}
int GameControllerClient::GetBlueTeamScore() {return blue_team_score;}
int GameControllerClient::GetYellowTeamScore() {return yellow_team_score;}
float GameControllerClient::GetBallDesignatedPositionX() {return blue_team_score;}
float GameControllerClient::GetBallDesignatedPositionY() {return yellow_team_score;}
int64_t GameControllerClient::GetStageTimeLeft() {return stage_time_left;}
bool GameControllerClient::BlueTeamOnPositiveHalf() {return blue_team_on_positive_half;}

} /* namespace ssl_interface */
} /* namesapce centralized_ai */