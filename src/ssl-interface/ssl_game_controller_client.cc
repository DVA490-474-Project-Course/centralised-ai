/* ssl_game_controller_client.cc
 *==============================================================================
 * Author: Emil Åberg, Aaiza A. Khan
 * Creation date: 2024-10-01
 * Last modified: 2024-10-30 by Emil Åberg
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
#include "../ssl-interface/referee_command_functions.h"
#include "../ssl-interface/generated/ssl_gc_referee_message.pb.h"
#include "../common_types.h"

namespace centralised_ai
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

  /* Set initial values for game state data */
  referee_command = RefereeCommand::kUnknownCommand;
  next_referee_command = RefereeCommand::kUnknownCommand;
  blue_team_score = 0;
  yellow_team_score = 0;
  stage_time_left = 0;
  ball_designated_position_x = 0.0F;
  ball_designated_position_y = 0.0F;
  Team team_on_positive_half = Team::kUnknown;
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
    if (packet.blue_team_on_positive_half())
    {
      team_on_positive_half = Team::kBlue;
    }
    else
    {
      team_on_positive_half = Team::kYellow;
    }
  }

  if (packet.has_next_command())
  {
    next_referee_command = ConvertRefereeCommand(packet.next_command());
  }
  else
  {
    next_referee_command = RefereeCommand::kUnknownCommand;
  }

  if (packet.has_stage_time_left())
  {
    stage_time_left = packet.stage_time_left();
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
int GameControllerClient::GetBlueTeamScore() {return blue_team_score;}
int GameControllerClient::GetYellowTeamScore() {return yellow_team_score;}
float GameControllerClient::GetBallDesignatedPositionX() {return ball_designated_position_x;}
float GameControllerClient::GetBallDesignatedPositionY() {return ball_designated_position_y;}
int64_t GameControllerClient::GetStageTimeLeft() {return stage_time_left;}
enum Team GameControllerClient::GetTeamOnPositiveHalf() {return team_on_positive_half;}
/* enum RefereeCommand GameControllerClient::GetNextRefereeCommand() {return next_referee_command;} */

} /* namespace ssl_interface */
} /* namesapce centralised_ai */