/* ssl_game_controller_client.h
 *==============================================================================
 * Author: Emil Åberg, Aaiza A. Khan
 * Creation date: 2024-10-01
 * Last modified: 2024-10-10 by Emil Åberg
 * Description: A client for receiveing game state from ssl game controller
 * License: See LICENSE file for license details.
 *=============================================================================
 */

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

/*!
 * @brief Class for communicating with ssl game controller.
 * 
 * Class that allows communication with ssl game controller and
 * provides methods to read game state variables including score
 * referee command, remaining stage time and ball designated position
 */
class GameControllerClient
{
public:
  /*!
    * @brief Constructor that sets up connection to ssl game controller.
    *
    * @param[in] ip IP of the publish address that the game controller is
    * configured to. This can be found in ssl-game-controller.yaml configuration
    * file of the game controller. When running on the game controller on
    * the same computer as this client, this value should be set to localhost
    * i.e. "127.0.0.1".
    *
    * @param[in] port The port of the publish address that the game controller is
    * configured to. This can be found in ssl-game-controller.yaml configuration
    * file of the game controller.
    */
  GameControllerClient(std::string ip, int port);

  /*!
    * @brief Reads a UDP packet from ssl game controller.
    * 
    * Reads a UDP packet from ssl game controller, and updates all
    * game state values that are available in the client.
    * 
    * @warning This method is blocking until a UDP packet has been received,
    * potentially introducing a delay in whatever other task the calling thread
    * is doing. It is recommended to continously run this method in a thread
    * separate from where the Get* functions are called.
    */
  void ReceivePacket();

  /*!
    * @brief Prints the game controller data that has been read by this client.
    * 
    * Prints the game controller data that has been read by this client including
    * referee command, next referee command, score, ball designated position,
    * remaining stage time and which team has been assigned to the positive half
    * of the field. Used for debugging purpuses.
    */
  void Print();

  /*!
    * @brief Returns the referee command.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  enum RefereeCommand GetRefereeCommand();

  /*!
    * @brief Returns the blue team score.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  int GetBlueTeamScore();

  /*!
    * @brief Returns the yellow team score.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  int GetYellowTeamScore();

  /*!
    * @brief Returns the X coordinate of the ball designated position.
    * 
    * Returns the X coordinate in mm of the ball designated position. This value is
    * relevant when the BALL_PLACEMENT_YELLOW or BALL_PLACEMENT_BlUE command is
    * issued by the referee, which means that a robot has to bring the ball to the
    * designated position.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  float GetBallDesignatedPositionX();

  /*!
    * @brief Returns the Y coordinate of the ball designated position.
    * 
    * Returns the X coordinate in mm of the ball designated position. This value is
    * relevant when the BALL_PLACEMENT_YELLOW or BALL_PLACEMENT_BlUE command is
    * issued by the referee, which means that a robot has to bring the ball to the
    * designated position.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  float GetBallDesignatedPositionY();

  /*!
    * @brief Returns the remaining stage time left.
    * 
    * Returns the remaining stage time left in seconds. If the stage time is passed
    * this value becomed negative.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  int64_t GetStageTimeLeft();

  /*!
    * @brief Returns the team that has been assigned to the positive half of the field.
    * 
    * @pre In order to have the data available ReceivePacket() needs to be called
    * beforehand.
    */
  enum Team GetTeamOnPositiveHalf();

  /* enum RefereeCommand GetNextRefereeCommand(); */

protected:
  /* Helper methods */
  std::string RefereeCommandToString(enum RefereeCommand referee_command);
  void ReadGameStateData(Referee packet);
  enum RefereeCommand ConvertRefereeCommand(enum Referee_Command command);

  /* Network variables */
  sockaddr_in client_address;
  int socket;

  /* Game state data */
  enum RefereeCommand referee_command;
  enum RefereeCommand next_referee_command;
  int blue_team_score;
  int yellow_team_score;
  int64_t stage_time_left;
  float ball_designated_position_x;
  float ball_designated_position_y;
  enum Team team_on_positive_half;
};

} /* namespace ssl_interface */
} /* namesapce centralized_ai */

#endif /* CENTRALIZEDAI_SSLGAMECONTROLLERCLIENT_H_ */