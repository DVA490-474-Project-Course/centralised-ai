/* ssl_game_controller_client.h
 *==============================================================================
 * Author: Emil Åberg, Aaiza A. Khan
 * Creation date: 2024-10-01
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: A client for receiveing game state from ssl game controller
 * License: See LICENSE file for license details.
 *=============================================================================
 */

#ifndef CENTRALISEDAI_SSLINTERFACE_SSLGMAECONTROLLERCLIENT_H_
#define CENTRALISEDAI_SSLINTERFACE_SSLGMAECONTROLLERCLIENT_H_

/* C system headers */
#include "arpa/inet.h"

/* C++ standard library headers */
#include "string"

/* Project .h files */
#include "../ssl-interface/referee_command_functions.h"
#include "../ssl-interface/generated/ssl_gc_referee_message.pb.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

/*!
 * @brief Class for communicating with ssl game controller.
 * 
 * Class that allows communication with ssl game controller and
 * provides methods to read game state variables including score
 * referee command, remaining stage time and ball designated position
 * 
 * @note not copyable, not moveable.
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
  virtual void ReceivePacket();

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
   * 
   * @return The current referee command as an enumeration of type
   * `RefereeCommand`.
   */
  enum RefereeCommand GetRefereeCommand();

  /*!
   * @brief Returns the blue team score.
   * 
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * @return The score of the blue team as an integer.
   * 
   * @return Blue team's score
   */
  int GetBlueTeamScore();

  /*!
   * @brief Returns the yellow team score.
   * 
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * 
   * @return Yellow team's score
   */
  int GetYellowTeamScore();

  /*!
   * @brief Returns the X coordinate of the ball designated position.
   *
   * @return the X coordinate in mm of the ball designated position. This value is
   * relevant when the BALL_PLACEMENT_YELLOW or BALL_PLACEMENT_BlUE command is
   * issued by the referee, which means that a robot has to bring the ball to the
   * designated position.
   * 
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * 
   * @return The X coordinate of the ball's designated position in millimeters.
   */
  float GetBallDesignatedPositionX();

  /*!
   * @brief Returns the Y coordinate of the ball designated position.
   *
   * @return the X coordinate in mm of the ball designated position. This value is
   * relevant when the BALL_PLACEMENT_YELLOW or BALL_PLACEMENT_BlUE command is
   * issued by the referee, which means that a robot has to bring the ball to the
   * designated position.
   * 
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * 
   * @return The Y coordinate of the ball's designated position in millimeters.
   */
  float GetBallDesignatedPositionY();

  /*!
   * @brief Returns the remaining stage time left.
   * 
   * Returns the remaining stage time left in seconds. If the stage time is passed
   * this value becomes negative.
   *
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * @return The remaining stage time in seconds. A negative value indicates that
   * the stage time has passed.
   * 
   * @return Remaining stage time
   */
  int64_t GetStageTimeLeft();

  /*!
   * @brief Returns the team that has been assigned to the positive half of the
   * field.
   * 
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * 
   * @return The team whose goal is assigned to the posistive half of the field.
   */
  enum Team GetTeamOnPositiveHalf();

  /*!
   * @brief Retrieve the next referee command.
   *
   * @pre In order to have the data available ReceivePacket() needs to be called
   * beforehand.
   * 
   * @return Next referee command as an enumeration of type
   * `RefereeCommand`.
   */
  enum RefereeCommand GetNextRefereeCommand();

 protected:
  /*!
   * @brief Read and store the relevant game state data from the Referee packet.
   *
   * @param[in] packet The `Referee` packet containing game state information to
   * be processed.
   */
  void ReadGameStateData(Referee packet);

  /*!
   * @brief The sockaddr_in structure used to store the client's address.
   */
  sockaddr_in client_address_;

  /*!
   * @brief The socket descriptor used for UDP communication.
   */
  int socket_;

  /*!
   * @brief Current command received from the referee.
   */
  enum RefereeCommand referee_command_;

  /*!
   * @brief The next referee command in the game.
   */
  enum RefereeCommand next_referee_command_;

  /*!
   * @brief Blue team's score.
   */
  int blue_team_score_;

  /*!
   * @brief Yellow team's score.
   */
  int yellow_team_score_;

  /*!
   * @brief Remaining stage time.
   */
  int64_t stage_time_left_;

  /*!
   * @brief X coordinate of the ball's designated position.
   */
  float ball_designated_position_x_;

  /*!
   * @brief X coordinate of the ball's designated position.
   */
  float ball_designated_position_y_;

  /*!
   * @brief The team currently on the positive half of the field.
   */
  enum Team team_on_positive_half_;
};

} /* namespace ssl_interface */
} /* namesapce centralised_ai */

#endif /* CENTRALISEDAI_SSLINTERFACE_SSLGMAECONTROLLERCLIENT_H_ */