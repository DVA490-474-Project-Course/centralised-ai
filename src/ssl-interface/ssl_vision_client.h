/* ssl_vision_client.h
 *==============================================================================
 * Author: Aaiza A. Khan, Shruthi Puthiya Kunnon, Emil Åberg
 * Creation date: 2024-09-20
 * Last modified: 2024-10-30 by Shruthi Puthiya Kunnon
 * Description: A UDP client receiving ball and robots positions from ssl-vision
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_SSLVISIONCLIENT_H_
#define CENTRALISEDAI_SSLVISIONCLIENT_H_

/* C system headers */
#include "arpa/inet.h"

/* C++ standard library headers */
#include "string" 

/* Project .h files */
#include "generated/ssl_vision_detection.pb.h"
#include "generated/ssl_vision_wrapper.pb.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

/*
 * @brief Max datagram size of received UDP packets
 */
constexpr int kMaxDatagramSize = 65536;

/*!
 * @brief Class for communicating with ssl Vision.
 * 
 * Class for communicating with ssl vision and provides methods to read ball and
 * robot positions and orientation.
 * 
 * @note Copyable, moveable.
 */
class VisionClient
{
 public:
  /*!
    * @brief Constructor that sets up connection to ssl Vision
    *
    * Constructor that sets up connection to ssl Vision. When running simulation,
    * the vision packets are sent by grSim.
    *
    * @param[in] ip Vision multicast address as is configured in ssl Vision or grSim.
    * If running on the same computer as this client, it is recommended that it is set
    * to localhost i.e. "127.0.0.1"
    *
    * @param[in] port The vision multicast port as is configured in ssl Vision or
    * grSim.
    */
  VisionClient(std::string ip, int port);

  /*!
    * @brief Reads a UDP packet from ssl Vision.
    * 
    * Reads a UDP packet from ssl Vision, and updates all
    * game state values that are available in the client.
    * 
    * @warning This method is blocking until a UDP packet has been received,
    * potentially introducing a delay in whatever other task the calling thread
    * is doing. It is recommended to continously run this method in a thread
    * separate from where the Get* functions are called.
    */
  virtual void ReceivePacket(); /* Set to virtual in order to mock 
                                 * receiving of packets when testing */

  /*!
    * @brief Prints the vision data that has been read by this client.
    * 
    * Prints the vision data that has been read by this client including
    * robot and ball positions, orientation and timestamp. Used for debugging
    * purpuses.
    */
  void Print();

  /*!
    * @brief Returns the Unix timestamp of the latest packet that has been received.
    */
  double GetTimestamp();

  /*!
    * @brief Returns the x coordinate in mm of robot with specified ID and team.
    * 
    * @param[in] id ID of the specified robot.
    * 
    * @param[in] id Team of the specified robot.
    */
  float GetRobotPositionX(int id, enum Team team);

  /*!
    * @brief Returns the y coordinate in mm of robot with specified ID and team.
    * 
    * @param[in] id ID of robot.
    * 
    * @param[in] id Team of robot.
    */
  float GetRobotPositionY(int id, enum Team team);

  /*!
    * @brief Returns the orientation in radians mm of robot with specified ID and
    * team.
    * 
    * @param[in] id ID of robot.
    * 
    * @param[in] id Team of robot.
    */
  float GetRobotOrientation(int id, enum Team team);

  /*!
    * @brief Returns the x coordinate of the ball in mm.
    */
  float GetBallPositionX();

  /*!
    * @brief Returns the y coordinate of the ball in mm.
    */
  float GetBallPositionY();

  /*!
    * @brief Reads a UDP packet from ssl Vision.
    * 
    * Reads a UDP packets from ssl Vision, and updates all
    * game state values that are available in the client
    * until positions of ball and each robot have been read
    * at least once.
    */
  void ReceivePacketsUntilAllDataRead();
 
 protected:
  /*********************/
  /* Network variables */
  /*********************/

  /*!
   * @brief Address of grSim.
   */
  sockaddr_in client_address_;

  /*!
   * @brief socket file descriptor.
   */
  int socket_;

  /**************************/
  /* Position data and time */
  /**************************/

  /*!
   * @brief The Unix timestamp of the latest packet that has been received.
   */
  double timestamp_;

  /*!
   * @brief Array containing the x coordinates of the blue team robots.
   */
  float blue_robot_positions_x_[team_size];

  /*!
   * @brief Array containing the y coordinates of the blue team robots.
   */
  float blue_robot_positions_y_[team_size];

  /*!
   * @brief Array containing the theta coordinates of the blue team robots.
   */
  float blue_robot_orientations_[team_size];

  /*!
   * @brief Array containing the x coordinates of the yellow team robots.
   */
  float yellow_robot_positions_x_[team_size];

  /*!
   * @brief Array containing the y coordinates of the yellow team robots.
   */
  float yellow_robot_positions_y_[team_size];

  /*!
   * @brief Array containing the theta coordinates of the yellow team robots.
   */
  float yellow_robot_orientations_[team_size];

  /*!
   * @brief X coordinate of the ball.
   */
  float ball_position_x_;

  /*!
   * @brief Y coordinate of the ball.
   */
  float ball_position_y_;

  float blue_robot_positions_read_[team_size];
  float yellow_robot_positions_read_[team_size];
  bool ball_data_read_;

  /**************************/
  /* Private methods */
  /**************************/

  /*!
   * @brief Read the data from the protobuf data in the argument and store it locally
   * in the class instance.
   */
  void ReadVisionData(SslWrapperPacket packet);
};

} /* namespace ssl_interface */
} /* namesapce centralised_ai */

#endif /* CENTRALISEDAI_SSLVISIONCLIENT_H_ */