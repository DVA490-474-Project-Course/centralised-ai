/* simulation_interface.h
 *==============================================================================
 * Author: Emil Åberg, Shruthi Puthiya Kunnon, Aaiza Aziz Khan
 * Creation date: 2024-09-25
 * Last modified: 2024-10-07 by Emil Åberg
 * Description: Interface that can send UDP data to grSim to make robots move
 * and kick
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_SIMULATIONINTERFACE_SIMULATIONINTERFACE_H_
#define CENTRALISEDAI_SIMULATIONINTERFACE_SIMULATIONINTERFACE_H_

/* C system headers */
#include "arpa/inet.h"
#include "netinet/in.h"
#include "sys/socket.h"

/* C++ standard library headers */
#include "memory"
#include "string"

/* Project .h files */
#include "../simulation-interface/generated/grsim_commands.pb.h"
#include "../simulation-interface/generated/grsim_packet.pb.h"
#include "../common_types.h"

namespace centralised_ai
{
namespace simulation_interface
{

/*!
 * @brief Class for interfacíng to grSim .
 * 
 * Class that allows communication with grSim and methods to control one robot
 * in the simulation. Multiple robots can be controlled with multiple 
 * instantiations of this class.
 * 
 * @note not copyable, not moveable.
 */
class SimulationInterface
{
 public:
  /*!
   * @brief Constructor that sets up connection to grSim for one robot.
   *
   * @param[in] ip Ip address of the computer that is running grSim. When 
   * running grSim on the same computer that the simulation interface is 
   * running on this value should be localhost i.e. "127.0.0.1".
   *
   * @param[in] port The command listen port of grSim. This should
   * be set to the same value as that which is set in the grSim 
   * configuration in Communication->Command listen port.
   * 
   * @param[in] id Id number of the robot that is conntrolled in grSim.
   * 
   * @param[in] team Team color of the robot that is controlled in grSim.
   */
  SimulationInterface(std::string ip, uint16_t port, int id, enum Team team);

  /*!
   * @brief Method to change robot to control with the class instance.
   *
   * @param[in] id ID of the robot in grSim that is to be controlled.
   *
   * @param[in] team Team color of the robot that is to be controlled.
   */
  void SetRobot(int id, enum Team team);

  /*!
   * @brief Method to set the velocity of the kicker.
   *
   * @param[in] kicker_speed Set the speed of the kicker in m/s.
   * 
   * @pre In order for robot commands to take effect, UDP packets need
   * to be sent periodica by calling SendPacket().
   * 
   * @see SendPacket for sending UDP packets.
   */
  void SetKickerSpeed(float kicker_speed);

  /*!
   * @brief Method to control the spinner.
   *
   * @param[in] spinner_on Set wheter the spinner is on.
   * 
   * @pre In order for robot commands to take effect, UDP packets need
   * to be send continuously by calling SendPacket()
   * 
   * @see SendPacket for sending UDP packets.
   */
  void SetSpinnerOn(bool spinner_on);

  /*!
   * @brief Method to set the robot velocity in terms of x, y and angular speeds.
   *
   * @param[in] x_speed The speed of the robot along the x axis in m/s.
   * 
   * @param[in] y_speed The speed of the robot along the y axis in m/s.
   * 
   * @param[in] angular_speed The angular speed of the robot in radians/s.
   * 
   * @note The velocity can either be set in terms of x, y and theta using 
   * this method,or alternatively by setting the speed of the individual 
   * wheels by using 
   * SetVelocity(float front_left_wheel_speed, float back_left_wheel_speed, 
   * float back_right_wheel_speed, float front_right_wheel_speed).
   * 
   * @pre In order for robot commands to take effect, UDP packets need
   * to be sent continuously by calling SendPacket().
   */
  void SetVelocity(float x_speed, float y_speed, float angular_speed);

  /*!
   * @brief Method to set the robot velocity by setting the speeds of the 
   * individual wheels.
   *
   * @param[in] front_left_wheel_speed The speed of the front left wheel in 
   * m/s.
   * 
   * @param[in] back_left_wheel_speed The speed of the back left wheel in m/s.
   * 
   * @param[in] back_right_wheel_speed The speed of the back right wheel in 
   * m/s.
   * 
   * @param[in] front_right_wheel_speed The speed of the front right wheel in 
   * m/s.
   * 
   * @note the velocity can either be set in terms of setting the individual 
   * wheel speeds using this method, ot by setting the velocity in terms of 
   * x, y and theta by using SetVelocity(float x_speed, float y_speed, 
   * float angular_speed).
   * 
   * @pre In order for robot commands to take effect, UDP packets need
   * to be sent continuously by calling SendPacket().
   */
  void SetVelocity(float front_left_wheel_speed, float back_left_wheel_speed,
      float back_right_wheel_speed, float front_right_wheel_speed);

  /*!
   * @brief Sends a UDP packet to grSim, carrying the robot command
   * 
   * Sends a UDP packet to grSim, needs to be called periodically in order for 
   * communication to be maintained, recommended minimum rate of 50Hz.
   * 
   * @warning A robot that is not continously receiving commands will just
   * stand still.
   */
  virtual void SendPacket();

 protected:
  /*********************/
  /* Network variables */
  /*********************/

  /*!
   * @brief socket file descriptor.
   */
  int socket_;

  /*!
   * @brief Address of grSim.
   */
  sockaddr_in destination_;

  /*******************/
  /* Robot variables */
  /*******************/

  /*!
   * @brief Robot id.
   */
  int id_;

  /*!
   * @brief Team of the robot.
   */
  enum Team team_;

  /*!
   * @brief Flag indicating wheter spinner is on.
   */
  bool spinner_on_;

  /*!
   * @brief Kicker speed in m/s.
   */
  float kicker_speed_;

  /*!
   * @brief Speed in x direction in m/s.
   */
  float x_speed_;

  /*!
   * @brief Speed in y direction in m/s.
   */
  float y_speed_;

  /*!
   * @brief Angular speed of the robot in radians/s.
   */
  float angular_speed_;

  /*!
   * @brief Speed of front left wheel in m/s.
   */
  float wheel_1_;

  /*!
   * @brief Speed of back left wheel in m/s.
   */
  float wheel_2_;

  /*!
   * @brief Speed of back right wheel in m/s.
   */
  float wheel_3_;

  /*!
   * @brief Speed of front right wheel in m/s.
   */
  float wheel_4_;

  /*!
   * @brief Flag indicating whether individual wheel speeds are set or if
   * velocity is determined by setting the x, y, theta speeds of the entire 
   * robot.
   */
  bool using_wheel_speed_;

  /********************/
  /* Helper functions */
  /********************/

  /*!
   * @brief Send a grSim protobuf packet.
   */
  void SendPacket(GrSimPacket packet);

  /*!
   * @brief Structure robot command into a grSim Protobuf packet.
   */
  GrSimPacket CreateProtoPacket();
};

} /* namespace simulation_interface */
} /* namesapce centralised_ai */

#endif /* CENTRALISEDAI_SIMULATIONINTERFACE_SIMULATIONINTERFACE_H_ */