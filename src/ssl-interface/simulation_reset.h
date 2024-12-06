/* simulation_reset.h
 *==============================================================================
 * Author: Shruthi P. Kunnon
 * Creation date: 2024-10-21
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Provides function to reset robots and ball in grSim
 * License: See LICENSE file for license details.
 *==============================================================================
 */
#ifndef CENTRALISEDAI_SIMULATIONRESET_H_
#define CENTRALISEDAI_SIMULATIONRESET_H_

/* C++ standard library headers */
#include <string>

/* Project .h files */
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

/*!
 * @brief Resets the position and attributes of all robots and the ball in grSim.
 * 
 * This function sends a reset command to grSim, resetting all robot positions 
 * and other relevant attributes, as well as resetting the ball's position. It 
 * is commonly used to start a fresh simulation or after a goal has been scored 
 * or both half times as well as both overtime periods (if needed) start with a 
 * kick-off.
 *
 * @param[in] ip IP address of the machine running grSim. Typically, this 
 * should be "127.0.0.1" when running grSim on the same machine as this code.
 * 
 * @param[in] port The port used by grSim for receiving commands. This should 
 * match the command listen port configured in grSim.
 * 
 * @param[in] team_on_positive_half The team color that plays on the positive 
 * half of the field. Used for resetting the robots' team positions.
 */
void ResetRobotsAndBall(std::string ip, uint16_t port,
  enum Team team_on_positive_half);

} /* namespace ssl_interface */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_SIMULATIONRESET_H_ */