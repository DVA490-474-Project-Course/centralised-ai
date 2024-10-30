/* simulation_reset.h
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-10-21
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Provides function to reset robots and ball in grSim
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef SIMULATION_RESET_H
#define SIMULATION_RESET_H

/* C++ standard library headers */
#include <string>

/* Project .h files */
#include "../common_types.h"

namespace centralised_ai
{
namespace ssl_interface
{

/* Reset ball and all robots position and other attributes */
void ResetRobotsAndBall(std::string ip, uint16_t port,
  enum Team team_on_positive_half);

} /* namespace ssl_interface */
} /* namespace centralised_ai */

#endif /* SIMULATION_RESET_H */