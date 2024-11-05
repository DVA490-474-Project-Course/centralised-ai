/* communication.h
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-11-05 by Jacob Johansson
 * Description: Communication header file.
 * License: See LICENSE file for license details.
 *==============================================================================
 */
#ifndef COMMUNICATION_H_H
#define COMMUNICATION_H_H

#include <torch/torch.h>
#include "../ssl-interface/automated_referee.h"
#include "../common_types.h"

/*Extern values*/
extern int input_size;
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;

namespace centralised_ai
{
namespace collective_robot_behaviour
{

struct State
{
  torch::Tensor rewards;

  torch::Tensor state;
};

/*!
 *@brief Get the current state from grSim
 *
 *@pre The following preconditions must be met before using this class:
 * - A connection to grSim.
 *
 * @param[In] referee: The automated referee, which is the source of the current state of the world.
 * @param[In] vision_client: The vision client, which is the source of the current state of the world.
 * @param[In] Team: The team that the agents are on.
 *@param[Out] Tensor array of the current state that includes ....
 */
State GetObservations(ssl_interface::AutomatedReferee referee, ssl_interface::VisionClient vision_client, Team team);

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/

#endif
