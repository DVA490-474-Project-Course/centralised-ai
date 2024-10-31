/* communication.h
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: Communication header file.
 * License: See LICENSE file for license details.
 *==============================================================================
 */
#ifndef COMMUNICATION_H_H
#define COMMUNICATION_H_H

/* PyTorch C++ API library  */
#include <torch/torch.h>

/*Extern values*/
extern int input_size;
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;

namespace centralised_ai {
namespace collective_robot_behaviour {

/*!
 *@brief Get the current state from grSim
 *
 *@pre The following preconditions must be met before using this class:
 * - A connection to grSim.
 *
 *@param[Out] Tensor array of the current state that includes ....
 */
torch::Tensor GetStates();

/*!
 *@brief Get the reward for the agent
 *
 *@pre The following preconditions must be met before using this class:
 * - A connection to grSim.
 *
 *@param[Out] float of the calculated reward
 */
  torch::Tensor GetRewards();

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //COMMUNICATION_H_H
