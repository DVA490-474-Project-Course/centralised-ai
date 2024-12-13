/* mappo.h
 *==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: MAPPO header file.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef MAPPO_H_
#define MAPPO_H_

/* C++ library */
#include <tuple>
#include <vector>
#include "../common_types.h"
/* PyTorch C++ API library  */
#include <torch/torch.h>

/* Projects .h files. */
#include "communication.h"
#include "network.h"
#include "../simulation-interface/simulation_interface.h"

/*Configuration values, change in MAPPO.cc if needed*/

namespace centralised_ai {
namespace collective_robot_behaviour {
/*!
 *@brief Resets hidden states and initializes trajectories for agents in a MAPPO implementation.
 *@return A tuple containing:
 * - A vector of `Trajectory` objects, each containing the reset hidden states for all agents.
 * - A tensor of zeros representing the initial action probabilities (`act_prob`) for all actions.
 * - An uninitialized tensor for storing the agent's actions.
 */
std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor> ResetHidden();

 /*!
  *@brief Algorithm for training the networks.
  *
  *@pre The following preconditions must be met before using this class:
  * - Saved or created models of policy and critic network is needed.
  *
  *@param[in] policy is the created/loaded policy network which will be used by all agents.
  *@param[in] critic is the created/loaded ctritic network that the MAPPO will be validating from.
  *@param[in] data_buffer is the buffer that stores all the chunks of time steps for updating the networks.
  */
void Mappo_Update(PolicyNetwork& policy, CriticNetwork& critic, std::vector<DataBuffer> data_buffer);

/*!
  *@brief Algorithm for stepping in the grSim environment and collecting the data needed for training.
  *
  *@param[in] policy is the created/loaded policy network which will be used by all agents.
  *@param[in] critic is the created/loaded ctritic network that the MAPPO will be validating from.
  *@param[in] referee is the automated referee that will be used to get the state of the game.
  *@param[in] vision_client is the vision client that will be used to get/set the state of the game.
  *@param[in] own_team is the team that the agents are in.
  *@param[in] simulation_interfaces is the simulation interfaces representing each robot in the game.
*/
std::vector<DataBuffer> MappoRun
(
  PolicyNetwork & policy,
  CriticNetwork & critic,
  ssl_interface::AutomatedReferee & referee,
  ssl_interface::VisionClient & vision_client,
  Team own_team,
  std::vector<simulation_interface::SimulationInterface> simulation_interfaces
);

/*!
*@brief Utility function for checking if the network parameters match.
*@param[in] saved_policy is the policy network that was saved.
*@param[in] loaded_policy is the policy network that was loaded.
*@param[in] saved_critic is the critic network that was saved.
*@param[in] loaded_critic is the critic network that was loaded.
*/
bool CheckModelParametersMatch
(
  const PolicyNetwork& saved_policy,
  const PolicyNetwork& loaded_policy,
  const CriticNetwork& saved_critic,
  const CriticNetwork& loaded_critic
);


}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //MAPPO_H_
