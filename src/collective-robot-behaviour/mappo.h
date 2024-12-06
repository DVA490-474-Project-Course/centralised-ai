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
 *
 *@param[in] amount_of_players_in_team Number of agents on the team whose hidden states need resetting.
 *@param[in] hidden_size Size of the hidden state for the LSTM network.
 *@param[in] num_actions Number of possible actions an agent can take.
 *
 *@return A tuple containing:
 * - A vector of `Trajectory` objects, each containing the reset hidden states for all agents.
 * - A tensor of zeros representing the initial action probabilities (`act_prob`) for all actions.
 * - An uninitialized tensor for storing the agent's actions.
 */
std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor> ResetHidden();

 /*!
  *@brief MAPPO algorithm for training and running in the grSim environment.
  *
  *@pre The following preconditions must be met before using this class:
  * - Saved or created models of policy and critic network is needed.
  *
  *@param[in] models is the created/loaded models for each agent, the amount of models is the amount_of_player_in_team which can be changed in MAPPO.cc.
  *@param[in] critic is the created/loaded ctritic network that the MAPPO will be validating from.
  */
void Mappo_Update(std::vector<Agents> &models,CriticNetwork &critic,std::vector<DataBuffer> data_buffer);

std::vector<DataBuffer> MappoRun(std::vector<Agents> Models, CriticNetwork critic,ssl_interface::AutomatedReferee & referee,
 ssl_interface::VisionClient & vision_client,
 Team own_team,
 std::vector<simulation_interface::SimulationInterface> simulation_interfaces);

 bool CheckModelParametersMatch(const std::vector<centralised_ai::collective_robot_behaviour::Agents>& saved_models,
                                  const std::vector<centralised_ai::collective_robot_behaviour::Agents>& loaded_models,
                                  const centralised_ai::collective_robot_behaviour::CriticNetwork& saved_critic,
                                  const centralised_ai::collective_robot_behaviour::CriticNetwork& loaded_critic);


}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //MAPPO_H_
