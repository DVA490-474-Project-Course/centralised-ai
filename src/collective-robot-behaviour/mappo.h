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

/* PyTorch C++ API library  */
#include <torch/torch.h>

/* Projects .h files. */
#include "communication.h"
#include "network.h"
#include "../simulation-interface/simulation_interface.h"

/*Configuration values, change in MAPPO.cc if needed*/
extern  int max_timesteps;
extern  int steps;
extern int step_max;
extern int batch_size;
extern int amount_of_players_in_team;
namespace centralised_ai {
namespace collective_robot_behaviour {
 /*!
  *@brief MAPPO algorithm for training and running in the grSim environment.
  *
  *@pre The following preconditions must be met before using this class:
  * - Saved or created models of policy and critic network is needed.
  *
  *@param[in] models is the created/loaded models for each agent, the amount of models is the amount_of_player_in_team which can be changed in MAPPO.cc.
  *@param[in] critic is the created/loaded ctritic network that the MAPPO will be validating from.
  */
void Mappo_Update(std::vector<Agents> models,CriticNetwork critic,
 std::vector<DataBuffer> data_buffer,
 std::vector<Agents> &old_net, CriticNetwork &old_net_critic);

std::vector<DataBuffer> MappoRun(std::vector<Agents> Models,
 CriticNetwork critic,ssl_interface::AutomatedReferee & referee,
 ssl_interface::VisionClient & vision_client,
 Team own_team,
 std::vector<robot_controller_interface::simulation_interface::SimulationInterface> simulation_interfaces);

 bool CheckModelParametersMatch(const std::vector<centralised_ai::collective_robot_behaviour::Agents>& saved_models,
                                  const std::vector<centralised_ai::collective_robot_behaviour::Agents>& loaded_models,
                                  const centralised_ai::collective_robot_behaviour::CriticNetwork& saved_critic,
                                  const centralised_ai::collective_robot_behaviour::CriticNetwork& loaded_critic);

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //MAPPO_H_
