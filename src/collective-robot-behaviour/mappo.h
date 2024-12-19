/* mappo.h
 *==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-23.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: MAPPO header file.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_MAPPO_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_MAPPO_H_

#include "../../src/common_types.h"
#include "../../src/simulation-interface/simulation_interface.h"
#include "chrono"
#include "communication.h"
#include "network.h"
#include "run_state.h"
#include "torch/torch.h"
#include "tuple"
#include "utils.h"
#include "vector"

namespace centralised_ai {
namespace collective_robot_behaviour {

/*!
 * @brief Resets hidden states and initializes trajectories for agents in a
 * MAPPO implementation.
 *
 * @return A tuple containing:
 * - A vector of `Trajectory` objects, each containing the reset hidden states
 * for all agents.
 *
 * - A tensor of zeros representing the initial action probabilities
 * (`act_prob`) for all actions.
 *
 * - An uninitialized tensor for storing the agent's actions.
 */
std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor> ResetHidden();

/*!
 * @brief Algorithm for training the networks.
 *
 * @details Global constant values is declared in common_types.h!
 *
 * @pre The following preconditions must be met before using this class:
 * - Saved or created models of policy and critic network is needed.
 *
 * @param[in] policy is the created/loaded policy network which will be used by
 * all agents.
 *
 * @param[in] critic is the created/loaded ctritic network that the MAPPO will
 * be validating from.
 *
 * @param[in] data_buffer is the buffer that stores all the chunks of time steps
 * for updating the networks.
 */
void MappoUpdate(PolicyNetwork& policy, CriticNetwork& critic,
                 std::vector<DataBuffer> data_buffer);

/*!
 * @brief Algorithm for stepping in the grSim environment and collecting the
 * data needed for training.
 *
 * @details Global constant values is declared in common_types.h!
 * @pre The following preconditions must be met before using this class:
 * - Saved or created models of policy and critic network is needed.
 *
 * @param[in] policy is the created/loaded policy network which will be used by
 * all agents.
 *
 * @param[in] critic is the created/loaded ctritic network that the MAPPO will
 * be validating from.
 *
 * @param[in] referee is the automated referee that will be used to get the
 * state of the game.
 *
 * @param[in] vision_client is the vision client that will be used to get/set
 * the state of the game.
 *
 * @param[in] own_team is the team that the agents are in.
 *
 * @param[in] simulation_interfaces is the simulation interfaces representing
 * each robot in the game.
 */
std::vector<DataBuffer>
MappoRun(PolicyNetwork& policy, CriticNetwork& critic,
         ssl_interface::AutomatedReferee& referee,
         ssl_interface::VisionClient& vision_client, Team own_team,
         std::vector<simulation_interface::SimulationInterface>
             simulation_interfaces);

/*!
 * @brief Utility function for checking if the network parameters match.
 * @returns A boolean that represents if the parameters match, true if they do,
 * else false.
 * @param[in] kSavedPolicy is the policy network that was saved.
 * @param[in] kLoadedPolicy is the policy network that was loaded.
 * @param[in] kSavedCritic is the critic network that was saved.
 * @param[in] kLoadedCritic is the critic network that was loaded.
 */
bool CheckModelParametersMatch(const PolicyNetwork& kSavedPolicy,
                               const PolicyNetwork& kLoadedPolicy,
                               const CriticNetwork& kSavedCritic,
                               const CriticNetwork& kLoadedCritic);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif