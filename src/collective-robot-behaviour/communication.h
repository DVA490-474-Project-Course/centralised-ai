/* communication.h
 * ==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-23.
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Communication header file for public functions for communicating
 * with the ssl interface and simulation interface. License: See LICENSE file
 * for license details.
 * ==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_COMMUNICATION_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_COMMUNICATION_H_

#include "../../src/common_types.h"
#include "../../src/simulation-interface/simulation_interface.h"
#include "../../src/ssl-interface/automated_referee.h"
#include "network.h"
#include "reward.h"
#include "torch/torch.h"
#include "vector"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Struct representing the configuration of the rewards.
 */
struct RewardConfiguration {

  /*!
   * @brief The reward that will be given to the robot when within
   * max_distance_from_center.
   */
  float average_distance_reward;

  /*!
   * @brief The maximum distance from the center within the robot will receive
   * the average_distance_reward.
   */
  float max_distance_from_center;

  /*!
   * @brief The reward that will be given to the robot when it has the ball.
   */
  float have_ball_reward;

  /*!
   * @brief The reward that will be multiplied with the distance.
   */
  float distance_to_ball_reward;
};

/*!
 * @brief Calculates the opponent team from the own team.
 * @returns The opponent team.
 * @param[in] own_team: The team that the robots are on.
 */
Team ComputeOpponentTeam(Team own_team);

/*!
 * @brief Get the current global state of the world from grSim.
 * @note The num_global_states doesn't match the number of states in the
 *	description due to the fact that not every state was used to train the
 *	current model, but is recommended to use in the final version.
 *
 * @returns A tensor representing the states of the world, with the shape [1, 1,
 *	num_global_states]. The states are as follows: [0] - Reserved for the
 *	robot id as input to the policy network. [1] - The x-coordinate of the
 *ball. [2] - The y-coordinate of the ball. [3] - The x-coordinate of the
 *teammate robot 0. [4]
 *		- The y-coordinate of the teammate robot 0. [5] - The
 *x-coordinate of the teammate robot 1. [6] - The y-coordinate of the teammate
 *robot 1. [7] - The x-coordinate of the teammate robot 2. [8] - The
 *y-coordinate of the teammate robot 2. [9] - The x-coordinate of the teammate
 *robot 3. [10] - The y-coordinate of the teammate robot 3. [11] - The
 *x-coordinate of the teammate robot 4. [12] - The y-coordinate of the teammate
 *robot 4. [13] - The x-coordinate of the teammate robot 5. [14] - The
 *y-coordinate of the teammate robot 5. [15] - The x-coordinate of the opponent
 *robot 0. [16] - The y-coordinate of the opponent robot 0. [17] - The
 *x-coordinate of the opponent robot 1. [18] - The y-coordinate of the opponent
 *robot 1. [19] - The x-coordinate of the opponent robot 2. [20] - The
 *y-coordinate of the opponent robot 2. [21] - The x-coordinate of the opponent
 *robot 3. [22] - The y-coordinate of the opponent robot 3. [23] - The
 *x-coordinate of the opponent robot 4. [24] - The y-coordinate of the opponent
 *robot 4. [25] - The x-coordinate of the opponent robot 5. [26] - The
 *y-coordinate of the opponent robot 5. [27] - The goal difference. [28] - The
 *teammate robot 0 have ball boolean. [29] - The teammate robot 1 have ball
 *boolean. [30] - The teammate robot 2 have ball boolean. [31] - The teammate
 *robot 3 have ball boolean. [32]
 *	- The teammate robot 4 have ball boolean. [33] - The teammate robot 5
 *	have ball boolean. [34] - The opponent robot 0 have ball boolean. [35] -
 *The opponent robot 1 have ball boolean. [36] - The opponent robot 2 have ball
 *boolean. [37] - The opponent robot 3 have ball boolean. [38] - The opponent
 *robot 4 have ball boolean. [39] - The opponent robot 5 have ball boolean. [40]
 *	- The remaining time in the current stage. [41] - The referee command.
 *
 * @pre The following preconditions must be met before using this class:
 * - A connection to grSim.
 * - AutomatedReferee::AnalyzeGameState() and VisionClient::ReceivePacket() must
 *	 be called on a separate thread.
 *
 *	@param[in] referee: The automated referee, which is the source of the
 *	current state of the world.
 *	@param[in] vision_client: The vision client, which is the source of the
 *	current state of the world.
 *	@param[in] own_team: The team that the robots are on.
 *	@param[in] opponent_team: The team that the robots are playing against.
 */
torch::Tensor GetGlobalState(ssl_interface::AutomatedReferee& referee,
                             ssl_interface::VisionClient& vision_client,
                             Team own_team, Team opponent_team);

/*!
 *	@brief Get the local state of the robot with the specified robot id.
 *	@returns A tensor representing the local state of the robot, with the
 *	shape [1, 1, num_local_states].
 *	@param[in] vision_client: The vision client, which is the source of the
 *	current state of the world.
 *	@param[in] own_team: The team that the robots are on.
 *	@param[in] robot_id: The id of the robot.
 */
torch::Tensor GetLocalState(ssl_interface::VisionClient& vision_client,
                            Team own_team, int robot_id);

/*!
 *	@brief Send actions to the robots.
 *
 *	@param[in] robot_interfaces: Array of robot interfaces of all robots.
 *	@param[in] action_ids: The ids of the actions which will be sent to all
 *	robots.
 */
void SendActions(
    std::vector<simulation_interface::SimulationInterface> robot_interfaces,
    torch::Tensor action_ids);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_COMMUNICATION_H_ */
