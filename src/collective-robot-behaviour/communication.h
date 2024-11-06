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
#include "world.h"

/*Extern values*/
extern int input_size;
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
*@brief Struct representing the observation of the state of the world.
*/
struct Observation
{
  /* Tensor representing the rewards for each agent, with the shape [num_agents].*/
  torch::Tensor rewards;
  /* Tensor representing the state of the world, with the shape [num_states]*/
  torch::Tensor state;
};

/*!
*@brief Struct representing the configuration of the rewards.
*/
struct RewardConfiguration
{
    /* The reward that will be given to the robot when within max_distance_from_center. */
    float average_distance_reward;
    /* The maximum distance from the center within the robot will receive the average_distance_reward. */
    float max_distance_from_center;

    /* The reward that will be given to the robot when it has the ball. */
    float have_ball_reward;
};

/*!
* @brief Calculates the opponent team from the own team.
* @returns The opponent team.
*/
Team ComputeOpponentTeam(Team own_team);

/*!
 *@brief Get the current state from grSim
 *
 *@pre The following preconditions must be met before using this class:
 * - A connection to grSim.
 *
 *@param[In] referee: The automated referee, which is the source of the current state of the world.
 *@param[In] vision_client: The vision client, which is the source of the current state of the world.
 *@param[In] Team: The team that the agents are on.
 *@param[Out] Tensor array of the current state that includes ....
 */
Observation GetObservations(ssl_interface::AutomatedReferee referee, ssl_interface::VisionClient vision_client, RewardConfiguration reward_configuration, RewardConfiguration, Team team);

/*!
*@brief Get the current state from grSim
*
*@returns A tensor representing the states of the world, with the shape [1, 1, num_states].
* The states are as follows:
* [0] - Reserved for the robot id as input to each policy network.
* [1] - The x-coordinate of the ball.
* [2] - The y-coordinate of the ball.
* [3] - The x-coordinate of the teammate robot 0.
* [4] - The y-coordinate of the teammate robot 0.
* [5] - The x-coordinate of the teammate robot 1.
* [6] - The y-coordinate of the teammate robot 1.
* [7] - The x-coordinate of the teammate robot 2.
* [8] - The y-coordinate of the teammate robot 2.
* [9] - The x-coordinate of the teammate robot 3.
* [10] - The y-coordinate of the teammate robot 3.
* [11] - The x-coordinate of the teammate robot 4.
* [12] - The y-coordinate of the teammate robot 4.
* [13] - The x-coordinate of the teammate robot 5.
* [14] - The y-coordinate of the teammate robot 5.
* [15] - The x-coordinate of the opponent robot 0.
* [16] - The y-coordinate of the opponent robot 0.
* [17] - The x-coordinate of the opponent robot 1.
* [18] - The y-coordinate of the opponent robot 1.
* [19] - The x-coordinate of the opponent robot 2.
* [20] - The y-coordinate of the opponent robot 2.
* [21] - The x-coordinate of the opponent robot 3.
* [22] - The y-coordinate of the opponent robot 3.
* [23] - The x-coordinate of the opponent robot 4.
* [24] - The y-coordinate of the opponent robot 4.
* [25] - The x-coordinate of the opponent robot 5.
* [26] - The y-coordinate of the opponent robot 5.
* [27] - The goal difference.
* [28] - The teammate robot 0 have ball boolean.
* [29] - The teammate robot 1 have ball boolean.
* [30] - The teammate robot 2 have ball boolean.
* [31] - The teammate robot 3 have ball boolean.
* [32] - The teammate robot 4 have ball boolean.
* [33] - The teammate robot 5 have ball boolean.
* [34] - The opponent robot 0 have ball boolean.
* [35] - The opponent robot 1 have ball boolean.
* [36] - The opponent robot 2 have ball boolean.
* [37] - The opponent robot 3 have ball boolean.
* [38] - The opponent robot 4 have ball boolean.
* [39] - The opponent robot 5 have ball boolean.
* [40] - The remaining time in the current stage.
* [41] - The referee command.
* [42] - The remaining time until the next referee command (if applicable, i.e. from kickoff to start).
*
*@pre The following preconditions must be met before using this class:
* - A connection to grSim.
* - AutomatedReferee::AnalyzeGameState() and VisionClient::ReceivePacket() must be called on a separate thread.
*
*@param[In] referee: The automated referee, which is the source of the current state of the world.
*@param[In] vision_client: The vision client, which is the source of the current state of the world.
*@param[In] own_team: The team that the robots are on.
*@param[In] opponent_team: The team that the robots are playing against.
*/
torch::Tensor GetStates(ssl_interface::AutomatedReferee & referee, ssl_interface::VisionClient & vision_client, Team own_team, Team opponent_team);

/*!
*@brief Computes the rewards for each agent.
*
*@returns A tensor representing the rewards for each agent, with the shape [num_agents].
*@param[In] states: The states of the world, with the shape [num_states].
*@param[In] reward_configuration: The configuration of the rewards.
*@param[In] own_team: The team that the agents are on.
*/
torch::Tensor ComputeRewards(torch::Tensor & states, RewardConfiguration reward_configuration, Team own_team);

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/

#endif
