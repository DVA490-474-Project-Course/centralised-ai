/* Communication.c
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-11-06 by Jacob Johansson
 * Description: Communication file to get information from grSim.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include <torch/torch.h>
#include "network.h"
#include "../ssl-interface/automated_referee.h"
#include "../simulation-interface/simulation_interface.h"
#include "world.h"
#include <vector>

namespace centralised_ai
{
namespace collective_robot_behaviour
{
struct BallOwner
{
  int32_t team_id;
  int32_t robot_id;
};

static int32_t ComputeGoalDifference(ssl_interface::AutomatedReferee referee, Team team)
{
    switch (team)
    {
    case Team::kBlue:
      return referee.GetBlueTeamScore() - referee.GetYellowTeamScore();
    case Team::kYellow:
      return referee.GetYellowTeamScore() - referee.GetBlueTeamScore();
    default:
      return 0;
    }
}

  torch::Tensor GetStates(ssl_interface::AutomatedReferee & referee, ssl_interface::VisionClient & vision_client, Team own_team, Team opponent_team)
  {
    vision_client.ReceivePacket();
    vision_client.ReceivePacket();
    referee.AnalyzeGameState();

    int32_t num_states = input_size;
    int32_t num_robots = amount_of_players_in_team;
    int32_t state_index = 0;
    torch::Tensor states = torch::empty(num_states);
    states[0] = 0; /* Reserved for the robot id. */

    state_index++;

    /* Ball position */
    states[state_index + 1] = vision_client.GetBallPositionX();
    states[state_index + 2] = vision_client.GetBallPositionY();

    state_index += 2;

    /* Own team positions */
    for (int32_t i = 0; i < num_robots; i++)
    {
      states[state_index + i * 2 + 0] = vision_client.GetRobotPositionX(i, own_team);
      states[state_index + i * 2 + 1] = vision_client.GetRobotPositionY(i, own_team);
    }

    state_index += 2 * num_robots;

    /* Opponent team positions */
    for (int32_t i = 0; i < num_robots; i++)
    {
      states[state_index + i * 2 + 0] = vision_client.GetRobotPositionX(i, opponent_team);
      states[state_index + i * 2 + 1] = vision_client.GetRobotPositionY(i, opponent_team);
    }

    state_index += 2 * num_robots;

    /* Goal difference */
    states[state_index] = ComputeGoalDifference(referee, own_team);

    state_index++;

    /* Own team ball owner */
    for (int32_t i = 0; i < num_robots; i++)
    {
      states[state_index + i] = referee.IsTouchingBall(i, own_team);
    }
    
    state_index += num_robots;

    /* Opponent team ball owner */
    for (int32_t i = 0; i < num_robots; i++)
    {
      states[state_index + i] = referee.IsTouchingBall(i, opponent_team);
    }

    state_index += num_robots;

    /* Rotation of each robot in the own team */
    for (int32_t i = 0; i < num_robots; i++)
    {
      states[state_index + i] = vision_client.GetRobotOrientation(i, own_team);
    }

    state_index += num_robots;

    /* Remaining time */
    states[state_index] = referee.GetStageTimeLeft();

    state_index++;

    /* Reshape the states to [1, 1, num_states], but keeping the data in the third dimension. */
    return states.view({1, 1, num_states});
  }

  Team ComputeOpponentTeam(Team own_team)
  {
    switch (own_team)
    {
    case Team::kBlue:
      return Team::kYellow;
    case Team::kYellow:
      return Team::kBlue;
    case Team::kUnknown:
      return Team::kUnknown;
    default:
      return Team::kUnknown;
    }
  }

  void SendActions(std::vector<robot_controller_interface::simulation_interface::SimulationInterface> robot_interfaces, torch::Tensor action_ids)
  {
    for (int32_t i = 0; i < action_ids.size(0); i++)
    {
      // 0: Stop, 1: Forward, 2: Backward
      switch (action_ids[i].item<int>())
      {
      case 0:
        robot_interfaces[i].SetVelocity(0.0F, 0.0F, 0.0F);
        break;
      case 1:
        robot_interfaces[i].SetVelocity(5.0F, 0.0F, 0.0F);
        break;
      case 2:
        robot_interfaces[i].SetVelocity(-5.0F, 0.0F, 0.0F);
        break;
      case 3:
        /* Rotate left. */
        robot_interfaces[i].SetVelocity(-5.0F, -5.0F, 5.0F, 5.0F);
        break;
      case 4:
        /* Rotate right. */
        robot_interfaces[i].SetVelocity(5.0F, 5.0F, -5.0F, -5.0F);
        break;
      default:
        break;
      }

      robot_interfaces[i].SendPacket();
    }
  }


}/* namespace centralised_ai */
}/*namespace collective_robot_behaviour*/

