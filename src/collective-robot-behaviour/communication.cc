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
    referee.AnalyzeGameState();

    int32_t num_states = 43;
    torch::Tensor states = torch::empty(num_states);
    states[0] = 0; /* Reserved for the robot id. */

    /* Ball position */
    states[1] = vision_client.GetBallPositionX();
    states[2] = vision_client.GetBallPositionY();

    /* Own team positions */
    states[3] = vision_client.GetRobotPositionX(0, own_team);
    states[4] = vision_client.GetRobotPositionY(0, own_team);
    states[5] = vision_client.GetRobotPositionX(1, own_team);
    states[6] = vision_client.GetRobotPositionY(1, own_team);
    states[7] = vision_client.GetRobotPositionX(2, own_team);
    states[8] = vision_client.GetRobotPositionY(2, own_team);
    states[9] = vision_client.GetRobotPositionX(3, own_team);
    states[10] = vision_client.GetRobotPositionY(3, own_team);
    states[11] = vision_client.GetRobotPositionX(4, own_team);
    states[12] = vision_client.GetRobotPositionY(4, own_team);
    states[13] = vision_client.GetRobotPositionX(5, own_team);
    states[14] = vision_client.GetRobotPositionY(5, own_team);

    /* Opponent team positions */
    states[15] = vision_client.GetRobotPositionX(0, opponent_team);
    states[16] = vision_client.GetRobotPositionY(0, opponent_team);
    states[17] = vision_client.GetRobotPositionX(1, opponent_team);
    states[18] = vision_client.GetRobotPositionY(1, opponent_team);
    states[19] = vision_client.GetRobotPositionX(2, opponent_team);
    states[20] = vision_client.GetRobotPositionY(2, opponent_team);
    states[21] = vision_client.GetRobotPositionX(3, opponent_team);
    states[22] = vision_client.GetRobotPositionY(3, opponent_team);
    states[23] = vision_client.GetRobotPositionX(4, opponent_team);
    states[24] = vision_client.GetRobotPositionY(4, opponent_team);
    states[25] = vision_client.GetRobotPositionX(5, opponent_team);
    states[26] = vision_client.GetRobotPositionY(5, opponent_team);

    /* Goal difference */
    states[27] = ComputeGoalDifference(referee, own_team);

    /* Own team ball owner */
    states[28] = referee.IsTouchingBall(0, own_team);
    states[29] = referee.IsTouchingBall(1, own_team);
    states[30] = referee.IsTouchingBall(2, own_team);
    states[31] = referee.IsTouchingBall(3, own_team);
    states[32] = referee.IsTouchingBall(4, own_team);
    states[33] = referee.IsTouchingBall(5, own_team);

    /* Opponent team ball owner */
    states[34] = referee.IsTouchingBall(0, opponent_team);
    states[35] = referee.IsTouchingBall(1, opponent_team);
    states[36] = referee.IsTouchingBall(2, opponent_team);
    states[37] = referee.IsTouchingBall(3, opponent_team);
    states[38] = referee.IsTouchingBall(4, opponent_team);
    states[39] = referee.IsTouchingBall(5, opponent_team);

    /* Remaining time */
    states[40] = referee.GetStageTimeLeft();

    /* Reshape the states to [1, 1, num_states], but keeping the data in the third dimension. */
    return states.view({1, 1, num_states});
  }

  torch::Tensor ComputeRewards(torch::Tensor & states, RewardConfiguration reward_configuration, Team own_team)
  {
    torch::Tensor positions = torch::zeros({2, 6});
    positions[0][0] = states[3];
    positions[1][0] = states[4];
    positions[0][1] = states[5];
    positions[1][1] = states[6];
    positions[0][2] = states[7];
    positions[1][2] = states[8];
    positions[0][3] = states[9];
    positions[1][3] = states[10];
    positions[0][4] = states[11];
    positions[1][4] = states[12];
    positions[0][5] = states[13];
    positions[1][5] = states[14];

    torch::Tensor average_distance_reward = ComputeAverageDistanceReward(positions, 1, reward_configuration.average_distance_reward);
    
    torch::Tensor have_ball = states.slice(0, 28, 34);
    torch::Tensor have_ball_reward = ComputeHaveBallReward(have_ball, reward_configuration.have_ball_reward);
    
    torch::Tensor total_reward = average_distance_reward + have_ball_reward;

    return total_reward;
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

  Observation GetObservations(ssl_interface::AutomatedReferee & referee, ssl_interface::VisionClient & vision_client, RewardConfiguration reward_configuration, Team team)
  {
    /* Correct the dessignated teams.*/
    Team teammate;
    Team opponent;
    switch (team)
    {
    case Team::kBlue:
      teammate = Team::kBlue;
      opponent = Team::kYellow;
      break;
    case Team::kYellow:
      teammate = Team::kYellow;
      opponent = Team::kBlue;
      break;
    case Team::kUnknown:
      teammate = Team::kUnknown;
      opponent = Team::kUnknown;
    default:
      teammate = Team::kUnknown;
      opponent = Team::kUnknown;
      break;
    }

    /* Get the states of the world. */
    torch::Tensor states = GetStates(referee, vision_client, teammate, opponent);

    /* Compute the rewards for each agent. */
    torch::Tensor rewards = ComputeRewards(states, reward_configuration, teammate);

    return Observation{rewards : rewards, state : states};
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
      default:
        break;
      }

      robot_interfaces[i].SendPacket();
    }
  }


}/* namespace centralised_ai */
}/*namespace collective_robot_behaviour*/

