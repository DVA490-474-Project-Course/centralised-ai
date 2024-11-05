/* Communication.c
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-11-05 by Jacob Johansson
 * Description: Communication file to get information from grSim.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include <torch/torch.h>
#include "network.h"
#include "../ssl-interface/automated_referee.h"
#include "world.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{
  struct BallOwner
  {
    int32_t team_id;
    int32_t robot_id;
  };

  struct RewardConfiguration
  {
    float average_distance_reward;
    float have_ball_reward;

    float max_distance_from_center;
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

  static BallOwner ComputeBallOwner(ssl_interface::AutomatedReferee referee, Team teammates, Team opponent, int32_t num_robots)
  {

    for (int32_t id = 0; id < num_robots; id++)
    {
      if (referee.IsTouchingBall(id, teammates))
      {
        return BallOwner{team_id : 1, robot_id : id};
      }
      else if (referee.IsTouchingBall(id, opponent))
      {
        return BallOwner{team_id : 0, robot_id : id};
      }
    }

    return BallOwner{team_id : -1, robot_id : -1};
  }

  static torch::Tensor ComputeHaveBall(ssl_interface::AutomatedReferee referee, Team team, int32_t num_robots)
  {
    torch::Tensor have_ball = torch::zeros({num_robots, 1});
    for(int32_t i = 0; i < num_robots; i++)
    {
      if(!referee.IsTouchingBall(i, team))
      {
        have_ball[i] = 1;
      }
    }

    return have_ball;
  }

  State GetObservations(ssl_interface::AutomatedReferee referee, ssl_interface::VisionClient vision_client, Team team)
  {
    /* Cache the current state of the game.*/
    referee.AnalyzeGameState();

    int32_t num_inputs = 34;
    int32_t num_robots = 6;
    torch::Tensor state_vector = torch::empty({1,1,num_inputs});

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

    /* Compute the ball owner*/
    BallOwner ball_owner = ComputeBallOwner(referee, teammate, opponent, num_robots);

    /* [0] is reservered for robot id.*/
    /* Ball position */
    state_vector[1] = vision_client.GetBallPositionX();
    state_vector[2] = vision_client.GetBallPositionY();
    /* Team positions */
    state_vector[3] = vision_client.GetRobotPositionX(0, teammate);
    state_vector[4] = vision_client.GetRobotPositionY(0, teammate);
    state_vector[5] = vision_client.GetRobotPositionX(1, teammate);
    state_vector[6] = vision_client.GetRobotPositionY(1, teammate);
    state_vector[7] = vision_client.GetRobotPositionX(2, teammate);
    state_vector[8] = vision_client.GetRobotPositionY(2, teammate);
    state_vector[9] = vision_client.GetRobotPositionX(3, teammate);
    state_vector[10] = vision_client.GetRobotPositionY(3, teammate);
    state_vector[11] = vision_client.GetRobotPositionX(4, teammate);
    state_vector[12] = vision_client.GetRobotPositionY(4, teammate);
    state_vector[13] = vision_client.GetRobotPositionX(5, teammate);
    state_vector[14] = vision_client.GetRobotPositionY(5, teammate);
    /* Opponent positions */
    state_vector[15] = vision_client.GetRobotPositionX(0, opponent);
    state_vector[16] = vision_client.GetRobotPositionY(0, opponent);
    state_vector[17] = vision_client.GetRobotPositionX(1, opponent);
    state_vector[18] = vision_client.GetRobotPositionY(1, opponent);
    state_vector[19] = vision_client.GetRobotPositionX(2, opponent);
    state_vector[20] = vision_client.GetRobotPositionY(2, opponent);
    state_vector[21] = vision_client.GetRobotPositionX(3, opponent);
    state_vector[22] = vision_client.GetRobotPositionY(3, opponent);
    state_vector[23] = vision_client.GetRobotPositionX(4, opponent);
    state_vector[24] = vision_client.GetRobotPositionY(4, opponent);
    state_vector[25] = vision_client.GetRobotPositionX(5, opponent);
    state_vector[26] = vision_client.GetRobotPositionY(5, opponent);
    /* Goal difference */
    state_vector[27] = ComputeGoalDifference(referee, teammate);
    /* Ball owned team */
    state_vector[28] = ball_owner.team_id;
    /* Robot id with ball */
    state_vector[29] = ball_owner.robot_id;
    /* Team yellow cards*/
    //state_vector[30] =
    /* Team red cards */
    //state_vector[31] =
    /* Opponent yellow cards */
    //state_vector[32] =
    /* Opponent red cards */
    //state_vector[33] =
    /* Time remaining */
    state_vector[34] = referee.GetStageTimeLeft();

    /* Calculate the rewards per robot. */
    torch::Tensor positions = torch::zeros({2, num_robots});
    positions[0][0] = state_vector[3];
    positions[1][0] = state_vector[4];
    positions[0][1] = state_vector[5];
    positions[1][1] = state_vector[6];
    positions[0][2] = state_vector[7];
    positions[1][2] = state_vector[8];
    positions[0][3] = state_vector[9];
    positions[1][3] = state_vector[10];
    positions[0][4] = state_vector[11];
    positions[1][4] = state_vector[12];
    positions[0][5] = state_vector[13];
    positions[1][5] = state_vector[14];

    torch::Tensor average_distance_reward = ComputeAverageDistanceReward(positions, 1, -0.0001);
    torch::Tensor have_ball = ComputeHaveBall(referee, teammate, num_robots);
    torch::Tensor have_ball_reward = ComputeHaveBallReward(have_ball, 0.0001);
    torch::Tensor total_reward = average_distance_reward + have_ball_reward;


    return State{rewards : total_reward, state : state_vector};
  }
}/* namespace centralised_ai */
}/*namespace collective_robot_behaviour*/

