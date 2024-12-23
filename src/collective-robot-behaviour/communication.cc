/* Communication.c
 * ==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-23.
 * Last modified: 2024-12-12 by Jacob Johansson
 * Description: Functions for communicating with the ssl interface and
 * simulation interface. License: See LICENSE file for license details.
 * ==============================================================================
 */

#include "communication.h"
#include "../../src/common_types.h"
#include "../../src/simulation-interface/simulation_interface.h"
#include "../../src/ssl-interface/automated_referee.h"
#include "network.h"
#include "reward.h"
#include "torch/torch.h"
#include "vector"

namespace centralised_ai {
namespace collective_robot_behaviour {

/* Utility function for calculating the goal difference. */
static int32_t ComputeGoalDifference(ssl_interface::AutomatedReferee referee,
                                     Team team) {
  switch (team) {
  case Team::kBlue:
    return referee.GetBlueTeamScore() - referee.GetYellowTeamScore();
  case Team::kYellow:
    return referee.GetYellowTeamScore() - referee.GetBlueTeamScore();
  default:
    return 0;
  }
}

torch::Tensor GetLocalState(ssl_interface::VisionClient& vision_client,
                            Team own_team, int robot_id) {
  torch::Tensor states = torch::zeros(4);

  float position_x = vision_client.GetRobotPositionX(robot_id, own_team);
  float position_y = vision_client.GetRobotPositionY(robot_id, own_team);
  float orientation = vision_client.GetRobotOrientation(robot_id, own_team);
  float ball_position_x = vision_client.GetBallPositionX();
  float ball_position_y = vision_client.GetBallPositionY();

  states[0] = position_x;
  states[1] = position_y;
  states[2] = orientation;
  states[3] = ball_position_x;
  states[4] = ball_position_y;

  return states.view({1, 1, states.size(0)});
}

torch::Tensor GetGlobalState(ssl_interface::AutomatedReferee& referee,
                             ssl_interface::VisionClient& vision_client,
                             Team own_team, Team opponent_team) {
  /* Important! Both ReceivePacket() and AnalyzeGameState() should ideally be on
     the same thread, but separate one. Not on the main thread as they are now.
     This is because ReceivePacket() is a blocking call and will wait for the
     next packet to arrive.
  */
  vision_client.ReceivePacket();
  referee.AnalyzeGameState();

  torch::Tensor states = torch::zeros(21);

  /* Reserved for the robot id */
  states[0] = 0;

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

  /* Own team orientations */
  states[15] = vision_client.GetRobotOrientation(0, own_team);
  states[16] = vision_client.GetRobotOrientation(1, own_team);
  states[17] = vision_client.GetRobotOrientation(2, own_team);
  states[18] = vision_client.GetRobotOrientation(3, own_team);
  states[19] = vision_client.GetRobotOrientation(4, own_team);
  states[20] = vision_client.GetRobotOrientation(5, own_team);

  /* Reshape the states to [1, 1, num_states], but keeping the data in the third
   * dimension. */
  return states.view({1, 1, states.size(0)});
}

Team ComputeOpponentTeam(Team own_team) {
  switch (own_team) {
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

void SendActions(
    std::vector<simulation_interface::SimulationInterface> robot_interfaces,
    torch::Tensor action_ids) {
  for (int32_t i = 0; i < action_ids.size(0); i++) {
    switch (action_ids[i].item<int>()) {
    case 0: /* Forward */
      robot_interfaces[i].SetVelocity(0.5F, 0.0F, 0.0F);
      break;
    case 1: /* Backward */
      robot_interfaces[i].SetVelocity(-0.5F, 0.0F, 0.0F);
      break;
    case 2: /* Left */
      robot_interfaces[i].SetVelocity(0.0F, 0.5F, 0.0F);
      break;
    case 3: /* Right */
      robot_interfaces[i].SetVelocity(0.0F, -0.5F, 0.0F);
      break;
    case 4: /* Rotate anti-clockwise */
      robot_interfaces[i].SetVelocity(0.0F, 0.0F, 1.0F);
      break;
    case 5: /* Rotate clockwise */
      robot_interfaces[i].SetVelocity(0.0F, 0.0F, -1.0F);
      break;
    default:
      break;
    }

    robot_interfaces[i].SendPacket();
  }
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */
