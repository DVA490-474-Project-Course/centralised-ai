//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-11-05 by Jacob Johansson
// Description: Stores all tests for the communication.cc and communication.h file.
// License: See LICENSE file for license details.
//==============================================================================

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/communication.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

TEST(ComputeRewardsTest, TestShape)
{
    torch::Tensor states = torch::ones(40);
    RewardConfiguration reward_configuration = {1, 1, 1};
    Team own_team = Team::kBlue;

    torch::Tensor output = ComputeRewards(states, reward_configuration, own_team);

    EXPECT_EQ(output.size(0), 6);
}

/* Mock class for VisionClient. */
class VisionClientDerived : public centralised_ai::ssl_interface::VisionClient
{
public:
  VisionClientDerived(std::string ip, int port) : VisionClient(ip, port) {}
  void SetBlueRobotPositionX(int id, float value) {blue_robot_positions_x[id] = value;}
  void SetBlueRobotPositionY(int id, float value) {blue_robot_positions_y[id] = value;}
  void SetBlueRobotOrientation(int id, float value) {blue_robot_orientations[id] = value;}
  void SetYellowRobotPositionX(int id, float value) {yellow_robot_positions_y[id] = value;}
  void SetYellowRobotPositionY(int id, float value) {yellow_robot_positions_x[id] = value;}
  void SetYellowRobotOrientation(int id, float value) {yellow_robot_orientations[id] = value;}
  void SetBallPositionX(float value) {ball_position_x = value;}
  void SetBallPositionY(float value) {ball_position_y = value;}
  void SetTimestamp (double value) {timestamp = value;}
};

/* Mock class for AutomatedReferee. */
class AutomatedRefereeDerived : public centralised_ai::ssl_interface::AutomatedReferee
{
public:
  AutomatedRefereeDerived(ssl_interface::VisionClient & vision_client, std::string ip, int port) : AutomatedReferee(vision_client, ip, port) {}
  void SetRefereeCommand(centralised_ai::RefereeCommand command) {referee_command = command;}
  void SetBlueTeamScore(int score) {blue_team_score = score;}
  void SetYellowTeamScore(int score) {yellow_team_score = score;}
};

/* Sets all robot and ball posiitons to zero */
void SetAllPositionsToZero(VisionClientDerived& vision_client)
{
  for (int id = 0; id < centralised_ai::team_size; id++)
  {
    vision_client.SetBlueRobotPositionX(id, 0.0F);
    vision_client.SetBlueRobotPositionY(id, 0.0F);
    vision_client.SetBlueRobotOrientation(id, 0.0F);
    vision_client.SetYellowRobotPositionX(id, 0.0F);
    vision_client.SetYellowRobotPositionY(id, 0.0F);
    vision_client.SetYellowRobotOrientation(id, 0.0F);
  }

  vision_client.SetBallPositionX(0.0F);
  vision_client.SetBallPositionY(0.0F);
}

TEST(GetStateTest, TestShape)
{
    VisionClientDerived vision_client = VisionClientDerived("127.0.0.1", 20001);
    centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client, "127.0.0.1", 10001);

    SetAllPositionsToZero(vision_client);

    RewardConfiguration reward_configuration = {1, 1, 1};

    torch::Tensor states = GetStates(automated_referee, vision_client, Team::kBlue, Team::kYellow);

    EXPECT_EQ(states.size(0), 1);
    EXPECT_EQ(states.size(1), 1);
    EXPECT_EQ(states.size(2), 43);
}

}
}