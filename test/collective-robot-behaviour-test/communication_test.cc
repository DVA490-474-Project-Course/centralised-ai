//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-16
// Last modified: 2024-11-07 by Jacob Johansson
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
/* Mock class for VisionClient. */
class VisionClientDerived : public centralised_ai::ssl_interface::VisionClient
{
public:
  VisionClientDerived(std::string ip, int port) : VisionClient(ip, port) {}
  void SetBlueRobotPositionX(int id, float value) {blue_robot_positions_x_[id] = value;}
  void SetBlueRobotPositionY(int id, float value) {blue_robot_positions_y_[id] = value;}
  void SetBlueRobotOrientation(int id, float value) {blue_robot_orientations_[id] = value;}
  void SetYellowRobotPositionX(int id, float value) {yellow_robot_positions_y_[id] = value;}
  void SetYellowRobotPositionY(int id, float value) {yellow_robot_positions_x_[id] = value;}
  void SetYellowRobotOrientation(int id, float value) {yellow_robot_orientations_[id] = value;}
  void SetBallPositionX(float value) {ball_position_x_ = value;}
  void SetBallPositionY(float value) {ball_position_y_ = value;}
  void SetTimestamp (double value) {timestamp_ = value;}
};

/* Mock class for AutomatedReferee. */
class AutomatedRefereeDerived : public centralised_ai::ssl_interface::AutomatedReferee
{
public:
  AutomatedRefereeDerived(ssl_interface::VisionClient & vision_client, std::string ip, int port) : AutomatedReferee(vision_client, ip, port) {}
  void SetRefereeCommand(centralised_ai::RefereeCommand command) {referee_command_ = command;}
  void SetBlueTeamScore(int score) {blue_team_score_ = score;}
  void SetYellowTeamScore(int score) {yellow_team_score_ = score;}
};

/* Sets all robot and ball posiitons to zero */
static void SetAllPositionsToZero(VisionClientDerived& vision_client)
{
  for (int id = 0; id < centralised_ai::amount_of_players_in_team; id++)
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


/*TEST(GetGlobalStateTest, TestShape)
{
  VisionClientDerived vision_client = VisionClientDerived("127.0.0.1", 20001);
  centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client, "127.0.0.1", 10001);

  SetAllPositionsToZero(vision_client);
  RewardConfiguration reward_configuration = {1, 1, 1, 1};
  torch::Tensor state = GetGlobalState(automated_referee, vision_client, Team::kBlue, Team::kYellow);
  EXPECT_EQ(state.size(0), 1);
  EXPECT_EQ(state.size(1), 1);
  EXPECT_EQ(state.size(2), 42);
}*/


TEST(ComputeOpponentTeamTest, TestOpponentTeam_1)
{
    Team own_team = Team::kBlue;
    Team opponent_team = ComputeOpponentTeam(own_team);

    EXPECT_EQ(own_team, Team::kBlue);
    EXPECT_EQ(opponent_team, Team::kYellow);
}

TEST(ComputeOpponentTeamTest, TestOpponentTeam_2)
{
    Team own_team = Team::kYellow;
    Team opponent_team = ComputeOpponentTeam(own_team);

    EXPECT_EQ(own_team, Team::kYellow);
    EXPECT_EQ(opponent_team, Team::kBlue);
}

}
}