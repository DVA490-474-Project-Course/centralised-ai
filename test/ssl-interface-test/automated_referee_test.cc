/* automated_referee_test.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-10-30
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: Test suite for the automatic referee.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* Related .h files */
#include "../../src/ssl-interface/automated_referee.h"

/* Other .h files */
#include "gtest/gtest.h"

/* Project .h files */
#include "../../src/ssl-interface/ssl_vision_client.h"
#include "../../src/common_types.h"

/* Duration for the PREPARE_KICKOFF_BLUE/YELLOW commands in all test cases */
const double prepare_kickoff_duration = 3.0D;

/* Define a vision client class where, for testing purposes, values are set
   manually instead of receiving data from SSL Vision */
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

/* Blue team has first kickoff and scores a goal */
TEST(AutomatedReferee, BlueTeamGoal)
{
  /* Instantiate vision client and automatic referee */
  VisionClientDerived vision_client("127.0.0.1", 20001);
  centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client,
   "127.0.0.1", 10001);
  
  /* Set dummy values to all robot and ball positions */
  SetAllPositionsToZero(vision_client);

  /* Start with blue team having first kickoff */
  automated_referee.StartGame(centralised_ai::Team::kBlue,
    centralised_ai::Team::kYellow, prepare_kickoff_duration, 300);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command before kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_BLUE);

  /* Let time pass until kickoff passes */
  vision_client.SetTimestamp(vision_client.GetTimestamp() + prepare_kickoff_duration
    + 0.1D);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command after kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::NORMAL_START);

  /* Put ball in yellow's goal */
  vision_client.SetBallPositionX(4550.0F);
  vision_client.SetBallPositionY(0.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct score and referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW);
  EXPECT_EQ(automated_referee.GetBlueTeamScore(), 1);
  EXPECT_EQ(automated_referee.GetYellowTeamScore(), 0);
}

/* Yellow team has first kickoff and scores a goal */
TEST(AutomatedReferee, YellowTeamGoal)
{
  /* Instantiate vision client and automatic referee */
  VisionClientDerived vision_client("127.0.0.1", 20001);
  centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client,
   "127.0.0.1", 10001);
  
  /* Set dummy values to all robot and ball positions */
  SetAllPositionsToZero(vision_client);

  /* Start with blue team having first kickoff */
  automated_referee.StartGame(centralised_ai::Team::kYellow,
    centralised_ai::Team::kYellow, prepare_kickoff_duration, 300);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command before kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW);

  /* Let time pass until kickoff passes */
  vision_client.SetTimestamp(vision_client.GetTimestamp() + prepare_kickoff_duration
    + 0.1D);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command after kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::NORMAL_START);

  /* Put ball in blue's goal */
  vision_client.SetBallPositionX(-4550.0F);
  vision_client.SetBallPositionY(0.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct score and referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_BLUE);
  EXPECT_EQ(automated_referee.GetBlueTeamScore(), 0);
  EXPECT_EQ(automated_referee.GetYellowTeamScore(), 1);
}

/* Blue team gets a freekick */
TEST(AutomatedReferee, BlueTeamFreekick)
{
  /* Instantiate vision client and automatic referee */
  VisionClientDerived vision_client("127.0.0.1", 20001);
  centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client,
   "127.0.0.1", 10001);
  
  /* Set dummy values to all robot and ball positions */
  SetAllPositionsToZero(vision_client);

  /* Start with blue team having first kickoff */
  automated_referee.StartGame(centralised_ai::Team::kYellow,
    centralised_ai::Team::kYellow, prepare_kickoff_duration, 300);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command before kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW);

  /* Let time pass until kickoff passes */
  vision_client.SetTimestamp(vision_client.GetTimestamp() + prepare_kickoff_duration
    + 0.1D);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command after kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::NORMAL_START);

  /* Put ball next to a yellow robot */
  vision_client.SetYellowRobotPositionX(0, 0.0F);
  vision_client.SetYellowRobotPositionY(0, 1000.0F);
  vision_client.SetBallPositionX(0.0F);
  vision_client.SetBallPositionY(1010.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Put ball out of field */
  vision_client.SetBallPositionY(3050.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::BALL_PLACEMENT_BLUE);

  /* Put ball at designated position */
  vision_client.SetBallPositionX(
    automated_referee.GetBallDesignatedPositionX());
  vision_client.SetBallPositionY(
    automated_referee.GetBallDesignatedPositionY());

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::DIRECT_FREE_BLUE);
}

/* Yellow team gets a freekick */
TEST(AutomatedReferee, YellowTeamFreekick)
{
  /* Instantiate vision client and automatic referee */
  VisionClientDerived vision_client("127.0.0.1", 20001);
  centralised_ai::ssl_interface::AutomatedReferee automated_referee(vision_client,
   "127.0.0.1", 10001);
  
  /* Set dummy values to all robot and ball positions */
  SetAllPositionsToZero(vision_client);

  /* Start with blue team having first kickoff */
  automated_referee.StartGame(centralised_ai::Team::kYellow,
    centralised_ai::Team::kYellow, prepare_kickoff_duration, 300);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command before kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW);

  /* Let time pass until kickoff passes */
  vision_client.SetTimestamp(vision_client.GetTimestamp() + prepare_kickoff_duration
    + 0.1D);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command after kickoff */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::NORMAL_START);

  /* Put ball next to a bluerobot */
  vision_client.SetBlueRobotPositionX(0, 0.0F);
  vision_client.SetBlueRobotPositionY(0, 1000.0F);
  vision_client.SetBallPositionX(0.0F);
  vision_client.SetBallPositionY(1010.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Put ball out of field */
  vision_client.SetBallPositionY(3050.0F);

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::BALL_PLACEMENT_YELLOW);

  /* Put ball at designated position */
  vision_client.SetBallPositionX(
    automated_referee.GetBallDesignatedPositionX());
  vision_client.SetBallPositionY(
    automated_referee.GetBallDesignatedPositionY());

  /* Let referee do its logic */
  automated_referee.AnalyzeGameState();

  /* Check for correct referee command */
  EXPECT_EQ(automated_referee.GetRefereeCommand(),
    centralised_ai::RefereeCommand::DIRECT_FREE_YELLOW);
}