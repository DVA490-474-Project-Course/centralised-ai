/* ssl_game_controller_client_test.cc
* Author: Aaiza A. Khan, Shruthi Puthiya Kunnon
* Creation date: 2024-10-22
* Last modified: 2024-10-23 by Aaiza A. Khan
* Description: A test suite for ssl_game_controller_client
* License: See LICENSE file for license details.
*==============================================================================
*/

/* Related .h files */
#include "../../src/ssl-interface/ssl_game_controller_client.h"

/* C++ standard library headers */

/* Other .h files */
#include <gtest/gtest.h>
#include <gmock/gmock.h>

/* Project .h files */
#include "../../src/ssl-interface/generated/ssl_gc_referee_message.pb.h"
#include "../../src/common_types.h"

/* Mock class for simulating network socket behavior */
class MockGameControllerClient : public centralised_ai::ssl_interface::GameControllerClient {
public:
  MockGameControllerClient(std::string ip, int port) : GameControllerClient(ip, port) {}
  MOCK_METHOD(void, ReceivePacket, (), (override));
  void TestReadGameStateData(Referee& packet) { GameControllerClient::ReadGameStateData(packet); }
};

centralised_ai::RefereeCommand ConvertProtoRefereeCommand(Referee_Command command) {
  switch (command) {
    case Referee_Command_HALT:
      return centralised_ai::RefereeCommand::HALT;
    case Referee_Command_STOP:
      return centralised_ai::RefereeCommand::STOP;
    case Referee_Command_NORMAL_START:
      return centralised_ai::RefereeCommand::NORMAL_START;
    case Referee_Command_PREPARE_KICKOFF_YELLOW:
      return centralised_ai::RefereeCommand::PREPARE_KICKOFF_YELLOW;
    case Referee_Command_PREPARE_KICKOFF_BLUE:
      return centralised_ai::RefereeCommand::PREPARE_KICKOFF_BLUE;
    case Referee_Command_FORCE_START:
      return centralised_ai::RefereeCommand::FORCE_START;
    case Referee_Command_PREPARE_PENALTY_YELLOW:
      return centralised_ai::RefereeCommand::PREPARE_PENALTY_YELLOW;
    case Referee_Command_PREPARE_PENALTY_BLUE:
      return centralised_ai::RefereeCommand::PREPARE_PENALTY_BLUE;
    case Referee_Command_DIRECT_FREE_YELLOW:
      return centralised_ai::RefereeCommand::DIRECT_FREE_YELLOW;
    case Referee_Command_INDIRECT_FREE_BLUE:
      return centralised_ai::RefereeCommand::DIRECT_FREE_BLUE;
    case Referee_Command_TIMEOUT_YELLOW:
      return centralised_ai::RefereeCommand::TIMEOUT_YELLOW;
    case Referee_Command_DIRECT_FREE_BLUE:
      return centralised_ai::RefereeCommand::DIRECT_FREE_BLUE;
    case Referee_Command_BALL_PLACEMENT_YELLOW:
      return centralised_ai::RefereeCommand::BALL_PLACEMENT_YELLOW;
    case Referee_Command_BALL_PLACEMENT_BLUE:
      return centralised_ai::RefereeCommand::BALL_PLACEMENT_BLUE;
    default:
      return centralised_ai::RefereeCommand::UNKNOWN_COMMAND;
  }
}
/* Test Fixture */
class GameControllerClientTest : public ::testing::Test {
protected:
  MockGameControllerClient mock_client;
  Referee dummyPacket;
  GameControllerClientTest() : mock_client("127.0.0.1", 10001) {}
  void SetUp() override {

    /* Code to set up the environment */
    dummyPacket.set_packet_timestamp(123456789);
    dummyPacket.set_stage(Referee::NORMAL_FIRST_HALF);
    dummyPacket.set_command(Referee::HALT);
    dummyPacket.set_command_counter(10);
    dummyPacket.set_command_timestamp(123456789);
    dummyPacket.mutable_designated_position()->set_x(0.0f);
    dummyPacket.mutable_designated_position()->set_y(0.0f);
    dummyPacket.set_stage_time_left(50);

    /* Set up yellow team info */
    Referee::TeamInfo* yellow_team = dummyPacket.mutable_yellow();
    yellow_team->set_name("Yellow Team");
    yellow_team->set_score(1);

    /* Set up blue team info */
    Referee::TeamInfo* blue_team = dummyPacket.mutable_blue();
    blue_team->set_name("Blue Team");
    blue_team->set_score(2);
  }

  void TearDown() override {
  /* Code to clean up after each test, if needed */
  }
};

TEST_F(GameControllerClientTest, TestReadGameStateData) {
  /* Call the TestReadGameStateData method */
  mock_client.TestReadGameStateData(dummyPacket);

  /* Verify values after calling ReadGameStateData */
  EXPECT_EQ(mock_client.GetRefereeCommand(), centralised_ai::RefereeCommand::HALT);
  EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
  EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
  EXPECT_EQ(mock_client.GetStageTimeLeft(), 50);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f);
}

/* Test GetRefereeCommand */
TEST_F(GameControllerClientTest, TestGetRefereeCommand) {
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_EQ(mock_client.GetRefereeCommand(), centralised_ai::RefereeCommand::HALT);
}

/* Test GetBlueTeamScore */
TEST_F(GameControllerClientTest, TestGetBlueTeamScore) {
    mock_client.TestReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
}

/* Test GetYellowTeamScore */
TEST_F(GameControllerClientTest, TestGetYellowTeamScore) {
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
}

/* Test GetBallDesignatedPositionX */
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionX) {
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f);
}

/* Test GetBallDesignatedPositionY */
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionY) {
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f);
}

/* Test GetStageTimeLeft */
TEST_F(GameControllerClientTest, TestGetStageTimeLeft) {
  mock_client.TestReadGameStateData(dummyPacket);
  /* Adjust as per the actual implementation */
  EXPECT_EQ(mock_client.GetStageTimeLeft(), 50);
}

/* Test GetTeamOnPositiveHalf */
TEST_F(GameControllerClientTest, TestGetTeamOnPositiveHalf) {
  /* Assuming the positive half team is set somewhere in your actual implementation.
     In this dummy packet, we may not have it, so you'd adjust accordingly */
  dummyPacket.set_blue_team_on_positive_half(true);
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(), centralised_ai::Team::kBlue);
  dummyPacket.set_blue_team_on_positive_half(false);
  mock_client.TestReadGameStateData(dummyPacket);
  EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(), centralised_ai::Team::kYellow);
}