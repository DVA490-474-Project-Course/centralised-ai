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

/* Other .h files */
#include "gtest/gtest.h"
#include "gmock/gmock.h"

/* Project .h files */
#include "../../src/ssl-interface/generated/ssl_gc_referee_message.pb.h"
#include "../../src/common_types.h"

/* Mock class for simulating network socket behavior */
class MockGameControllerClient : public centralised_ai::ssl_interface
    ::GameControllerClient {
public:
  /* Constructor to initialize with IP and port */
  MockGameControllerClient(std::string ip, int port)
      : GameControllerClient(ip, port) {}

  /* Mock method to simulate receiving a packet */
  MOCK_METHOD(void, ReceivePacket, (), (override));

  /* Method to test reading game state data */
  void TestReadGameStateData(Referee& packet) {
    GameControllerClient::ReadGameStateData(packet);
  }
};

centralised_ai::RefereeCommand ConvertProtoRefereeCommand(Referee_Command command) {
  switch (command) {
    case Referee_Command_HALT:
      return centralised_ai::RefereeCommand::kHalt;
    case Referee_Command_STOP:
      return centralised_ai::RefereeCommand::kStop;
    case Referee_Command_NORMAL_START:
      return centralised_ai::RefereeCommand::kNormalStart;
    case Referee_Command_PREPARE_KICKOFF_YELLOW:
      return centralised_ai::RefereeCommand::kPrepareKickoffYellow;
    case Referee_Command_PREPARE_KICKOFF_BLUE:
      return centralised_ai::RefereeCommand::kPrepareKickoffBlue;
    case Referee_Command_FORCE_START:
      return centralised_ai::RefereeCommand::kForceStart;
    case Referee_Command_PREPARE_PENALTY_YELLOW:
      return centralised_ai::RefereeCommand::kPreparePenaltyYellow;
    case Referee_Command_PREPARE_PENALTY_BLUE:
      return centralised_ai::RefereeCommand::kPreparePenaltyBlue;
    case Referee_Command_DIRECT_FREE_YELLOW:
      return centralised_ai::RefereeCommand::kDirectFreeYellow;
    case Referee_Command_INDIRECT_FREE_BLUE:
      return centralised_ai::RefereeCommand::kDirectFreeBlue;
    case Referee_Command_TIMEOUT_YELLOW:
      return centralised_ai::RefereeCommand::kTimeoutYellow;
    case Referee_Command_DIRECT_FREE_BLUE:
      return centralised_ai::RefereeCommand::kDirectFreeBlue;
    case Referee_Command_BALL_PLACEMENT_YELLOW:
      return centralised_ai::RefereeCommand::kBallPlacementYellow;
    case Referee_Command_BALL_PLACEMENT_BLUE:
      return centralised_ai::RefereeCommand::kBallPlacementBlue;
    default:
      return centralised_ai::RefereeCommand::kUnknownCommand;
  }
}
/* Test Fixture: Set up the test environment and mock client */
class GameControllerClientTest : public ::testing::Test {
protected:
  /* Mocked GameControllerClient */
  MockGameControllerClient mock_client;

  /* Dummy Referee packet used for testing */
  Referee dummy_packet;

  /* Constructor: Initialize mock client with IP and port */
  GameControllerClientTest() : mock_client("127.0.0.1", 10001) {}
  void SetUp() override {

    /* Code to set up the environment */
    dummy_packet.set_packet_timestamp(123456789);
    dummy_packet.set_stage(Referee::NORMAL_FIRST_HALF);
    dummy_packet.set_command(Referee::HALT);
    dummy_packet.set_command_counter(10);
    dummy_packet.set_command_timestamp(123456789);
    dummy_packet.mutable_designated_position()->set_x(0.0f);
    dummy_packet.mutable_designated_position()->set_y(0.0f);
    dummy_packet.set_stage_time_left(50);

    /* Set up yellow team info */
    Referee::TeamInfo* yellow_team = dummy_packet.mutable_yellow();
    yellow_team->set_name("Yellow Team");
    yellow_team->set_score(1);

    /* Set up blue team info */
    Referee::TeamInfo* blue_team = dummy_packet.mutable_blue();
    blue_team->set_name("Blue Team");
    blue_team->set_score(2);
  }

  /* Clean up after each test if needed */
  void TearDown() override {
  }
};

/* Test case for reading game state data */
TEST_F(GameControllerClientTest, TestReadGameStateData) {
  /* Call the TestReadGameStateData method */
  mock_client.TestReadGameStateData(dummy_packet);

  /* Verify values after calling ReadGameStateData */
  EXPECT_EQ(mock_client.GetRefereeCommand(),
      centralised_ai::RefereeCommand::kHalt);
  EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
  EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
  EXPECT_EQ(mock_client.GetStageTimeLeft(), 50);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f);
}

/* Test GetRefereeCommand */
TEST_F(GameControllerClientTest, TestGetRefereeCommand) {
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_EQ(mock_client.GetRefereeCommand(),
      centralised_ai::RefereeCommand::kHalt);
}

/* Test GetBlueTeamScore */
TEST_F(GameControllerClientTest, TestGetBlueTeamScore) {
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
}

/* Test GetYellowTeamScore */
TEST_F(GameControllerClientTest, TestGetYellowTeamScore) {
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
}

/* Test GetBallDesignatedPositionX */
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionX) {
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f);
}

/* Test GetBallDesignatedPositionY */
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionY) {
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f);
}

/* Test GetStageTimeLeft */
TEST_F(GameControllerClientTest, TestGetStageTimeLeft) {
  mock_client.TestReadGameStateData(dummy_packet);
  /* Adjust as per the actual implementation */
  EXPECT_EQ(mock_client.GetStageTimeLeft(), 50);
}

/* Test GetTeamOnPositiveHalf */
TEST_F(GameControllerClientTest, TestGetTeamOnPositiveHalf) {
  /* Assuming the positive half team is set somewhere in your actual implementation.
     In this dummy packet, we may not have it, so you'd adjust accordingly */
  dummy_packet.set_blue_team_on_positive_half(true);
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(),
      centralised_ai::Team::kBlue);
  dummy_packet.set_blue_team_on_positive_half(false);
  mock_client.TestReadGameStateData(dummy_packet);
  EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(),
      centralised_ai::Team::kYellow);
}