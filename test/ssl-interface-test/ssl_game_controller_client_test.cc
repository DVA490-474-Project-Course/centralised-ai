// ssl_game_controller_client_test.cc
//==============================================================================
// Author: Aaiza A. Khan, Shruthi Puthiya Kunnon
// Creation date: 2024-10-22
// Last modified: 2024-10-23 by Aaiza A. Khan
// Description: A test suite for ssl_game_controller_client
// License: See LICENSE file for license details.
//==============================================================================
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../src/ssl-interface/ssl_game_controller_client.h"
#include "../../src/ssl-interface/generated/ssl_gc_referee_message.pb.h" // Protobuf file
#include "../../src/common_types.h"

using namespace centralised_ai::ssl_interface;
using namespace centralised_ai;
using ::testing::Return;
using ::testing::_;

// Mock class for simulating network socket behavior
class MockGameControllerClient : public GameControllerClient {
public:
    MockGameControllerClient(std::string ip, int port) : GameControllerClient(ip, port) {}

    MOCK_METHOD(void, ReceivePacket, (), (override));
    using GameControllerClient::ReadGameStateData;
   
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
// Test Fixture
class GameControllerClientTest : public ::testing::Test {
protected:
    MockGameControllerClient mock_client;
    Referee dummyPacket;

    GameControllerClientTest() : mock_client("127.0.0.1", 10001) {}

    void SetUp() override {
        // Code to set up the environment, if needed
        dummyPacket.set_packet_timestamp(123456789);               // Dummy timestamp
        dummyPacket.set_stage(Referee::NORMAL_FIRST_HALF);          // Game stage
        dummyPacket.set_command(Referee::HALT);                    // Referee command
        dummyPacket.set_command_counter(10);                       // Command counter
        dummyPacket.set_command_timestamp(123456789);              // Command timestamp
        dummyPacket.mutable_designated_position()->set_x(0.0f);
        dummyPacket.mutable_designated_position()->set_y(0.0f);
        dummyPacket.set_stage_time_left(50);

        // Set up yellow team info
        Referee::TeamInfo* yellow_team = dummyPacket.mutable_yellow();
        yellow_team->set_name("Yellow Team");
        yellow_team->set_score(1);

        // Set up blue team info
        Referee::TeamInfo* blue_team = dummyPacket.mutable_blue();
        blue_team->set_name("Blue Team");
        blue_team->set_score(2);
    }

    void TearDown() override {
        // Code to clean up after each test, if needed
    }
};
TEST_F(GameControllerClientTest, TestReadGameStateData) {
    // Call the method you're testing
    mock_client.ReadGameStateData(dummyPacket);

    // Verify values after calling ReadGameStateData
    //EXPECT_EQ(mock_client.GetRefereeCommand(), ConvertProtoRefereeCommand(dummyPacket.command()));
    EXPECT_EQ(mock_client.GetRefereeCommand(), RefereeCommand::HALT);
    EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
    EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
    EXPECT_EQ(mock_client.GetStageTimeLeft(), 50); 
    EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f); 
    EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f); 
}
// Test GetRefereeCommand
TEST_F(GameControllerClientTest, TestGetRefereeCommand) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetRefereeCommand(), RefereeCommand::HALT);
}

// Test GetBlueTeamScore
TEST_F(GameControllerClientTest, TestGetBlueTeamScore) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetBlueTeamScore(), 2);
}

// Test GetYellowTeamScore
TEST_F(GameControllerClientTest, TestGetYellowTeamScore) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetYellowTeamScore(), 1);
}

// Test GetBallDesignatedPositionX
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionX) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 0.0f);
}

// Test GetBallDesignatedPositionY
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionY) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), 0.0f);
}

// Test GetStageTimeLeft
TEST_F(GameControllerClientTest, TestGetStageTimeLeft) {
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetStageTimeLeft(), 50); // Adjust as per the actual implementation
}

// Test GetTeamOnPositiveHalf
TEST_F(GameControllerClientTest, TestGetTeamOnPositiveHalf) {
    // Assuming the positive half team is set somewhere in your actual implementation.
    // In this dummy packet, we may not have it, so you'd adjust accordingly.
    dummyPacket.set_blue_team_on_positive_half(true);
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(), centralised_ai::Team::kBlue);
    dummyPacket.set_blue_team_on_positive_half(false);
    mock_client.ReadGameStateData(dummyPacket);
    EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(), centralised_ai::Team::kYellow);

    //EXPECT_EQ(mock_client.GetTeamOnPositiveHalf(), Team::YELLOW); // Or whatever the correct expected value is
}