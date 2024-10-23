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

using namespace centralized_ai::ssl_interface;
using ::testing::Return;
using ::testing::_;

// Mock class for simulating network socket behavior
class MockGameControllerClient : public GameControllerClient {
public:
    MockGameControllerClient(std::string ip, int port) : GameControllerClient(ip, port) {}

    MOCK_METHOD(void, ReceivePacket, (), (override));
    MOCK_METHOD(centralized_ai::RefereeCommand, GetRefereeCommand, (), (override));
    MOCK_METHOD(int, GetBlueTeamScore, (), (override));
    MOCK_METHOD(int, GetYellowTeamScore, (), (override));
    MOCK_METHOD(float, GetBallDesignatedPositionX, (), (override));
    MOCK_METHOD(float, GetBallDesignatedPositionY, (), (override));
    MOCK_METHOD(int64_t, GetStageTimeLeft, (), (override));
    MOCK_METHOD(centralized_ai::Team, GetTeamOnPositiveHalf, (), (override));
};

// Test Fixture
class GameControllerClientTest : public ::testing::Test {
protected:
    MockGameControllerClient mock_client;

    GameControllerClientTest() : mock_client("127.0.0.1", 10001) {}

    void SetUp() override {
        // Code to set up the environment, if needed
    }

    void TearDown() override {
        // Code to clean up after each test, if needed
    }
};

// Test for checking referee command reception
TEST_F(GameControllerClientTest, TestGetRefereeCommand) {
    EXPECT_CALL(mock_client, GetRefereeCommand())
        //.WillOnce(Return(centralized_ai::ssl_interface::RefereeCommand::NORMAL_START));
        .WillOnce(Return(centralized_ai::RefereeCommand::NORMAL_START));

    ASSERT_EQ(mock_client.GetRefereeCommand(), centralized_ai::RefereeCommand::NORMAL_START);
}

// Test for blue team score
TEST_F(GameControllerClientTest, TestGetBlueTeamScore) {
    EXPECT_CALL(mock_client, GetBlueTeamScore())
        .WillOnce(Return(2));

    ASSERT_EQ(mock_client.GetBlueTeamScore(), 2);
}

// Test for yellow team score
TEST_F(GameControllerClientTest, TestGetYellowTeamScore) {
    EXPECT_CALL(mock_client, GetYellowTeamScore())
        .WillOnce(Return(3));

    ASSERT_EQ(mock_client.GetYellowTeamScore(), 3);
}

// Test for ball designated position X
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionX) {
    EXPECT_CALL(mock_client, GetBallDesignatedPositionX())
        .WillOnce(Return(1500.0f));

    ASSERT_FLOAT_EQ(mock_client.GetBallDesignatedPositionX(), 1500.0f);
}

// Test for ball designated position Y
TEST_F(GameControllerClientTest, TestGetBallDesignatedPositionY) {
    EXPECT_CALL(mock_client, GetBallDesignatedPositionY())
        .WillOnce(Return(-750.0f));

    ASSERT_FLOAT_EQ(mock_client.GetBallDesignatedPositionY(), -750.0f);
}

// Test for remaining stage time
TEST_F(GameControllerClientTest, TestGetStageTimeLeft) {
    EXPECT_CALL(mock_client, GetStageTimeLeft())
        .WillOnce(Return(120));

    ASSERT_EQ(mock_client.GetStageTimeLeft(), 120);
}

// Test for team assigned to positive half of the field
TEST_F(GameControllerClientTest, TestGetTeamOnPositiveHalf) {
    EXPECT_CALL(mock_client, GetTeamOnPositiveHalf())
        .WillOnce(Return(centralized_ai::Team::kBlue));

    ASSERT_EQ(mock_client.GetTeamOnPositiveHalf(), centralized_ai::Team::kBlue);
}

// Test for packet reception
TEST_F(GameControllerClientTest, TestReceivePacket) {
    EXPECT_CALL(mock_client, ReceivePacket())
        .Times(1);

    mock_client.ReceivePacket();
}
