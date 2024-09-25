// ssl_vision_client_test.cc
//==============================================================================
// Author: Aaiza A. Khan, Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 2024-09-25 by Aaiza A. Khan
// Description: A test suite for ssl-vision
// License: See LICENSE file for license details.
//==============================================================================

// test_vision_client.cc
#include <gtest/gtest.h>
#include "ssl_vision_client.h"

// Test case 1: Initialization
TEST(VisionClientTest, InitializesCorrectly) {

    VisionClient client("127.0.0.1", 10006);

    EXPECT_EQ(client.client_address.sin_family, AF_INET);
    EXPECT_EQ(ntohs(client.client_address.sin_port), 10006);
    EXPECT_EQ(client.client_address.sin_addr.s_addr, inet_addr("127.0.0.1"));
    EXPECT_GE(client.socket, 0);  // socket should be valid (>= 0)

}

// Test case 2: Receives and parses packet
TEST(VisionClientTest, ReceivesAndParsesPacket) {
    VisionClient client("127.0.0.1", 10006);
    PositionData position_data;

    // Mock SSL_WrapperPacket
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();
    
    // Add a blue robot to the detection frame
    SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
    robot_blue->set_robot_id(0);
    robot_blue->set_x(50.0f);
    robot_blue->set_y(100.0f);
    robot_blue->set_orientation(1.57f); // Remove set_has_orientation if it doesn't exist
    
    // Add a ball to the detection frame
    SSL_DetectionBall* ball = detection->add_balls();
    ball->set_x(75.0f);
    ball->set_y(150.0f);

    // Serialize packet into a buffer
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);
    
    // Create a buffer for the mock recvfrom
    char buffer[1024]; // Ensure this size is sufficient for your data

    // Simulate receiving a packet by setting the socket buffer
    auto mock_recvfrom = [&serialized_data, &buffer](...) {
        memcpy(buffer, serialized_data.data(), serialized_data.size());
        return serialized_data.size();
    };

    // Call the method and verify the results
    client.ReceivePacket(&position_data);

    EXPECT_EQ(position_data.blue_robot_position[0].x, 50.0f);
    EXPECT_EQ(position_data.blue_robot_position[0].y, 100.0f);
    EXPECT_EQ(position_data.blue_robot_position[0].orientation, 1.57f);
    EXPECT_EQ(position_data.ball_position.x, 75.0f);
    EXPECT_EQ(position_data.ball_position.y, 150.0f);
}

   



// Test case 3: Handles empty packet
TEST(VisionClientTest, HandlesEmptyPacket) {
    //  test code here
}

// Test case 4: Handles missing orientation
TEST(VisionClientTest, HandlesMissingRobotOrientation) {
    // test code here
}

// Test case 5: Handles multiple robots and balls
TEST(VisionClientTest, HandlesMultipleRobotsAndBalls) {
    // test code here
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
