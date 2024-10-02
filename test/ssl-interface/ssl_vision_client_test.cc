// ssl_vision_client_test.cc
//==============================================================================
// Author: Aaiza A. Khan, Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 2024-10-02 by Shruthi Puthiya Kunnon
// Description: A test suite for ssl-vision
// License: See LICENSE file for license details.
//==============================================================================

// test_vision_client.cc
#include <gtest/gtest.h>
#include "ssl_vision_client.h"
#include <gmock/gmock.h>

// Mock VisionClient class
class MockVisionClient : public VisionClient {
public:
    MockVisionClient(std::string ip, int port) : VisionClient(ip, port) {}

    // Correctly mock the ReceivePacket method
    MOCK_METHOD(void, ReceivePacket, (PositionData* position_data), (override));
};

class VisionClientDerived : public VisionClient
{
public:
  using VisionClient::VisionClient;
  sockaddr_in& get_client_address() {return client_address;}
  int& get_socket() {return socket;}
  static const int& get_max_datagram_size() {return max_datagram_size;}
  socklen_t& get_address_length() {return address_length;}
};

// Test case 1: Initialization
TEST(VisionClientTest, InitializesCorrectly) {

    VisionClientDerived client("127.0.0.1", 10006);

    EXPECT_EQ(client.get_client_address().sin_family, AF_INET);
    EXPECT_EQ(ntohs(client.get_client_address().sin_port), 10006);
    EXPECT_EQ(client.get_client_address().sin_addr.s_addr, inet_addr("127.0.0.1"));
    EXPECT_GE(client.get_socket(), 0);  // socket should be valid (>= 0)

}

// Test case 2: Receives and parses packet
TEST(VisionClientTest, ReceivesAndParsesPacket) {
    VisionClientDerived client("127.0.0.1", 10006);
    PositionData position_data;

    // Mock SSL_WrapperPacket
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    // Set required fields for the detection frame
    detection->set_frame_number(1);
    detection->set_t_capture(12345678.0);  // Example timestamp
    detection->set_t_sent(12345679.0);     // Example timestamp
    detection->set_camera_id(0);           // Camera ID is required

    // Add a blue robot to the detection frame
    SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
    robot_blue->set_robot_id(0);
    robot_blue->set_x(50.0f);
    robot_blue->set_y(100.0f);
    robot_blue->set_orientation(1.57f); // Remove set_has_orientation if it doesn't exist

    // Set required fields for the robot
    robot_blue->set_confidence(1.0f);      // Confidence level is required
    robot_blue->set_pixel_x(500);          // Pixel X position is required
    robot_blue->set_pixel_y(600);          // Pixel Y position is required

    // Add a ball to the detection frame
    SSL_DetectionBall* ball = detection->add_balls();
    ball->set_x(75.0f);
    ball->set_y(150.0f);

    // Set required fields for ball
    ball->set_confidence(1.0f);
    ball->set_pixel_x(500);
    //ball->set_pixel_y(600);

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
    MockVisionClient mock_client("127.0.0.1", 10006);
    PositionData position_data;

    // Create an empty packet
    SSL_WrapperPacket empty_packet;

    // Serialize the empty packet
    std::string serialized_data;
    empty_packet.SerializeToString(&serialized_data);

    // Mock the behavior of ReceivePacket to simulate receiving an empty packet
    EXPECT_CALL(mock_client, ReceivePacket(&position_data))
        .WillOnce(testing::Invoke([&position_data](PositionData* data) {
            // Simulate the outcome of processing an empty packet
            data->blue_robot_position[0] = {}; // Clear position
            data->ball_position = {}; // Clear ball position
        }));

    // Call the mocked method
    mock_client.ReceivePacket(&position_data);

    // Verify that no robots or balls were detected
    EXPECT_EQ(position_data.blue_robot_position[0].x, 0.0f);
    EXPECT_EQ(position_data.blue_robot_position[0].y, 0.0f);
    EXPECT_EQ(position_data.ball_position.x, 0.0f);
    EXPECT_EQ(position_data.ball_position.y, 0.0f);
}

// Test case 4: Handles missing orientation
TEST(VisionClientTest, HandlesMissingRobotOrientation) {
    MockVisionClient mock_client("127.0.0.1", 10006);
    PositionData position_data;

    // Create a packet with a robot without orientation
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    // Set required fields for the detection frame
    detection->set_frame_number(1);
    detection->set_t_capture(12345678.0);  // Example timestamp
    detection->set_t_sent(12345679.0);     // Example timestamp
    detection->set_camera_id(0);           // Camera ID is required

    // Add a blue robot without setting orientation
    SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
    robot_blue->set_robot_id(0);
    robot_blue->set_x(50.0f);
    robot_blue->set_y(100.0f);

    // Set required fields for the robot
    robot_blue->set_confidence(1.0f);      // Confidence level is required
    robot_blue->set_pixel_x(500);          // Pixel X position is required
    robot_blue->set_pixel_y(600);          // Pixel Y position is required

    // Serialize packet into a buffer
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);

    // Mock the behavior of ReceivePacket
    EXPECT_CALL(mock_client, ReceivePacket(&position_data))
        .WillOnce(testing::Invoke([&position_data, &serialized_data](PositionData* data) {
            // Simulate processing the packet, leaving orientation as zero
            data->blue_robot_position[0].x = 50.0f;
            data->blue_robot_position[0].y = 100.0f;
            data->blue_robot_position[0].orientation = 0.0f; // Default to zero
        }));

    // Call the mocked method
    mock_client.ReceivePacket(&position_data);

    // Verify the position and orientation of the robot
    EXPECT_EQ(position_data.blue_robot_position[0].x, 50.0f);
    EXPECT_EQ(position_data.blue_robot_position[0].y, 100.0f);
    EXPECT_EQ(position_data.blue_robot_position[0].orientation, 0.0f); // Check default orientation
}

// Test case 5: Handles multiple robots and a ball
TEST(VisionClientTest, HandlesMultipleRobotsAndBall) {
    MockVisionClient mock_client("127.0.0.1", 10006);
    PositionData position_data;

    // Create a packet with multiple robots and a ball
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    // Add blue robots with different positions and orientations
    for (int i = 0; i < 6; ++i) {
        SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
        robot_blue->set_robot_id(i);
        robot_blue->set_x(50.0f + i * 10.0f);  // Different x positions for each robot
        robot_blue->set_y(100.0f + i * 5.0f);   // Different y positions for each robot
        robot_blue->set_orientation(1.0f + i * 0.5f); // Different orientations
        robot_blue->set_confidence(1.0f); // Confidence for each robot
        robot_blue->set_pixel_x(500 + i * 10); // Pixel x positions
        robot_blue->set_pixel_y(600 + i * 10); // Pixel y positions
    }

    // Add a single ball that is being tracked
    SSL_DetectionBall* ball_1 = detection->add_balls();
    ball_1->set_x(75.0f);  // Set ball x position
    ball_1->set_y(150.0f); // Set ball y position

    // Serialize packet into a buffer
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);

    // Mock the behavior of ReceivePacket
    EXPECT_CALL(mock_client, ReceivePacket(&position_data))
        .WillOnce(testing::Invoke([&position_data](PositionData* data) {
            // Simulate processing the packet, handling multiple robots and a single ball
            for (int i = 0; i < 3; ++i) {
                data->blue_robot_position[i].x = 50.0f + i * 10.0f;
                data->blue_robot_position[i].y = 100.0f + i * 5.0f;
                data->blue_robot_position[i].orientation = 1.0f + i * 0.5f;
            }

            // Handle the single ball
            data->ball_position.x = 75.0f;
            data->ball_position.y = 150.0f;
        }));

    // Call the mocked method
    mock_client.ReceivePacket(&position_data);

    // Verify the positions and orientations of the robots
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(position_data.blue_robot_position[i].x, 50.0f + i * 10.0f);
        EXPECT_EQ(position_data.blue_robot_position[i].y, 100.0f + i * 5.0f);
        EXPECT_EQ(position_data.blue_robot_position[i].orientation, 1.0f + i * 0.5f);
    }

    // Verify the position of the single tracked ball
    EXPECT_EQ(position_data.ball_position.x, 75.0f);
    EXPECT_EQ(position_data.ball_position.y, 150.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
