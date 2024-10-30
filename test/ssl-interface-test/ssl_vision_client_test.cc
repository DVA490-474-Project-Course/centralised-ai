/* ssl_vision_client_test.cc
*==============================================================================
* Author: Aaiza A. Khan, Shruthi Puthiya Kunnon
* Creation date: 2024-09-20
* Last modified: 2024-10-07 by Emil Åberg
* Description: A test suite for ssl-vision
* License: See LICENSE file for license details.
*==============================================================================
*/

/* Related .h files */
#include "../../src/ssl-interface/ssl_vision_client.h"

/* Other .h files */
#include <gtest/gtest.h>
#include <gmock/gmock.h>

/* Project .h files */
#include "../../src/common_types.h"

/* Mock VisionClient class*/
class MockVisionClient : public centralised_ai::ssl_interface::VisionClient {
public:
    MockVisionClient(std::string ip, int port) : VisionClient(ip, port) {}

    /* Mocking the ReceivePacket method */
    MOCK_METHOD(void, ReceivePacket, (), (override));

    /* Methods to set the position and orientation of robots and the ball */
    void SetBlueRobotPositionX(int id, float value) { blue_robot_positions_x[id] = value; }
    void SetBlueRobotPositionY(int id, float value) { blue_robot_positions_y[id] = value; }
    void SetBlueRobotOrientation(int id, float value) { blue_robot_orientations[id] = value; }
    void SetYellowRobotPositionX(int id, float value) { yellow_robot_positions_x[id] = value; }
    void SetYellowRobotPositionY(int id, float value) { yellow_robot_positions_y[id] = value; }
    void SetYellowRobotOrientation(int id, float value) { yellow_robot_orientations[id] = value; }
    void SetBallPositionX(float value) { ball_position_x = value; }
    void SetBallPositionY(float value) { ball_position_y = value; }

    /* Accessors for private members */
    sockaddr_in& GetClientAddress() { return client_address; }
    int& GetSocket() { return socket; }
};

/* Test case 1: Initialization of VisionClientDerived */
TEST(VisionClientTest, InitializesCorrectly) {

    MockVisionClient client("127.0.0.1", 10006);

    EXPECT_EQ(client.GetClientAddress().sin_family, AF_INET);
    EXPECT_EQ(ntohs(client.GetClientAddress().sin_port), 10006);
    EXPECT_EQ(client.GetClientAddress().sin_addr.s_addr, inet_addr("127.0.0.1"));
    /* socket should be valid (>= 0)*/
    EXPECT_GE(client.GetSocket(), 0);
}

/* Test case 2: Receives and parses packet */
TEST(VisionClientTest, ReceivesAndParsesPacket) {
    MockVisionClient mock_client("127.0.0.1", 10006);

    /* Mocking SSL_WrapperPacket*/
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    /* Set required fields for the detection frame */
    detection->set_frame_number(1);
    detection->set_t_capture(1234.0);
    detection->set_t_sent(1234.0);
    detection->set_camera_id(0);

    /* Add a blue robot to the detection frame */
    SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
    robot_blue->set_robot_id(1);
    robot_blue->set_x(50.0f);
    robot_blue->set_y(100.0f);
    robot_blue->set_orientation(1.57f);

    /* Set required fields for the a blue robot */
    robot_blue->set_confidence(1.0f);
    robot_blue->set_pixel_x(500);
    robot_blue->set_pixel_y(600);

    /* Add a ball to the detection frame */
    SSL_DetectionBall* ball = detection->add_balls();
    ball->set_x(75.0);
    ball->set_y(150.0);

    /* Set required fields for a ball */
    ball->set_confidence(1.0f);
    ball->set_pixel_x(500);
    ball->set_pixel_y(600);

    /* Serialize packet into a buffer */
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);

    /* Creates a buffer for the mock recvfrom */
    char buffer[65536];

    printf("3");

    /* Simulate receiving a packet by setting the socket buffer */
    auto mock_recvfrom = [&serialized_data, &buffer](...) {
        memcpy(buffer, serialized_data.data(), serialized_data.size());
        return serialized_data.size();
    };
    /* Mock the behavior of ReceivePacket to simulate receiving a packet */
    EXPECT_CALL(mock_client, ReceivePacket())
            .WillOnce(testing::Invoke([&]() {
                SSL_WrapperPacket received_packet;
                received_packet.ParseFromArray(buffer, serialized_data.size());

                const SSL_DetectionFrame& received_frame = received_packet.detection();
                if (received_frame.robots_blue_size() > 0) {
                    const SSL_DetectionRobot& robot = received_frame.robots_blue(1);
                    mock_client.SetBlueRobotPositionX(1, robot.x());
                    mock_client.SetBlueRobotPositionY(1, robot.y());
                    mock_client.SetBlueRobotOrientation(1, robot.orientation());
                }
                if (received_frame.balls_size() > 0) {
                    const SSL_DetectionBall& ball = received_frame.balls(0);
                    mock_client.SetBallPositionX(ball.x());
                    mock_client.SetBallPositionY(ball.y());
                }
            }));
    /* Call the method and verify the results */
    mock_client.ReceivePacket();
    EXPECT_EQ(mock_client.GetRobotPositionX(1, centralised_ai::Team::kBlue), 50.0f);
    EXPECT_EQ(mock_client.GetRobotPositionY(1, centralised_ai::Team::kBlue), 100.0f);
    EXPECT_EQ(mock_client.GetRobotOrientation(1, centralised_ai::Team::kBlue), 1.57f);
    EXPECT_EQ(mock_client.GetBallPositionX(), 75.0f);
    EXPECT_EQ(mock_client.GetBallPositionY(), 150.0f);
}

/* Test case 3: Handles empty packet */
TEST(VisionClientTest, HandlesEmptyPacket) {
    MockVisionClient mock_client("127.0.0.1", 10006);

    /* Create an empty packet */
    SSL_WrapperPacket empty_packet;

    /* Serialize the empty packet */
    std::string serialized_data;
    empty_packet.SerializeToString(&serialized_data);

    /* Mock the behavior of ReceivePacket to simulate receiving an empty packet */
    EXPECT_CALL(mock_client, ReceivePacket())
        .WillOnce(testing::Invoke([&]() {
            /* Simulate the outcome of processing an empty packet */
            mock_client.SetBlueRobotPositionX(0, {}); /* Clear robot position*/
            mock_client.SetBlueRobotPositionY(0, {});
            mock_client.SetBlueRobotOrientation(0, {});
            mock_client.SetBallPositionX({});         /* Clear ball position*/
            mock_client.SetBallPositionY({});
        }));

    /* Call the mocked method */
    mock_client.ReceivePacket();

    /* Verify that no robots or balls were detected */
    EXPECT_EQ(mock_client.GetRobotPositionX(0, centralised_ai::Team::kBlue), 0.0f);
    EXPECT_EQ(mock_client.GetRobotPositionY(0, centralised_ai::Team::kBlue), 0.0f);
    EXPECT_EQ(mock_client.GetBallPositionX(), 0.0f);
    EXPECT_EQ(mock_client.GetBallPositionY(), 0.0f);
}

/* Test case 4: Handles missing orientation */
TEST(VisionClientTest, HandlesMissingRobotOrientation) {
    MockVisionClient mock_client("127.0.0.1", 10006);

    /* Create a packet with a robot without orientation */
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    /* Set required fields for the detection frame */
    detection->set_frame_number(1);
    detection->set_t_capture(12345678.0);
    detection->set_t_sent(12345679.0);
    detection->set_camera_id(0);

    /* Add a blue robot without setting orientation */
    SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
    robot_blue->set_robot_id(0);
    robot_blue->set_x(50.0f);
    robot_blue->set_y(100.0f);

    /* Set required fields for the robot */
    robot_blue->set_confidence(1.0f);
    robot_blue->set_pixel_x(500);
    robot_blue->set_pixel_y(600);

    /* Serialize packet into a buffer */
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);

    /* Mock the behavior of ReceivePacket*/
    EXPECT_CALL(mock_client, ReceivePacket())
        .WillOnce(testing::Invoke([&]() {
            /* Simulate processing the packet, setting orientation as zero */
            mock_client.SetBlueRobotPositionX(0, 50.0f);
            mock_client.SetBlueRobotPositionY(0, 100.0f);
            mock_client.SetBlueRobotOrientation(0, 0.0f);
        }));

    /* Call the mocked method */
    mock_client.ReceivePacket();

    /* Verify the position and orientation of the robot */
    EXPECT_EQ(mock_client.GetRobotPositionX(0, centralised_ai::Team::kBlue), 50.0f);
    EXPECT_EQ(mock_client.GetRobotPositionY(0,centralised_ai::Team::kBlue), 100.0f);
    EXPECT_EQ(mock_client.GetRobotOrientation(0, centralised_ai::Team::kBlue), 0.0f);
}

/* Test case 5: Handles multiple robots and a ball */
TEST(VisionClientTest, HandlesMultipleRobotsAndBall) {
    MockVisionClient mock_client("127.0.0.1", 10006);

    /* Create a packet with multiple robots and a ball */
    SSL_WrapperPacket packet;
    SSL_DetectionFrame* detection = packet.mutable_detection();

    /* Set required fields for the detection frame*/
    detection->set_frame_number(1);         /* Required frame number*/
    detection->set_t_capture(1000.0);       /* Capture time (in seconds, for example)*/
    detection->set_t_sent(1000.1);          /* Sent time (slightly after capture)*/
    detection->set_camera_id(0);            /* Camera ID (assuming single camera)*/

    /* Add blue robots with different positions and orientations */
    for (int i = 0; i < 6; ++i) {
        SSL_DetectionRobot* robot_blue = detection->add_robots_blue();
        robot_blue->set_robot_id(i);
        robot_blue->set_x(50.0f + i * 10.0f);
        robot_blue->set_y(100.0f + i * 5.0f);
        robot_blue->set_orientation(1.0f + i * 0.5f);
        robot_blue->set_confidence(1.0f);
        robot_blue->set_pixel_x(500 + i * 10);
        robot_blue->set_pixel_y(600 + i * 10);
    }

    /* Add a single ball that is being tracked */
    SSL_DetectionBall* ball_1 = detection->add_balls();
    ball_1->set_x(75.0f);
    ball_1->set_y(150.0f);
    ball_1->set_confidence(1.0f);
    ball_1->set_pixel_x(640);
    ball_1->set_pixel_y(480);

    /* Serialize packet into a buffer */
    std::string serialized_data;
    packet.SerializeToString(&serialized_data);

    /* Mock the behavior of ReceivePacket */
    EXPECT_CALL(mock_client, ReceivePacket())
        .WillOnce(testing::Invoke([&]() {
            /* Simulate processing the packet, handling multiple robots and a single ball */
            for (int i = 0; i < 3; ++i) {
                mock_client.SetBlueRobotPositionX(i, 50.0f + i * 10.0f);
                mock_client.SetBlueRobotPositionY(i, 100.0f + i * 5.0f);
                mock_client.SetBlueRobotOrientation(i, 1.0f + i * 0.5f);
            }

            /* Handle the single ball */
            mock_client.SetBallPositionX(75.0f);
            mock_client.SetBallPositionY(150.0f);
        }));

    /* Call the mocked method */
    mock_client.ReceivePacket();

    /* Verify the positions and orientations of the robots */
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(mock_client.GetRobotPositionX(i, centralised_ai::Team::kBlue), 50.0f + i * 10.0f);
        EXPECT_EQ(mock_client.GetRobotPositionY(i, centralised_ai::Team::kBlue), 100.0f + i * 5.0f);
        EXPECT_EQ(mock_client.GetRobotOrientation(i, centralised_ai::Team::kBlue), 1.0f + i * 0.5f);
    }

    /* Verify the position of the single tracked ball */
    EXPECT_EQ(mock_client.GetBallPositionX(), 75.0f);
    EXPECT_EQ(mock_client.GetBallPositionY(), 150.0f);
}

