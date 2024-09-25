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
    // test code here
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
