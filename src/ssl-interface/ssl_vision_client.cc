// ssl_vision_client.cc
//==============================================================================
// Author: Aaiza A. Khan \and Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 
// Description: A simple client receiving ball and robots positions and other information from grSim
// License: See LICENSE file for license details.
//==============================================================================
#include "ssl_vision_client.h"
#include "timer.h"
#include <stdio.h>

void printRobotInfo(const SSL_DetectionRobot & robot) {
    printf("CONF=%4.2f ", robot.confidence());
    if (robot.has_robot_id()) {
        printf("ID=%3d ", robot.robot_id());
    } else {
        printf("ID=N/A ");
    }
    printf(" HEIGHT=%6.2f POS=<%9.2f,%9.2f> ", robot.height(), robot.x(), robot.y());
    if (robot.has_orientation()) {
        printf("ANGLE=%6.3f ", robot.orientation());
    } else {
        printf("ANGLE=N/A    ");
    }
    printf("RAW=<%8.2f,%8.2f>\n", robot.pixel_x(), robot.pixel_y());
}

void handleDetectionData(const SSL_WrapperPacket &packet) {
    if (packet.has_detection()) {
        SSL_DetectionFrame detection = packet.detection();
        double t_now = GetTimeSec();

        printf("-[Detection Data]-------\n");
        printf("Camera ID=%d FRAME=%d T_CAPTURE=%.4f T_CAPTURE_CAM=%.4f\n", 
            detection.camera_id(), detection.frame_number(), 
            detection.t_capture(), detection.t_capture_camera());

        printf("SSL-Vision Processing Latency %7.3fms\n", 
            (detection.t_sent() - detection.t_capture()) * 1000.0);
        printf("Network Latency %7.3fms\n", 
            (t_now - detection.t_sent()) * 1000.0);
        printf("Total Latency %7.3fms\n", 
            (t_now - detection.t_capture()) * 1000.0);

        // Ball info
        for (int i = 0; i < detection.balls_size(); ++i) {
            const SSL_DetectionBall &ball = detection.balls(i);
            printf("-Ball (%2d/%2d): CONF=%4.2f POS=<%9.2f,%9.2f> ", 
                i + 1, detection.balls_size(), ball.confidence(), ball.x(), ball.y());
            if (ball.has_z()) {
                printf("Z=%7.2f ", ball.z());
            } else {
                printf("Z=N/A   ");
            }
            printf("RAW=<%8.2f,%8.2f>\n", ball.pixel_x(), ball.pixel_y());
        }

        // Blue robot info
        for (int i = 0; i < detection.robots_blue_size(); ++i) {
            const SSL_DetectionRobot &robot = detection.robots_blue(i);
            printf("-Robot(B) (%2d/%2d): ", i + 1, detection.robots_blue_size());
            printRobotInfo(robot);
        }

        // Yellow robot info
        for (int i = 0; i < detection.robots_yellow_size(); ++i) {
            const SSL_DetectionRobot &robot = detection.robots_yellow(i);
            printf("-Robot(Y) (%2d/%2d): ", i + 1, detection.robots_yellow_size());
            printRobotInfo(robot);
        }
    }
}

void handleGeometryData(const SSL_WrapperPacket &packet) {
    if (packet.has_geometry()) {
        const SSL_GeometryData &geom = packet.geometry();
        const SSL_GeometryFieldSize &field = geom.field();

        printf("-[Geometry Data]-------\n");
        printf("Field Dimensions:\n");
        printf("  -field_length=%d (mm)\n", field.field_length());
        printf("  -field_width=%d (mm)\n", field.field_width());
        // Further field data printing as in the original code...
    }
}

void processSSLPackets() {
    SSLVisionClient client; // Updated class name
    client.open(true);
    SSL_WrapperPacket packet;

    while (true) {
        if (client.receive(packet)) {
            printf("-----Received Wrapper Packet---------------------------------------------\n");

            handleDetectionData(packet);
            handleGeometryData(packet);
        }
    }
}

// Implementing the SSLVisionClient class

SSLVisionClient::SSLVisionClient(int port,
                     string net_address,
                     string net_interface) {
    _port = port;
    _net_address = net_address;
    _net_interface = net_interface;
    in_buffer = new char[65536];
}

SSLVisionClient::~SSLVisionClient() {
    delete[] in_buffer;
}

void SSLVisionClient::close() {
    mc.close();
}

bool SSLVisionClient::open(bool blocking) {
    close();
    if (!mc.open(_port, true, true, blocking)) {
        fprintf(stderr, "Unable to open UDP network port: %d\n", _port);
        fflush(stderr);
        return false;
    }

    Net::Address multiaddr, interface;
    multiaddr.setHost(_net_address.c_str(), _port);
    if (_net_interface.length() > 0) {
        interface.setHost(_net_interface.c_str(), _port);
    } else {
        interface.setAny();
    }

    if (!mc.addMulticast(multiaddr, interface)) {
        fprintf(stderr, "Unable to setup UDP multicast\n");
        fflush(stderr);
        return false;
    }

    return true;
}

bool SSLVisionClient::receive(SSL_WrapperPacket & packet) {
    Net::Address src;
    int r = 0;
    r = mc.recv(in_buffer, MaxDataGramSize, src);
    if (r > 0) {
        fflush(stdout);
        // Decode packet:
        return packet.ParseFromArray(in_buffer, r);
    }
    return false;
}