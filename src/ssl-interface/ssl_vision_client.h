// ssl_vision_client.cc
//==============================================================================
// Author: Aaiza A. Khan \and Shruthi Puthiya Kunnon
// Creation date: 2024-09-20
// Last modified: 2024-09-23
// Description: A simple client receiving ball and robots positions and other information from grSim
// License: See LICENSE file for license details.
//==============================================================================
#ifndef SSLVISIONCLIENT_H
#define SSLVISIONCLIENT_H

#include "netraw.h"
#include <string>
#include "messages_robocup_ssl_detection.pb.h"
#include "messages_robocup_ssl_geometry.pb.h"
#include "messages_robocup_ssl_wrapper.pb.h"

using namespace std;


// SSLVisionClient class declaration
class SSLVisionClient {
protected:
    static const int MaxDataGramSize = 65536;
    char * in_buffer;
    Net::UDP mc; // multicast client
    int _port;
    string _net_address;
    string _net_interface;

public:
    SSLVisionClient(int port = 10020,
                    string net_ref_address = "127.0.0.1",
                    string net_ref_interface = "");

    ~SSLVisionClient();
    bool open(bool blocking = false);
    void close();
    bool receive(SSL_WrapperPacket & packet);
};

// Function to print robot info
void printRobotInfo(const SSL_DetectionRobot & robot);

// Function to handle detection data
void handleDetectionData(const SSL_WrapperPacket &packet);

// Function to handle geometry data
void handleGeometryData(const SSL_WrapperPacket &packet);

// Function to receive and process SSL packets
void processSSLPackets();

#endif // SSLVISIONCLIENT_H