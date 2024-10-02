// main.cc
//==============================================================================
// Author: Emil Åberg
// Creation date: 2024-09-16
// Last modified: 2024-09-19 by Emil Åberg
// Description: demo of the ssl vision client
// License: See LICENSE file for license details.
//==============================================================================

#include <iostream>
#include <fstream>
#include <string>
#include "ssl_vision_client.h"

int main(int argc, char *argv[])
{
  // Demo the sslvision client
  VisionClient vision_client("224.5.23.1", 10003);
  struct PositionData position_data;
  while (true)
  {
    vision_client.ReceivePacket(&position_data);
    vision_client.PrintPositionData(position_data);
  }
  return 0;
}