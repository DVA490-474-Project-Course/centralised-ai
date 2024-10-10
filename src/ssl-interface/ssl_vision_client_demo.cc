/* ssl_vision_client_demo.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-09-16
 * Last modified: 2024-09-19 by Emil Åberg
 * Description: demo of the ssl vision client
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include <iostream>
#include <fstream>
#include <string>
#include "ssl_vision_client.h"

int main(int argc, char *argv[])
{
  /* Demo the sslvision client */
  centralized_ai::ssl_interface::VisionClient vision_client("224.5.23.1", 10003);
  while (true)
  {
    vision_client.ReceivePacket();
    vision_client.Print();
  }
  return 0;
}