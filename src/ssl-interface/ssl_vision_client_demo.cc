/* ssl_vision_client_demo.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-09-16
 * Last modified: 2024-10-30 by Emil Åberg
 * Description: demo of the ssl vision client
 * License: See LICENSE file for license details.
 *==============================================================================
 */

/* C system headers */
#include <iostream>
#include <fstream>
#include <string>

/* Project .h files */
#include "ssl_vision_client.h"

int main(int argc, char *argv[])
{
  /* Demo the sslvision client */
  centralised_ai::ssl_interface::VisionClient vision_client("127.0.0.1", 10006);
  while (true)
  {
    vision_client.ReceivePacket();
    vision_client.Print();
  }
  return 0;
}