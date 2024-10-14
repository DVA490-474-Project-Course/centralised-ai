/* ssl_game_controller_client_demo.cc
 *==============================================================================
 * Author: Emil Åberg
 * Creation date: 2024-10-01
 * Last modified: 2024-10-02 by Emil Åberg
 * Description: demo of the ssl game controller client
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include <iostream>
#include <fstream>
#include <string>
#include "ssl_game_controller_client.h"

int main(int argc, char *argv[])
{
  /* Demo the game controller client */
  centralized_ai::ssl_interface::GameControllerClient game_controller_client("127.0.0.1", 10006);
  while (true)
  {
    game_controller_client.ReceivePacket();
    game_controller_client.Print();
  }
  return 0;
}