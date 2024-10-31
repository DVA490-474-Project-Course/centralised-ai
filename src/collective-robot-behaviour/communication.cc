/* Communication.c
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-10-23.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: Communication file to get information from grSim.
 * License: See LICENSE file for license details.
 *==============================================================================
 */
/* PyTorch C++ API library  */
#include <torch/torch.h>
/* Projects .h files. */
#include "network.h"
namespace centralised_ai {
namespace collective_robot_behaviour {
  /* Get the current state*/
  torch::Tensor GetStates() {
    /* Example state data stored in a std::vector*/
    torch::Tensor state_vector = torch::randn({1,1,input_size});
    /*Convert the std::vector to a tensor and reshape if needed*/
    return state_vector;
  }
  /*Get the agents reward*/
  torch::Tensor GetRewards() {
    // Generate a tensor of shape [1, 6] with random values
    auto rewards = torch::randn({1, amount_of_players_in_team}); // Adjust the second dimension if your number of players changes
    return rewards; // Return the rewards tensor
  }
}/* namespace centralised_ai */
}/*namespace collective_robot_behaviour*/

