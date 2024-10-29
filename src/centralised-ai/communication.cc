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

/* Get the current state*/
torch::Tensor get_states() {
    /* Example state data stored in a std::vector*/
    torch::Tensor state_vector = torch::randn({1,1,input_size});
    /*Convert the std::vector to a tensor and reshape if needed*/
    return state_vector;
}
/*Get the agents reward*/
float get_rewards() {
    float reward = 1.0;
    return reward;
}


