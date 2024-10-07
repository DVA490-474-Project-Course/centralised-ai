//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-10-07 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>
#include <stdint.h>

namespace centralised_ai{
namespace collective_robot_behaviour{

    /* Returns the reward-to-go values, where
        - discounts: The discount for each time step in a (1 X #time_steps) tensor, i.e. discount¹, discount², discount³, ...
        - rewards: The accumulated reward for each time step in a (#time_steps X 1) tensor.
        - num_time_steps: The number of time steps to compute the reward-to-go values over.
    */
    Tensor compute_reward_to_go(Tensor discounts, Tensor rewards, uint32_t num_time_steps);

} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */


#endif /* UTILS_H */

