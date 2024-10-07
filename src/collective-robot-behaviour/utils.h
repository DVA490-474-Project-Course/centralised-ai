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
        - discounts: The discount for each time step in a [1, num_time_steps] tensor, i.e. discount¹, discount², discount³, ...
        - rewards: The accumulated reward for each time step in a tensor of shape [num_time_steps, 1].
        - num_time_steps: The number of time steps to compute the reward-to-go values over.
    */
    Tensor compute_reward_to_go(const Tensor& rewards, uint32_t num_time_steps);

    /* Returns the general advantage estimation represented by a tensor of shape [num_time_steps, num_agents], where
        - temporal_differences: Tensor of shape [num_time_steps, num_agents]
        - discount: Discount factor.
        - gae_parameter: GAE parameter.
    */
    Tensor compute_general_advantage_estimation(const Tensor& temporal_differences, double discount, double gae_parameter);

} /* namespace centralised_ai */
} /* namespace collective_robot_behaviour */


#endif /* UTILS_H */

