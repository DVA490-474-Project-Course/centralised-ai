//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-07
// Last modified: 2024-10-07 by Jacob Johansson
// Description: Headers for utils.h.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include <stdint.h>
#include "utils.h"

namespace centralised_ai{
namespace collective_robot_behaviour{

    Tensor compute_reward_to_go(Tensor rewards, uint32_t num_time_steps){
        
        output = torch.zeros(num_time_steps);

        for (int32_t t = 0; t < num_time_steps; t++){
            output[t] = rewards.slice(0, t, num_time_steps).sum();
        }
        
        return output;
    }

    } /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */