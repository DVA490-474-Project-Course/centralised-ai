//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-11-61
// Last modified: 2024-11-06 by Jacob Johansson
// Description: Source file for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#include "world.h"
#include "run_state.h"
#include <torch/torch.h>

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    torch::Tensor RunState::ComputeActionMasks(const torch::Tensor & states)
    {
        return torch::ones({6, 10});
    }

    torch::Tensor RunState::ComputeRewards(const torch::Tensor & states)
    {
        return torch::ones(6);
    }
}
}