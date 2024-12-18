//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-11-61
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Header for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef RUN_STATE_H
#define RUN_STATE_H

#include "reward.h"
#include "torch/torch.h"
#include "communication.h"
#include "state.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    class RunState : public GameStateBase
    {
        public:
            torch::Tensor ComputeActionMasks(const torch::Tensor & states) override;
            torch::Tensor ComputeRewards(const torch::Tensor & states, struct RewardConfiguration reward_configuration) override;
    };
}
}

#endif