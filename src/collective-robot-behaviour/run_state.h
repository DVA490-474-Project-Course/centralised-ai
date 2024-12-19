//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-11-61
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Header for run_state.cc.
// License: See LICENSE file for license details.
//==============================================================================

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_RUNSTATE_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_RUNSTATE_H_

#include "communication.h"
#include "reward.h"
#include "state.h"
#include "torch/torch.h"

namespace centralised_ai {
namespace collective_robot_behaviour {
class RunState : public GameStateBase {
public:
  torch::Tensor ComputeActionMasks(const torch::Tensor& states) override;
  torch::Tensor
  ComputeRewards(const torch::Tensor& states,
                 struct RewardConfiguration reward_configuration) override;
};
} // namespace collective_robot_behaviour
} // namespace centralised_ai

#endif