//
// Created by viktor on 2024-10-31.
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/network.h"
#include "../../src/common_types.h"
#include "../../src/collective-robot-behaviour/mappo.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

  /*
  TEST(MappoRunTest, MappoRun) {

    std::vector<DataBuffer> a = MappoRun();
    EXPECT_EQ(a.type, DataBuffer);

  }
  */

  /*TEST(ParametersNewTest, NewParameters) {
    auto models = CreateAgents(amount_of_players_in_team);
    auto old_models = CreateAgents(amount_of_players_in_team);
    CriticNetwork critic_network;
    CriticNetwork old_critic_network;
    bool check1 = CheckModelParametersMatch(models, models,critic_network, critic_network);
    EXPECT_TRUE(check1);

    bool check2 = CheckModelParametersMatch(models, old_models, critic_network, old_critic_network);
    EXPECT_FALSE(check2);
  }

  TEST(ResetHidden, HiddenParameters) {
    // Call the ResetHidden function
    std::vector<Trajectory> trajectory;
    torch::Tensor act_prob;
    torch::Tensor action;

    std::tie(trajectory, act_prob, action) = ResetHidden();

    // Check that the trajectories have the correct size
    EXPECT_EQ(trajectory.size(), 1) << "Trajectory vector should contain one trajectory.";
    EXPECT_EQ(trajectory[0].hidden_p.size(), amount_of_players_in_team)
        << "Each trajectory should contain hidden states for all agents.";

    // Check that the hidden states are initialized to zeros
    for (const auto& hidden_state : trajectory[0].hidden_p) {
      EXPECT_TRUE(hidden_state.ct_p.equal(torch::zeros({1, 1, hidden_size})))
          << "Cell state should be initialized to zeros.";
      EXPECT_TRUE(hidden_state.ht_p.equal(torch::zeros({1, 1, hidden_size})))
          << "Hidden state should be initialized to zeros.";
    }

    // Check that the action probabilities are initialized to zeros
    EXPECT_TRUE(act_prob.equal(torch::zeros({num_actions})))
        << "Action probabilities should be initialized to zeros.";

    // Ensure the action tensor is uninitialized (should have no values yet)
    EXPECT_EQ(action.numel(), 0) << "Action tensor should initially be uninitialized.";
  }
  */


}
}