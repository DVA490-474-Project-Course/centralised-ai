//
// Created by viktor on 2024-11-07.
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/network.h"
#include "../../src/common_types.h"


namespace centralised_ai
{
namespace collective_robot_behaviour
{

TEST(CreateAgentsTest, AgentsTest) {
  std::vector<Agents> output_agents1 = CreateAgents(1);
  EXPECT_EQ(output_agents1.size(), 1);

  std::vector<Agents> output_agents2 = CreateAgents(2);
  EXPECT_EQ(output_agents2.size(), 2);

  std::vector<Agents> output_agents6 = CreateAgents(6);
  EXPECT_EQ(output_agents6.size(), 6);
}

TEST(UpdateNetwork, Update) {
  // Create agents and networks
  auto policynet = CreateAgents(1);
  CriticNetwork critic_network;

  // Clone the initial parameters of the policy network (deep copy)
  std::vector<torch::Tensor> old_critic_params;
  for (const auto& param : critic_network.parameters()) {
    old_critic_params.push_back(param.clone());
  }
  std::vector<torch::Tensor> old_policy_params;
  for (const auto& param : policynet[0].policy_network->parameters()) {
    old_policy_params.push_back(param.clone());
  }

  // Define dummy input for forward pass
  auto input = torch::randn({1, 1, input_size});
  auto hx = torch::randn({1, 1, hidden_size});

  // Forward pass through the critic network
  auto [output, hc] = critic_network.Forward(input, hx);

  // Example target for critic network
  auto target_critic = torch::rand({1}, torch::kFloat);
  std::cout << target_critic << std::endl;
  auto critic_loss = torch::mse_loss(output, target_critic);  // Compute the critic loss

  // Forward pass through the policy network (dummy output for this example)
  auto [policy_output,hp] = policynet[0].policy_network->Forward(input, hx);

  // Example target for policy network (for illustration, can be replaced with actual target)
  auto target_policy = torch::rand({1,num_actions}, torch::kFloat);
  auto policy_loss = torch::mse_loss(policy_output, target_policy);  // Compute the policy loss

  // Update networks using the policy and critic loss
  UpdateNets(policynet, critic_network, policy_loss, critic_loss);  // Assuming UpdateNets handles both updates

  auto new_critic_params = critic_network.parameters();
  for (size_t i = 0; i < old_critic_params.size(); ++i) {
    EXPECT_FALSE(old_critic_params[i].equal(new_critic_params[i]))
        << "Critic network parameter " << i << " did not update.";
  }

  auto new_policy_params = policynet[0].policy_network->parameters();
  for (size_t i = 0; i < old_policy_params.size(); ++i) {
    EXPECT_FALSE(old_policy_params[i].equal(new_policy_params[i]))
        << "Policy network parameter " << i << " did not update.";
  }
}





}
}