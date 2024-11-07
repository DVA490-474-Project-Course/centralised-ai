//
// Created by viktor on 2024-11-07.
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/network.h"

int max_timesteps = 100;
int steps = 0; /*move into mappo------------------------*/
int step_max = 0;
int batch_size = 30;
int amount_of_players_in_team = 6;
int input_size = 43; // Number of input features
int num_actions = 4;
int hidden_size = 7;


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

}
}