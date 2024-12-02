//
// Created by viktor on 2024-10-31.
//
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "../../src/collective-robot-behaviour/network.h"
#include "../../src/collective-robot-behaviour/config.h"
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

    TEST(ParametersNewTest, NewParameters) {
      auto models = CreateAgents(amount_of_players_in_team);
      auto old_models = CreateAgents(amount_of_players_in_team);
      CriticNetwork critic_network;
      CriticNetwork old_critic_network;
      bool check1 = CheckModelParametersMatch(models, models,critic_network, critic_network);
      EXPECT_TRUE(check1);

      bool check2 = CheckModelParametersMatch(models, old_models, critic_network, old_critic_network);
      EXPECT_FALSE(check2);
    }

    /*
    TEST(SameParametersTest, SameParameters) {
      auto models = CreateAgents(1);
      CriticNetwork critic_network;
      auto params = CheckModelParametersMatch(models, models,critic_network, critic_network);
      EXPECT_TRUE(params);
    }
    */

  }
}