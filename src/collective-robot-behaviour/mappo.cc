/* Mappo.cc
*==============================================================================
* Author: Viktor Eriksson
* Creation date: 2024-10-23
* Last modified: 2024-10-24 by Viktor Eriksson
* Description: Multi Agent Proximal Policy Optimization (Mappo), training algorithm
* connected into grSim so simulate the robocup competition.
* License: See LICENSE file for license details.
*==============================================================================
*/

/* C++ library */
#include <tuple>
#include <vector>

/* PyTorch C++ API library  */
#include <torch/torch.h>

/* Projects .h files for communication functions. */
#include "communication.h"
#include "network.h"
#include "utils.h"

/*Configuration values*/
extern int max_timesteps;
extern int steps; /*move into mappo------------------------*/
extern int step_max;
extern int batch_size;
extern int amount_of_players_in_team;

namespace centralised_ai {
namespace collective_robot_behaviour {
/*
*
*/
static std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor> ResetHidden(){
  std::vector<Trajectory> trajectories;
  HiddenStates reset_states;
  Trajectory initial_trajectory;
  for (int i = 0; i < amount_of_players_in_team; i++) {
    initial_trajectory.hidden_p.push_back(reset_states);
  }

  trajectories.push_back(initial_trajectory);

  torch::Tensor act_prob = torch::zeros({num_actions});
  torch::Tensor action;

  return std::make_tuple(trajectories, act_prob, action);
};




/*
* The full Mappo function for robot decision-making and training.
* Follows the algorithm from the paper: "The Surprising Effectiveness of PPO in Cooperative
* Multi-Agent Games" by Yu et al., available at: https://arxiv.org/pdf/2103.01955
*/
  void Mappo(std::vector<Agents> Models,CriticNetwork critic ) {

  auto old_net = Models; /*Initalise old network*/
  auto old_net_critic = critic;

  /*Amount of steps the model will be trained for*/
  while (steps < step_max) {
  std::vector<DataBuffer> data_buffer; /*Initialise data buffer D. Reset each step*/
  /* Gain enough batch for training */
  for (int i = 1; i <= batch_size; i++) {
    /*Reset/initialise hidden states for timestep 0*/
    auto[trajectories,act_prob,action] = ResetHidden();
    std::vector<torch::Tensor> criticvalues = {torch::zeros({1,1})};

    /*Loop for amount of timestamps in each bach */
    for (int timestep = 1; timestep < max_timesteps; timestep++) {

    /*Initialise values*/
    torch::Tensor actions_agents = torch::zeros({amount_of_players_in_team});
    std::vector<float> rewards;
    Trajectory exp;
    HiddenStates new_states;
    torch::Tensor state = torch::zeros({1,1,input_size}); //GetStates(); /*Get current state as vector*/
    torch::Tensor prob_actions_stored(torch::zeros({amount_of_players_in_team}));

    /* Get hidden states and output probabilities for critic network, input is state and previous timestep (initialised values)*/
    auto [valNetOutput, V_hx, V_cx] = critic.Forward(state, trajectories[timestep-1].hidden_p[0].ht_p, trajectories[timestep-1].hidden_p[0].ct_p);
    criticvalues.push_back(valNetOutput);
    /* For each agent in one timestep, get probabilities and hidden states*/
    for (auto& agent : Models) {
      /*Get action probabilities and hidden states, input is previous timestep hidden state for the robots index*/
      auto [act_prob, hx_new, ct_new] = agent.policy_network.Forward(state,trajectories[timestep-1].hidden_p[agent.robotId].ht_p,trajectories[timestep-1].hidden_p[agent.robotId].ct_p);
      act_prob = act_prob.squeeze();
      int act = argmax(act_prob).item().toInt();
      prob_actions_stored[agent.robotId] = act_prob[act];
      //exp.all_actions.push_back(act_prob.squeeze());
      exp.all_actions[0][agent.robotId] = act_prob;
      /*Store the higest probablity of the actions_prob*/
      actions_agents[agent.robotId] = act;

      /*Save hidden states*/
      new_states.ht_p = hx_new;
      new_states.ct_p = ct_new;
      exp.hidden_p.push_back(new_states); /*Store hidden states*/

      } /*end for agent*/

    /*
    *NEEDS TO ME IMPLEMENTED:
    *execute actions_prob
    *from actions_agents
    */

    /*Update all values*/
    exp.actions_prob = prob_actions_stored;
    exp.actions = actions_agents;
    exp.state = state;
    exp.criticvalues = valNetOutput.squeeze().expand({amount_of_players_in_team});
    exp.rewards = torch::zeros({1,6}); //GetRewards();
    exp.new_state = torch::zeros({1,1,input_size}); //GetStates();
    exp.hidden_v.ht_p = V_hx;
    exp.hidden_v.ct_p = V_cx;
    trajectories.push_back(exp); /*Store into trajectories*/

    } /*end for timestep*/

  std::vector<torch::Tensor> reward_arr;
  for (auto& trajectory: trajectories) {
    reward_arr.push_back(trajectory.rewards);
  }
  torch::Tensor rewards_tensor = torch::cat(reward_arr, /*dim=*/0);
  torch::Tensor critic_values_array = torch::cat(criticvalues, /*dim=*/0);

  /*compute GAE and reward-go-to*/
  auto TemporalDifference = ComputeTemporalDifference(critic_values_array,rewards_tensor,0.9);
  torch::Tensor A =  ComputeGeneralAdvantageEstimation(TemporalDifference,0.99, 0.95);

  torch::Tensor rew_sum = rewards_tensor.sum(1);
  torch::Tensor R = ComputeRewardToGo(rew_sum,0.99);

  /*Split trajectories into chunks of lenght L*/
  int L = 2;
  for (int l = 1; l < max_timesteps/L; l++) /* T/L */
  {
    /*Add each Trajectory into dat.t value for all timesteps in chunk*/
    DataBuffer dat;
    for (int i = l; i < l+L; i++) {
      dat.t.push_back(trajectories[i]);
    }

    /*A and R is callculated in each and a small batch saved*/
    dat.A = A.slice(0, l, std::min(l + L, max_timesteps));
    dat.R = R.slice(0, l, std::min(l + L, max_timesteps));

    data_buffer.push_back(dat); /*Store [t, A,R] in D (DataBuffer)*/
  } /*end for*/
} /*end for batch_size*/

  int len = data_buffer.size();
  std::vector<DataBuffer> min_batch; /*b (minbatch)*/
  /* Random mini-batch from D with all agent data*/
  for (int k = 1; k <= 1; k++) { /*should be k = 1*/
    int rand_index = torch::randint(1, len, {1}).item<int>();
    auto rand_batch = data_buffer[rand_index];/*Take random saved chunks*/
    /*Send in the minibatch and update the hidden states by the saved states and hidden values.*/
    /*for each timestep in batch*/
    for (int i = 0; i <= rand_batch.t.size()-1; i++) {
      auto state_read = rand_batch.t[i].state; /*get state for the timestep in minibatch*/
      /*
      *For each agents actions_prob in one timestep
      *UPDATE SO ALL TIMESTEPS GET SENT AS ONE VECTOR AS STATE
      */
        /*Update the hidden stated for policy and critic networks
       *for each data chunk in the mini-batch b*/
      for (int agent = 0; agent < amount_of_players_in_team; agent++)
      {
        auto hx_read = rand_batch.t[i].hidden_p[agent].ht_p;
        auto ct_read = rand_batch.t[i].hidden_p[agent].ct_p;

        /*update LSTM hidden states for policy from first hidden state in data chunk.*/
        Models[agent].policy_network.Forward(state_read,hx_read,ct_read);
        /*
         * FIX SO WE UPDATE A STATE VECTOR OF MULTIPLE TIMESTEPS INSTEAD OF USING LOOP
         */
      }
      auto hv_read = rand_batch.t[i].hidden_v.ht_p;
      auto cv_read = rand_batch.t[i].hidden_v.ct_p;
      /*update LSTM hidden states for critic network from first hidden state in data chunk.*/
      critic.Forward(state_read,hv_read,cv_read);
    }
    min_batch.push_back(rand_batch); //push rand_batch to minbatch
  } /*end for*/


  /*ADAM UPDATE NETWORKS*/

  /*Make the arrays fit update functions*/
  int64_t length = static_cast<int64_t>(min_batch[0].t.size()); /*Timesteps*/
  torch::Tensor old_predicts_p=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor new_predicts_p=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor general_arr=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor all_actions_probs = torch::zeros({length,amount_of_players_in_team,num_actions});

  torch::Tensor old_predicts_c=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor new_predicts_c=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor critic_values = torch::zeros({length,amount_of_players_in_team});
  torch::Tensor reward_arr = torch::zeros({length,1});

  for (int samp_i = 0; samp_i < min_batch.size(); samp_i++) { /*Each sample in mini batch*/
    for (int t = 0; t < min_batch[samp_i].t.size(); ++t) { /*each timestep in batch*/
      for (int agent = 0; agent < amount_of_players_in_team; ++agent) { /*each agent in timestep*/

        /* Pass each agent's state at the current timestep through the policy network*/
        auto state = min_batch[samp_i].t[t].state; /*Get saved state for agent*/
        auto hs_p = min_batch[samp_i].t[t].hidden_p[agent].ht_p; /*Get saved hidden state for agent*/
        auto mc_p = min_batch[samp_i].t[t].hidden_p[agent].ct_p; /*Get saved memory cell for agent*/
        auto old_pi = old_net[agent].policy_network.Forward(state,hs_p,mc_p); /*Make new predictions*/

        auto hs_c = min_batch[samp_i].t[t].hidden_v.ht_p;
        auto mc_c = min_batch[samp_i].t[t].hidden_v.ct_p;
        auto old_ci = old_net_critic.Forward(state,hs_c,mc_c);
        /*Save new and old predicts for each agent*/
        torch::Tensor output_old_p = std::get<0>(old_pi).squeeze(); /*Output from old_net*/
        auto act = min_batch[samp_i].t[t].actions[agent].item<int>();
        old_predicts_p[t][agent] = output_old_p[act];
        new_predicts_p[t][agent] = min_batch[samp_i].t[t].actions_prob[agent];

        all_actions_probs[t] = min_batch[samp_i].t[t].all_actions[0];

        torch::Tensor output_old_c = std::get<0>(old_ci).squeeze();
        old_predicts_c[t][agent] = output_old_c;
      } /*end agent loop*/
      new_predicts_c[t] = min_batch[samp_i].t[t].criticvalues;
      reward_arr[t] = min_batch[samp_i].R[t];
    }
  }
  auto probratio = ComputeProbabilityRatio(new_predicts_p,old_predicts_p);
  auto policy_entropy = ComputePolicyEntropy(all_actions_probs,0.9);
  auto policyloss = ComputePolicyLoss(min_batch[0].A,probratio,0.9,policy_entropy);



  auto norm_rew = NormalizeRewardToGo(reward_arr);
  auto critic_loss = ComputeCriticLoss(new_predicts_c, old_predicts_c,norm_rew,0.9);


  UpdateNets(Models,critic,policyloss,critic_loss); /*update networks*/
  SaveModels(Models,critic); /*save models of all networks(policy and critic)*/
  std::cout << "Step is done: " << steps << std::endl;
  steps++;
  } /*end while*/
}

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/