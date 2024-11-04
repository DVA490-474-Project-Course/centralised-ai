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


  /*Amount of steps the model will be trained for*/
  while (steps < step_max) {
  std::vector<DataBuffer> data_buffer; /*Initialise data buffer D. Reset each step*/
  /* Gain enough batch for training */
  for (int i = 1; i <= batch_size; i++) {
    /*Reset/initialise hidden states for timestep 0*/
    auto[trajectories,act_prob,action] = ResetHidden();
    std::vector<torch::Tensor> criticvalues = {torch::zeros({1,1})};
    std::cout << criticvalues << std::endl;
    /*Loop for amount of timestamps in each bach */
    for (int timestep = 1; timestep < max_timesteps; timestep++) {

    /*Initialise values*/
    std::vector<int> actions_agents;
    std::vector<float> rewards;
    Trajectory exp;
    HiddenStates new_states;
    torch::Tensor state = GetStates(); /*Get current state as vector*/
    /* Get hidden states and output probabilities for critic network, input is state and previous timestep (initialised values)*/
    auto [valNetOutput, V_hx, V_cx] = critic.Forward(state, trajectories[timestep-1].hidden_p[0].ht_p, trajectories[timestep-1].hidden_p[0].ct_p);
    criticvalues.push_back(valNetOutput);
    /* For each agent in one timestep, get probabilities and hidden states*/
    for (auto& agent : Models) {
      /*Get action probabilities and hidden states, input is previous timestep hidden state for the robots index*/
      auto [act_prob, hx_new, ct_new] = agent.policy_network.Forward(state,trajectories[timestep-1].hidden_p[agent.robotId].ht_p,trajectories[timestep-1].hidden_p[agent.robotId].ct_p);
      /*store action probabilities for the agent*/
      exp.actions = act_prob.reshape({num_actions});;
      /*Store the higest probablity of the actions*/
      actions_agents.push_back(torch::argmax(act_prob).item().toInt());

      /*Save hidden states*/
      new_states.ht_p = hx_new;
      new_states.ct_p = ct_new;
      exp.hidden_p.push_back(new_states); /*Store hidden states*/

      } /*end for agent*/

    /*
    *NEEDS TO ME IMPLEMENTED:
    *execute actions
    *from actions_agents
    */

    /*Update all values*/
    exp.state = state;
    exp.rewards = GetRewards();
    exp.new_state = GetStates();
    exp.hidden_v.ht_p = V_hx;
    exp.hidden_v.ct_p = V_cx;
    trajectories.push_back(exp); /*Store into trajectories*/

    } /*end for timestep*/

  std::vector<torch::Tensor> reward_arr;
  for (auto& trajectory: trajectories) {
    std::cerr << trajectory.rewards << std::endl;
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
  int L = 6;
  for (int l = 0; l < max_timesteps/L; l++) /* T/L */
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
  for (int k = 0; k <= 10; k++) {
    int rand_index = torch::randint(0, len, {1}).item<int>();
    min_batch.push_back(data_buffer[rand_index]); /*Take random saved chunks*/
    /*Send in the minibatch and update the hidden states by the saved states and hidden values.*/
    /*for each timestep in batch*/
    for (int i = 0; i < min_batch[k].t.size(); i++) {
      auto state_read = min_batch[0].t[i].state; /*get state for the timestep in minibatch*/

      /*
      *For each agents actions in one timestep
      *UPDATE SO ALL TIMESTEPS GET SENT AS ONE VECTOR AS STATE
      */

      /*Update the hidden stated for policy and critic networks
       *for each data chunk in the mini-batch b*/
      for (int agent = 0; agent < amount_of_players_in_team; agent++)
      {
        auto hx_read = min_batch[k].t[i].hidden_p[agent].ht_p;
        auto ct_read = min_batch[k].t[i].hidden_p[agent].ct_p;
        /*update LSTM hidden states for policy from first hidden state in data chunk.*/
        Models[agent].policy_network.Forward(state_read,hx_read,ct_read);
        std::cerr << state_read << std::endl;
        /*
         * FIX SO WE UPDATE A STATE VECTOR OF MULTIPLE TIMESTEPS INSTEAD OF USING LOOP
         */
      }
      auto hv_read = min_batch[k].t[i].hidden_v.ht_p;
      auto cv_read = min_batch[k].t[i].hidden_v.ct_p;
      /*update LSTM hidden states for critic network from first hidden state in data chunk.*/
      critic.Forward(state_read,hv_read,cv_read);
    }
} /*end for*/


  /*ADAM UPDATE NETWORKS*/

  torch::Tensor old_predicts=torch::zeros({sizeof(data_buffer),amount_of_players_in_team});
  torch::Tensor new_predicts=torch::zeros({sizeof(data_buffer),amount_of_players_in_team});;
    for (int t = 0; t < sizeof(data_buffer); ++t) {
      for (int agent = 0; agent < amount_of_players_in_team; ++agent) {
        // Pass each agent's state at the current timestep through the policy network
        auto state = data_buffer[t].t[agent].state; /*Get saved state*/
        auto hc = data_buffer[t].t[agent].hidden_p[agent].ht_p; /*Get saved hidden state*/
        auto ct = data_buffer[t].t[agent].hidden_p[agent].ct_p; /*Get saved memory cell*/
        auto old_pi = old_net[agent].policy_network.Forward(state,hc,ct); /*Make new predictions*/

        torch::Tensor output_old = std::get<0>(old_pi); /*Output from old_net*/
        auto act = torch::argmax(data_buffer[t].t[agent].actions); /*Get the done action from previously*/
        old_predicts[t][agent] = output_old.index({0,act});
        //new_predicts[t][agent] = data_buffer[t].t[agent].actions[act];
        //std::cout << act << std::endl;
        //std::cout << data_buffer[t].t[agent].actions  << std::endl;
        //std::cout << data_buffer[t].t[agent].actions[act] << "act value" << std::endl;
        /*SIZE IS DIFFERENT FOR ACTIONS SOMEHOW!!??? 1,9 and 9 at start*/
        std::cout << "Shape: " << data_buffer[t].t[agent].actions.sizes() << std::endl;
        //std::cout << old_predicts << std::endl;
        //std::cout << new_predicts << std::endl;


      }
    }

  //ComputeProbabilityRatio()
  UpdateNets(Models, critic,data_buffer); /*update networks*/
  SaveModels(Models,critic); /*save models of all networks(policy and critic)*/

  steps++;
  } /*end while*/
}

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/