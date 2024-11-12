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
#include <chrono>
#include "network.h"
#include "utils.h"
#include "run_state.h"
#include "../simulation-interface/simulation_interface.h"

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

bool CheckModelParametersMatch(const std::vector<centralised_ai::collective_robot_behaviour::Agents>& saved_models,
                                 const std::vector<centralised_ai::collective_robot_behaviour::Agents>& loaded_models,
                                 const centralised_ai::collective_robot_behaviour::CriticNetwork& saved_critic,
                                 const centralised_ai::collective_robot_behaviour::CriticNetwork& loaded_critic) {
    // Check each agent model
    for (size_t i = 0; i < saved_models.size(); ++i) {
      auto saved_params = saved_models[i].policy_network.lstm->parameters();
      auto loaded_params = loaded_models[i].policy_network.lstm->parameters();

      if (saved_params.size() != loaded_params.size()) {
        std::cerr << "Parameter count mismatch for agent " << i << std::endl;
        return false;
      }

      for (size_t j = 0; j < saved_params.size(); ++j) {
        if (!saved_params[j].equal(loaded_params[j])) {
          std::cerr << "Parameter mismatch in agent " << i << " at parameter " << j << std::endl;
          return false;
        }
      }
    }

    // Check critic model parameters
    auto saved_critic_params = saved_critic.lstm->parameters();
    auto loaded_critic_params = loaded_critic.lstm->parameters();

    if (saved_critic_params.size() != loaded_critic_params.size()) {
      std::cerr << "Parameter count mismatch in critic network." << std::endl;
      return false;
    }

    for (size_t j = 0; j < saved_critic_params.size(); ++j) {
      if (!saved_critic_params[j].equal(loaded_critic_params[j])) {
        std::cerr << "Parameter mismatch in critic network at parameter " << j << std::endl;
        return false;
      }
    }

    // If no mismatches are found
    std::cout << "All parameters match successfully!" << std::endl;
    return true;
  }

static std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor> ResetHidden(){
  std::vector<Trajectory> trajectories;
  HiddenStates reset_states;
  reset_states.ct_p = torch::zeros({1, 1, hidden_size});
  reset_states.ht_p = torch::zeros({1, 1, hidden_size});


  Trajectory initial_trajectory;
  for (int i = 0; i < amount_of_players_in_team; i++) {
    initial_trajectory.hidden_p.push_back(reset_states);
  }

  trajectories.push_back(initial_trajectory);

  torch::Tensor act_prob = torch::zeros({num_actions});
  torch::Tensor action;

  return std::make_tuple(trajectories, act_prob, action);
};

std::vector<DataBuffer> MappoRun(std::vector<Agents> Models, CriticNetwork critic,ssl_interface::AutomatedReferee & referee,
  ssl_interface::VisionClient & vision_client, Team own_team,
  std::vector<robot_controller_interface::simulation_interface::SimulationInterface> simulation_interfaces) {

  std::vector<DataBuffer> data_buffer; /*Initialise data buffer D. Reset each step*/
  Team opponent_team = ComputeOpponentTeam(own_team);
  std::vector<Trajectory> trajectories;
  torch::Tensor act_prob;
  torch::Tensor action;
  RunState run_state;

  /* Gain enough batch for training */
  for (int i = 1; i <= batch_size; i++) {
    /*Reset/initialise hidden states for timestep 0*/
    std::tie(trajectories,act_prob,action) = ResetHidden();
    torch::Tensor state = GetStates(referee,vision_client,own_team,opponent_team); /*Get current state as vector*/

    /*Loop for amount of timestamps in each bach */
    for (int timestep = 1; timestep < max_timesteps; timestep++) {
      /*Initialise values*/
      torch::Tensor actions_agents = torch::zeros({amount_of_players_in_team});
      std::vector<float> rewards;
      Trajectory exp;
      HiddenStates new_states;
      /*GET GAMESTATE*/
      torch::Tensor prob_actions_stored(torch::zeros({amount_of_players_in_team,num_actions}));

      /* Get hidden states and output probabilities for critic network, input is state and previous timestep (initialised values)*/
      auto [valNetOutput, V_hx, V_cx] = critic.Forward(state, trajectories[timestep-1].hidden_v.ht_p, trajectories[timestep-1].hidden_v.ct_p);

      /* For each agent in one timestep, get probabilities and hidden states*/
      for (auto& agent : Models) {
        /*Get action probabilities and hidden states, input is previous timestep hidden state for the robots index*/
        auto agent_state = state.clone();
        agent_state.index({0, 0, 0}) = agent.robotId;
        auto [act_prob, hx_new, ct_new] = agent.policy_network.Forward(
          agent_state, trajectories[timestep - 1].hidden_p[agent.robotId].ht_p,
          trajectories[timestep - 1].hidden_p[agent.robotId].ct_p);

        prob_actions_stored.index_put_({agent.robotId}, act_prob);
        /*Save hidden states*/
        new_states.ht_p = hx_new;
        new_states.ct_p = ct_new;
        exp.hidden_p.push_back(new_states); /*Store hidden states*/

      } /*end for agent*/

      /*Get the actions with higest probabilities for each agent*/
      exp.actions = std::get<1>(prob_actions_stored.max(1));

      //std::cout << exp.actions << std::endl;
      SendActions(simulation_interfaces,exp.actions);
      /*
      *NEEDS TO ME IMPLEMENTED:
      *execute actions_prob
      *from actions_agents
      *actionmask
      */
      // Decalare in actions_agents which actions each agent does
      /*Update all values*/

      exp.actions_prob = prob_actions_stored;
      exp.actions = actions_agents;
      exp.state = state;
      exp.criticvalues = valNetOutput.squeeze().expand({amount_of_players_in_team});
      exp.rewards = run_state.ComputeRewards(state.squeeze(0).squeeze(0), {-0.001, 500, 0.0001, 0.0001}).expand({1, amount_of_players_in_team});

      exp.hidden_v.ht_p = V_hx;
      exp.hidden_v.ct_p = V_cx;
      trajectories.push_back(exp); /*Store into trajectories*/
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      state = GetStates(referee,vision_client,own_team,opponent_team);

    } /*end for timestep*/

    /*Erase initial trajectory*/
    trajectories.erase(trajectories.begin());
    /*Amount of steps the model will be trained for*/
    std::vector<torch::Tensor> criticvalues;
    std::vector<torch::Tensor> reward_arr;

    for (auto& trajectory: trajectories) {
      reward_arr.push_back(trajectory.rewards);
      torch::Tensor single_critic_value = trajectory.criticvalues[0].view({1,1});
      criticvalues.push_back(single_critic_value); /*Reformat to use in function*/
    }
    torch::Tensor rewards_tensor = torch::cat(reward_arr, /*dim=*/0);
    torch::Tensor critic_values_array = torch::cat(criticvalues, /*dim=*/0);
    /*compute GAE and reward-go-to*/
    auto TemporalDifference = ComputeTemporalDifference(critic_values_array,rewards_tensor,0.9);
    torch::Tensor A =  ComputeGeneralAdvantageEstimation(TemporalDifference,0.99, 0.95);

    torch::Tensor rew_sum = rewards_tensor.sum(1);
    torch::Tensor R = ComputeRewardToGo(rew_sum,0.99);

    /*Split amount of timesteps in trajectories to L*/
    int L = max_timesteps-1;
    for (int l = 0; l < max_timesteps/L; l++) /* T/L */ {
      /*Add each Trajectory into dat.t value for all timesteps in chunk*/
      DataBuffer dat;
      for (int i = l; i < l+L; i++) {
        dat.t.push_back(trajectories[i]);
      }

      /*A and R is callculated in each and a small batch saved*/
      dat.A = A.slice(0, l, std::min(l + L, max_timesteps));
      dat.R = R.slice(0, l, std::min(l + L, max_timesteps));

      data_buffer.push_back(dat); /*Store [t, A,R] in D (DataBuffer)*/
    } /*end chunks loop*/
    //std::cerr <<"Batch size is"<< i << std::endl;
  }/*end for batch_size*/
  return data_buffer;
}


  /*
  * The full Mappo function for robot decision-making and training.
  * Follows the algorithm from the paper: "The Surprising Effectiveness of PPO in Cooperative
  * Multi-Agent Games" by Yu et al., available at: https://arxiv.org/pdf/2103.01955
  */
void Mappo_Update(std::vector<Agents> Models,CriticNetwork critic, std::vector<DataBuffer> data_buffer,std::vector<Agents> &old_net, CriticNetwork &old_net_critic) {

  int len = data_buffer.size();
  std::vector<DataBuffer> min_batch; /* Mini-batch */
  for (int k = 1; k <= 2; k++) {
    /* Should be k = 1 */
    int rand_index = torch::randint(0, len, {1}).item<int>();
    auto rand_batch = data_buffer[rand_index]; /* Take random saved chunks */

    int t_len = rand_batch.t.size();

    // Create state tensor for the batch of all sequences
    torch::Tensor state_read = torch::zeros({1, t_len, input_size}); // [batch_size, sequence_len, input_size]
    // Create hidden state and cell state tensors for all agents
    torch::Tensor ht_read = torch::zeros({amount_of_players_in_team, t_len, hidden_size}); // [batch_size, sequence_len, hidden_size]
    torch::Tensor ct_read = torch::zeros({amount_of_players_in_team, t_len, hidden_size}); // [batch_size, sequence_len, hidden_size]

    // Populate the state, hidden state, and cell state for each timestep
    for (int t = 0; t < t_len; t++) {
      state_read[0][t] = rand_batch.t[t].state.squeeze();
      // Now you can send the entire batch (ht_read, ct_read, and state_read) to the model for each agent
      for (int agent = 0; agent < amount_of_players_in_team; agent++) {
        ht_read[agent][t] = rand_batch.t[t].hidden_p[agent].ht_p.squeeze();
        ct_read[agent][t] = rand_batch.t[t].hidden_p[agent].ct_p.squeeze();
      }
    }
    for (auto agent : Models) {
      auto agent_state_read = state_read;
      for (int t = 0; t < t_len; ++t) {
        agent_state_read[0][t][0] = agent.robotId;  // Change the first feature value at each timestep
      }
      agent.policy_network.Forward(agent_state_read, ht_read[agent.robotId].unsqueeze(0), ct_read[agent.robotId].unsqueeze(0));
    }
    min_batch.push_back(rand_batch);
  }

  /*Make the arrays fit update functions*/
  int64_t length = static_cast<int64_t>(min_batch[0].t.size()); /*Timesteps in batch*/
  torch::Tensor old_predicts_p=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor new_predicts_p=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor general_arr=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor all_actions_probs = torch::zeros({length,amount_of_players_in_team,num_actions});

  torch::Tensor old_predicts_c=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor new_predicts_c=torch::zeros({length,amount_of_players_in_team});
  torch::Tensor critic_values = torch::zeros({length,amount_of_players_in_team});
  torch::Tensor reward_arr_minbatch = torch::zeros({length,1});

  for (int samp_i = 0; samp_i < min_batch.size(); samp_i++) {
    /*Each sample in mini batch*/
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
        new_predicts_p[t][agent] = min_batch[samp_i].t[t].actions_prob[agent][act];

        all_actions_probs[t] = min_batch[samp_i].t[t].actions_prob[0];

        torch::Tensor output_old_c = std::get<0>(old_ci).squeeze();
        old_predicts_c[t][agent] = output_old_c;
      } /*end agent loop*/

      new_predicts_c[t] = min_batch[samp_i].t[t].criticvalues;
      reward_arr_minbatch[t] = min_batch[samp_i].R[t];
    }
  }

  auto probratio = ComputeProbabilityRatio(new_predicts_p,old_predicts_p);
  auto policy_entropy = ComputePolicyEntropy(all_actions_probs,0.9);

  auto policyloss = ComputePolicyLoss(min_batch[0].A,probratio,0.9,policy_entropy);
  //why minbtach[0]------------------------------------------------------------
  auto norm_rew = NormalizeRewardToGo(reward_arr_minbatch);
  auto critic_loss = ComputeCriticLoss(new_predicts_c, old_predicts_c,norm_rew,0.9);

  //old_net[0].policy_network.lstm->parameters()[0].data() = Models[0].policy_network.lstm->parameters()[0].data();

  CheckModelParametersMatch(old_net,Models,old_net_critic,critic);

  UpdateNets(Models,critic,policyloss,critic_loss); /*update networks*/
  SaveModels(Models,critic); /*save models of all networks(policy and critic)*/

  std::cout << "Old net validation: " << std::endl;
  CheckModelParametersMatch(old_net,Models,old_net_critic,critic);

  std::cout << "Training of buffer done! " << std::endl;
  std::cout << "==============================================" << std::endl;
  std::cout << "Policy loss: " << policyloss << std::endl;
  std::cout << "Critic loss: " << critic_loss << std::endl;

}


}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/