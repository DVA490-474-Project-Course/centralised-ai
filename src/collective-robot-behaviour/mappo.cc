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
* Check if the old network have the same as the new network, prints out wrong parameter of network.
* Used to verify if the networks are getting updated.
* Returns true if networks match, False if networks doesnt match
*/
static bool CheckModelParametersMatch(const std::vector<centralised_ai::collective_robot_behaviour::Agents>& saved_models,
                                 const std::vector<centralised_ai::collective_robot_behaviour::Agents>& loaded_models,
                                 const centralised_ai::collective_robot_behaviour::CriticNetwork& saved_critic,
                                 const centralised_ai::collective_robot_behaviour::CriticNetwork& loaded_critic) {
  bool check = true;
    for (size_t i = 0; i < saved_models.size(); ++i) {
      auto saved_params = saved_models[i].policy_network->lstm->parameters();
      auto loaded_params = loaded_models[i].policy_network->lstm->parameters();

      if (saved_params.size() != loaded_params.size()) {
        std::cerr << "Parameter count mismatch for agent " << i << std::endl;
        check = false;
      }

      for (size_t j = 0; j < saved_params.size(); ++j) {
        if (!saved_params[j].equal(loaded_params[j])) {
          std::cerr << "Parameter mismatch in agent " << i << " at parameter " << j << std::endl;
          check = false;
        }
      }
    }

    // Check critic model parameters
    auto saved_critic_params = saved_critic.lstm->parameters();
    auto loaded_critic_params = loaded_critic.lstm->parameters();

    if (saved_critic_params.size() != loaded_critic_params.size()) {
      std::cerr << "Parameter count mismatch in critic network." << std::endl;
      check = false;
    }

    for (size_t j = 0; j < saved_critic_params.size(); ++j) {
      if (!saved_critic_params[j].equal(loaded_critic_params[j])) {
        std::cerr << "Parameter mismatch in critic network at parameter " << j << std::endl;
        check = false;
      }
    }

    // If no mismatches are found
  if (check == true) {
    std::cout << "All parameters match successfully!" << std::endl;
    return true;
  }
  else {
    return false;
  }

  }

/*Reset the initalise state of the networks and hidden states.
 *This for timestep 0 used in MappoRun
 */
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

/*
 * Where the agents run and training-data getting received.
 */
std::vector<DataBuffer> MappoRun(std::vector<Agents> Models, CriticNetwork critic,ssl_interface::AutomatedReferee & referee,
ssl_interface::VisionClient & vision_client, Team own_team,
std::vector<robot_controller_interface::simulation_interface::SimulationInterface> simulation_interfaces) {

  std::vector<DataBuffer> data_buffer; /*Initialise data buffer D*/
  Team opponent_team = ComputeOpponentTeam(own_team); /*Get opponent team class*/
  std::vector<Trajectory> trajectories; /*Create trajectory vector*/
  torch::Tensor act_prob;
  torch::Tensor action;
  RunState run_state;

  /* Gain enough batches for training */
  for (int i = 1; i <= batch_size; i++) {
    std::tie(trajectories,act_prob,action) = ResetHidden(); /* Reset/initialise hidden states for timestep 0 */
    torch::Tensor state = GetStates(referee,vision_client,own_team,opponent_team); /*Get current state as vector*/
    state = GetStates(referee,vision_client,own_team,opponent_team); /*because of first read gets wrong info, duplicate get state, further investigation needed*/

    /*Loop for amount of timestamps in each bach
     * Start at 1 because timestep 0 is initalised, and hidden states is following previous input (timestep - 1).
     */
    for (int timestep = 1; timestep < max_timesteps; timestep++) {
      /*Initialise values*/
      Trajectory exp;
      HiddenStates new_states;
      torch::Tensor prob_actions_stored(torch::zeros({amount_of_players_in_team,num_actions}));

      /* Get hidden states and output probabilities for critic network, input is state and previous timestep (initialised values)*/
      auto [valNetOutput, V_hx, V_cx] = critic.Forward(state, trajectories[timestep-1].hidden_v.ht_p, trajectories[timestep-1].hidden_v.ct_p);

      /* For each agent in one timestep, get probabilities and hidden states*/
      for (auto& agent : Models) {
        auto agent_state = state.clone();
        agent_state.index({0, 0, 0}) = agent.robotId; /*Update the first index value to robot ID*/

        /*Get action probabilities and hidden states, input is previous timestep hidden state for the robots index*/
        auto [act_prob, hx_new, ct_new] = agent.policy_network->Forward(
          agent_state, trajectories[timestep - 1].hidden_p[agent.robotId].ht_p,
          trajectories[timestep - 1].hidden_p[agent.robotId].ct_p);

        prob_actions_stored.index_put_({agent.robotId}, act_prob[0][0].softmax(0)); /*Store probabilities*/

        /*Save hidden states*/
        new_states.ht_p = hx_new;
        new_states.ct_p = ct_new;
        exp.hidden_p.push_back(new_states); /*Store hidden states*/

      } /*end for agent*/

      /*Get the actions with the highest probabilities for each agent*/
      exp.actions = std::get<1>(prob_actions_stored.max(1));
      //std::cout << prob_actions_stored << std::endl;
      /*
       *Action mask could be implemented here to define legal moves,
       *such as shooting/passing only if player have ball
      */

      SendActions(simulation_interfaces,exp.actions);

      /*Let agents run for 20 ms until next timestep*/
      for(int a = 0; a < 20; a++){
      SendActions(simulation_interfaces,exp.actions);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      // TOOD: Fix to use the new state
      /*Update all values*/
      exp.actions_prob = prob_actions_stored;
      exp.state = state.clone();
      exp.criticvalues = valNetOutput.squeeze(0).expand({amount_of_players_in_team});
      exp.hidden_v.ht_p = V_hx;
      exp.hidden_v.ct_p = V_cx;

      /*Update state and use it for next itteration*/
      state = GetStates(referee,vision_client,own_team,opponent_team);

      /* Get rewards from the actions. */
      exp.rewards = run_state.ComputeRewards(state.squeeze(0).squeeze(0), {-0.001, 500, 0.0001, 0.0001}).expand({1, amount_of_players_in_team});
      trajectories.push_back(exp); /*Store into trajectories*/
    
    } /*end for timestep*/

    /* Erase initial trajectory (First index in trajectories) */
    trajectories.erase(trajectories.begin());

    /*Calculate GAE and Reward To Go from the stored trajectories */
    std::vector<torch::Tensor> criticvalues;
    std::vector<torch::Tensor> reward_arr;

    /*Configure arrays for later functions*/
    for (auto& trajectory: trajectories) {
      reward_arr.push_back(trajectory.rewards);
      torch::Tensor single_critic_value = trajectory.criticvalues[0].view({1,1});
      criticvalues.push_back(single_critic_value); /*Reformat to use in function*/
    }
    torch::Tensor rewards_tensor = torch::cat(reward_arr, /*dim=*/0);
    torch::Tensor critic_values_array = torch::cat(criticvalues, /*dim=*/0);

    /*Compute GAE and Reward To Go*/
    auto TemporalDifference = ComputeTemporalDifference(critic_values_array,rewards_tensor,0.9);
    torch::Tensor A =  ComputeGeneralAdvantageEstimation(TemporalDifference,0.99, 0.95); /*GAE*/
    torch::Tensor rew_sum = rewards_tensor.sum(1);
    torch::Tensor R = ComputeRewardToGo(rew_sum,0.99); /*Reward To Go*/

    /* Split amount of timesteps in trajectories to length L */
    int L = 10;
    for (int l = 0; l < max_timesteps/L; l++) /* T/L */ {
      /*Add each Trajectory into dat.t value for all timesteps in chunk*/
      DataBuffer dat;
      for (int i = l; i < l+L; i++) {
        dat.t.push_back(trajectories[i]);
      }

      /*A and R is saved in small batch intervals*/
      dat.A = A.slice(0, l, std::min(l + L, max_timesteps));
      dat.R = R.slice(0, l, std::min(l + L, max_timesteps));

      data_buffer.push_back(dat); /*Store [t, A,R] in D (DataBuffer)*/
    } /*end chunks loop*/
  }/*end for batch_size*/

  return data_buffer;
}


  /*
  * The full Mappo function for robot decision-making and training.
  * Follows the algorithm from the paper: "The Surprising Effectiveness of PPO in Cooperative
  * Multi-Agent Games" by Yu et al., available at: https://arxiv.org/pdf/2103.01955
  */
void Mappo_Update(std::vector<Agents> &Models,CriticNetwork &critic, std::vector<DataBuffer> data_buffer) {


  int len = data_buffer.size();

  /* Mini-batch */
  std::vector<DataBuffer> min_batch;

  /*Create random min batches that the agents network will update on
   * from papaer, should be set to 1
   */
  int num_mini_batch = 1;
  int mini_batch_size = batch_size / num_mini_batch;
  for (int k = 1; k <= num_mini_batch; k++)
  {
    std::vector<DataBuffer> mini_batch_data;

    // Take a random number of chunks from the data buffer.
    for (int i = 0; i < mini_batch_size; i++) {
      int rand_index = torch::randint(0, len, {1}).item<int>();
      mini_batch_data.push_back(data_buffer[rand_index]);
    }

    /*Create state tensor for the batch of all sequences*/
    int num_chunks = mini_batch_data.size();
    int num_layers = 1;
    torch::Tensor input = torch::zeros({max_timesteps, num_chunks, input_size}); /* [sequence_len, batch_size, input_size] */
    /* Create hidden state and cell state tensors for all agents*/
    torch::Tensor c0_policy = torch::zeros({amount_of_players_in_team, num_layers, num_chunks, hidden_size}); /* [num_players, num_layers, batch_size, hidden_size] */
    torch::Tensor c0_critic = torch::zeros({num_layers, num_chunks, hidden_size}); /* [num_layers, batch_size, hidden_size]*/
    torch::Tensor h0_critic = torch::zeros({num_layers, num_chunks, hidden_size}); /* [num_layers, batch_size, hidden_size]*/
    torch::Tensor h0_policy = torch::zeros({amount_of_players_in_team, num_layers, num_chunks, hidden_size}); /* [num_players, num_layers, batch_size, hidden_size] */

    /* For each chunk in the mini batch, update hidden states from first hidden state.*/
    for (int c = 0; c < mini_batch_data.size(); c++)
    {
        auto chunk = mini_batch_data[c];

        int num_time_steps = chunk.t.size();

        c0_critic[0][c] = chunk.t[0].hidden_v.ct_p.squeeze();
        h0_critic[0][c] = chunk.t[0].hidden_v.ht_p.squeeze();

        for (int t = 0; t < num_time_steps; t++)
        {
          input[t][c] = chunk.t[t].state.squeeze();
        }

        for(int agent = 0; agent < amount_of_players_in_team; agent++)
        {
          c0_policy[agent][0][c] = chunk.t[0].hidden_p[agent].ct_p.squeeze();
          h0_policy[agent][0][c] = chunk.t[0].hidden_p[agent].ht_p.squeeze(); 
        }
    }

    for(int agent = 0; agent < amount_of_players_in_team; agent++)
    {
      auto model = Models[agent];
      
      for (int c = 0; c < num_chunks; c++)
      {
        input[0][c][0] = model.robotId;
      }

      /*Update the policy networks hidden states */
      model.policy_network->Forward(input, h0_policy[agent], c0_policy[agent]);
    }

    /*Update the critic networks hidden states */
    for (int c = 0; c < num_chunks; c++)
    {
      for (int t = 0; t < max_timesteps; t++)
      {
        input[t][c][0] = -1;
      }
    }

    critic.Forward(input, c0_critic, h0_critic);

    /* Push chunk to min batch. */
    for (int c = 0; c < num_chunks; c++)
    {
      min_batch.push_back(mini_batch_data[c]);
    }
  }  

  /*Create the arrays fit update functions*/
  int num_chunks = min_batch.size();
  int64_t length = static_cast<int64_t>(min_batch[0].t.size()); /*Timesteps in batch*/
  torch::Tensor old_predicts_p=torch::zeros({num_chunks, length, amount_of_players_in_team}); /*old network predicts of action of policy network*/
  torch::Tensor new_predicts_p=torch::zeros({num_chunks, length, amount_of_players_in_team}); /*new network predicts of action of policy network*/
  torch::Tensor all_actions_probs = torch::zeros({num_chunks, length, amount_of_players_in_team, num_actions}); /*predictions of all actions per agent*/

  torch::Tensor old_predicts_c=torch::zeros({num_chunks, length, amount_of_players_in_team});  /*old network predicts of action of critic network*/
  torch::Tensor new_predicts_c=torch::zeros({num_chunks, length, amount_of_players_in_team});  /*old network predicts of action of critic network*/
  torch::Tensor critic_values = torch::zeros({num_chunks, length, amount_of_players_in_team});
  torch::Tensor reward_arr_minbatch = torch::zeros({num_chunks, length, 1});

  std::vector<Agents> old_net; /*Create Models class for each robot.*/
  CriticNetwork old_net_critic;
  old_net = LoadOldAgents(amount_of_players_in_team,old_net_critic);

  /*Configure arrays from min_batch to fit into later functions*/
  for (int c = 0; c < num_chunks; c++) /* For each chunk in mini batch. */
  {
    auto batch = min_batch[c];

    // For each timestep in the chunk.
    for (int t = 0; t < batch.t.size(); t++)
    {
      assert(old_predicts_p.size(0) == num_chunks);
      assert(old_predicts_p.size(1) == length);
      assert(old_predicts_p.size(2) == amount_of_players_in_team);
      assert(old_predicts_c.size(0) == num_chunks);
      assert(old_predicts_c.size(1) == length);
      assert(old_predicts_c.size(2) == amount_of_players_in_team);
      assert(new_predicts_p.size(0) == num_chunks);
      assert(new_predicts_p.size(1) == length);
      assert(new_predicts_p.size(2) == amount_of_players_in_team);
      assert(new_predicts_c.size(0) == num_chunks);
      assert(new_predicts_c.size(1) == length);
      assert(new_predicts_c.size(2) == amount_of_players_in_team);
      assert(critic_values.size(0) == num_chunks);
      assert(critic_values.size(1) == length);
      assert(critic_values.size(2) == amount_of_players_in_team);
      assert(all_actions_probs.size(0) == num_chunks);
      assert(all_actions_probs.size(1) == length);
      assert(all_actions_probs.size(2) == amount_of_players_in_team);
      assert(all_actions_probs.size(3) == num_actions);
      assert(reward_arr_minbatch.size(0) == num_chunks);
      assert(reward_arr_minbatch.size(1) == length);
      assert(reward_arr_minbatch.size(2) == 1);

      auto state = batch.t[t].state; /*Get saved state for critic*/
      auto hs_c = batch.t[t].hidden_v.ht_p; /*Get hidden state for critic*/
      auto mc_c = batch.t[t].hidden_v.ct_p; /*Get memory cell for critic*/
      auto old_ci = old_net_critic.Forward(state,hs_c,mc_c); /*Get predictions from old critic network*/

      torch::Tensor output_old_c = std::get<0>(old_ci).squeeze();
      old_predicts_c[c][t] = output_old_c.expand({amount_of_players_in_team}); /*Array needs to be same value for all agents*/
      new_predicts_c[c][t] = batch.t[t].criticvalues;
      reward_arr_minbatch[c][t] = batch.R[t]; /*Store Reward To Go*/

      /*each agent in timestep*/
      for (int agent = 0; agent < amount_of_players_in_team; agent++) {
        /* Pass each agent's state at the current timestep through the policy network*/
        auto hs_p = batch.t[t].hidden_p[agent].ht_p; /*Get saved hidden state for agent*/
        auto mc_p = batch.t[t].hidden_p[agent].ct_p; /*Get saved memory cell for agent*/

        auto agent_state = state.clone();
        agent_state.index({0, 0, 0}) = agent; /*Update the first index value to robot ID*/
        auto old_pi = old_net[agent].policy_network->Forward(agent_state, hs_p, mc_p); /*Make prediction from the agents old network*/

        /*Save new and old predicts for each agent*/
        torch::Tensor output_old_p = std::get<0>(old_pi).squeeze(); /*Output from old_net*/
        auto act = batch.t[t].actions[agent].item<int>(); /*action did in the recorded timestep*/
        old_predicts_p[c][t][agent] = output_old_p[act]; /*Save prediction of the old networks probability of agents done action in the timestep*/
        new_predicts_p[c][t][agent] = batch.t[t].actions_prob[agent][act]; /*Save stored prediction of the action*/
        all_actions_probs[c][t][agent] = batch.t[t].actions_prob[agent]; /*Store all predictions the agent did at the timestep*/
      } /*end agent loop*/

    }
  }

  /*Calculate from configured arrays*/

  /*Save to old network*/
  SaveOldModels(Models,critic);

  /*Verify the old saved networks is the same as the current networks*/
  CheckModelParametersMatch(old_net,Models,old_net_critic,critic);

  auto policy_entropy = ComputePolicyEntropy(all_actions_probs, 0.9);
  auto probability_ratios = ComputeProbabilityRatio(new_predicts_p, old_predicts_p);

  torch::Tensor probratio

  /*Policy Loss calculations*/
  for(int c = 0; c < num_mini_batch; c++)
  {
    
    auto policyloss = ComputePolicyLoss(min_batch[c].A, probratio, 0.2, policy_entropy);

    /*Critic Loss calculations*/
    auto norm_rew = NormalizeRewardToGo(reward_arr_minbatch[c]);
    auto critic_loss = -ComputeCriticLoss(new_predicts_c[c], old_predicts_c[c], norm_rew,0.2);

    /*Update the networks by the loss values*/
    UpdateNets(Models, critic, policyloss, critic_loss);
  }

  /*save updated models of all networks(Policy and Critic)*/
  SaveModels(Models,critic);

  std::cout << "Old net validation: " << std::endl;
  /*This should be false after updateNets() if networks were updated correctly*/
  CheckModelParametersMatch(old_net, Models, old_net_critic, critic);

  std::cout << "Training of buffer done! " << std::endl;
  //std::cout << "Policy loss: " << policyloss << std::endl;
  //std::cout << "Critic loss: " << critic_loss << std::endl;
  std::cout << "==============================================" << std::endl;
}


}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/