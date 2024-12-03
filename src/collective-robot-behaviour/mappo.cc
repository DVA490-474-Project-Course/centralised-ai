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
#include "../common_types.h"


namespace centralised_ai {
namespace collective_robot_behaviour {
  
/*
* Check if the old network have the same as the new network, prints out wrong parameter of network.
* Used to verify if the networks are getting updated.
* Returns true if networks match, False if networks doesnt match
*/
  bool CheckModelParametersMatch(
    const std::vector<centralised_ai::collective_robot_behaviour::Agents>& saved_models,
    const std::vector<centralised_ai::collective_robot_behaviour::Agents>& loaded_models,
    const centralised_ai::collective_robot_behaviour::CriticNetwork& saved_critic,
    const centralised_ai::collective_robot_behaviour::CriticNetwork& loaded_critic) {

    bool check = true;

    // Check agent models
    for (size_t i = 0; i < saved_models.size(); ++i) {
        auto saved_params = saved_models[i].policy_network->rnn->parameters();
        auto loaded_params = loaded_models[i].policy_network->rnn->parameters();

        if (saved_params.size() != loaded_params.size()) {
            std::cerr << "Parameter count mismatch for agent " << i << ": saved(" << saved_params.size() << "), loaded(" << loaded_params.size() << ")" << std::endl;
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
    auto saved_critic_params = saved_critic.rnn->parameters();
    auto loaded_critic_params = loaded_critic.rnn->parameters();

    if (saved_critic_params.size() != loaded_critic_params.size()) {
        std::cerr << "Parameter count mismatch in critic network: saved("
                  << saved_critic_params.size() << "), loaded("
                  << loaded_critic_params.size() << ")" << std::endl;
        check = false;
    }

    for (size_t j = 0; j < saved_critic_params.size(); ++j) {
        if (!saved_critic_params[j].equal(loaded_critic_params[j])) {
            std::cerr << "Parameter mismatch in critic network at parameter " << j << std::endl;
            check = false;
        }
    }

    // Output result
    if (check == true) {
        std::cout << "All parameters match successfully!" << std::endl;
        return true;
    } else {
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

  torch::Tensor act_prob;
  torch::Tensor action;
  RunState run_state;

  /* Gain enough batches for training */
  for (int i = 1; i <= batch_size; i++)
  {
    std::vector<Trajectory> trajectory; /*Create trajectory vector*/

    std::tie(trajectory, act_prob, action) = ResetHidden(); /* Reset/initialise hidden states for timestep 0 */
    torch::Tensor state = GetGlobalState(referee, vision_client, own_team, opponent_team); /*Get current state as vector*/
    state = GetGlobalState(referee, vision_client, own_team, opponent_team); /*because of first read gets wrong info, duplicate get state, further investigation needed*/

    /*Loop for amount of timestamps in each bach
    * Start at 1 because timestep 0 is initalised, and hidden states is following previous input (timestep - 1).
    */
    for (int timestep = 1; timestep < max_timesteps; timestep++)
    {
      /*Initialise values*/
      Trajectory exp;
      HiddenStates new_states;
      torch::Tensor prob_actions_stored = torch::zeros({amount_of_players_in_team, num_actions});

      /* Get hidden states and output probabilities for critic network, input is state and previous timestep (initialised values)*/
      auto [valNetOutput, V_hx] = critic.Forward(state,trajectory[timestep-1].hidden_v.ht_p);

      /* For each agent in one timestep, get probabilities and hidden states*/
      for (auto& agent : Models)
      {
        auto agent_state = state.clone();
        agent_state.index({0, 0, 0}) = agent.robotId; /*Update the first index value to robot ID*/

        /*Get action probabilities and hidden states, input is previous timestep hidden state for the robots index*/
        auto [act_prob, hx_new] = Models[0].policy_network->Forward(agent_state, trajectory[timestep - 1].hidden_p[agent.robotId].ht_p);
        prob_actions_stored[agent.robotId] = act_prob[0][0]; /*Store probabilities*/

        /*Save hidden states*/
        new_states.ht_p = hx_new;
        exp.hidden_p.push_back(new_states); /*Store hidden states*/

      } /*end for agent*/

      
      auto prob_actions_stored_softmax = torch::softmax(prob_actions_stored,1);

      /*Get the actions with the highest probabilities for each agent*/
      exp.actions = prob_actions_stored_softmax.argmax(1);

      /*
       *Action mask could be implemented here to define legal moves,
       *such as shooting/passing only if player have ball
      */

      /*Let agents run for 20 ms until next timestep*/
      SendActions(simulation_interfaces,exp.actions);

      /* Convert critic value from shape [1, 1] to a value. */
      auto critic_value = valNetOutput.squeeze();

      /*Update all values*/
      exp.actions_prob = prob_actions_stored_softmax;
      exp.state = state.clone();
      exp.critic_value = valNetOutput.squeeze().expand({amount_of_players_in_team});
      exp.hidden_v.ht_p = V_hx;

      /*Update state and use it for next itteration*/
      state = GetGlobalState(referee, vision_client, own_team, opponent_team);

      /* Get rewards from the actions. */
      exp.rewards = run_state.ComputeRewards(state.squeeze(0).squeeze(0),
        {-0.001, 500, 10, 0.1});

      //assert(exp.rewards.size(0) == amount_of_players_in_team);
      trajectory.push_back(exp); /*Store into trajectories*/
    } /*end for timestep*/

    /* Erase initial trajectory (First index in trajectories) */
    trajectory.erase(trajectory.begin());

    int32_t trajectory_length = trajectory.size();

    /* Calculate Reward to go */
    torch::Tensor reward_to_go = torch::zeros({amount_of_players_in_team, trajectory_length});
    torch::Tensor rewards = torch::zeros({amount_of_players_in_team, trajectory_length});
    for (int32_t a = 0; a < 2; a++)
    {
      for (int32_t t = 0; t < trajectory_length; t++)
      {
        assert(trajectory[t].rewards.size(0) == amount_of_players_in_team);
        rewards[a][t] = trajectory[t].rewards[a];
      }

      reward_to_go[a] = ComputeRewardToGo(rewards[a], 0.99);
      assert(reward_to_go.size(0) == amount_of_players_in_team);
      assert(reward_to_go.size(1) == trajectory_length);
    }

    /* Calculate temporal difference. */
    torch::Tensor critic_values = torch::zeros(trajectory_length);
    for (int32_t t = 0; t < trajectory_length; t++)
    {
      critic_values[t] = trajectory[t].critic_value[0];
    }

    torch::Tensor temporal_difference = ComputeTemporalDifference(critic_values, rewards, 0.99);
    torch::Tensor gae = ComputeGeneralAdvantageEstimation(temporal_difference, 0.99, 0.95);
    assert(temporal_difference.size(0) == amount_of_players_in_team);
    assert(temporal_difference.size(1) == trajectory_length);
    assert(gae.size(0) == amount_of_players_in_team);
    assert(gae.size(1) == trajectory_length);

    /* Split amount of timesteps in trajectories to length L */
    int32_t L = 10;
    int32_t num_chunks = trajectory_length / L;

    for (int32_t l = 0; l < num_chunks; l++) /* T/L */
    {
      int32_t start_index = l * L;
      int32_t end_index = std::min(start_index + L, trajectory_length);

      /*Add each Trajectory into dat.t value for all timesteps in chunk*/
      DataBuffer chunk;

      /* Add each time step of the slice of the trajectory to the chunk. */
      for (int i = start_index; i < end_index; i++) /* End index is exclusive. */
      {
        chunk.t.push_back(trajectory[i]);
      }

      assert(chunk.t.size() == std::min(L, trajectory_length - start_index));

      /*A and R is saved in small batch intervals*/
      chunk.A = gae.slice(1, start_index, end_index);
      chunk.R = reward_to_go.slice(1, start_index, end_index);
      assert(chunk.A.sizes() == chunk.R.sizes());
      assert(chunk.A.size(0) == amount_of_players_in_team);
      assert(chunk.A.size(1) == std::min(L, trajectory_length - start_index));
      assert(chunk.R.size(0) == amount_of_players_in_team);
      assert(chunk.R.size(1) == std::min(L, trajectory_length - start_index));

      //std::cout << "A size: " << chunk.A.sizes() << "Start index: " << start_index << " End index (Exclusion): " << end_index << std::endl;

      /*Store [t, A, R] in D (DataBuffer)*/
      //std::cout << "chunk: " << l << " Size: " << chunk.t.size() << std::endl;
      data_buffer.push_back(chunk);

    } /*end chunks loop*/

  }/*end for batch_size*/

  //std::cout << "Returning data buffer " << " Size: " << data_buffer.size() << std::endl;
  return data_buffer;
}


  /*
  * The full Mappo function for robot decision-making and training.
  * Follows the algorithm from the paper: "The Surprising Effectiveness of PPO in Cooperative
  * Multi-Agent Games" by Yu et al., available at: https://arxiv.org/pdf/2103.01955
  */
void Mappo_Update(std::vector<Agents> &Models, CriticNetwork &critic, std::vector<DataBuffer> data_buffer){

  /* Total number of chunks in D. */
  int data_buffer_size = data_buffer.size();

  /* Mini-batch. */
  std::vector<DataBuffer> mini_batch;

  std::cout << "Updating hidden states" << std::endl;

  /*Create random min batches that the agents network will update on
  * from papaer, should be set to 1
  */
  int num_mini_batch = 1;
  int mini_batch_size = batch_size / num_mini_batch;
  for (int k = 1; k <= num_mini_batch; k++)
  {
    std::vector<DataBuffer> chunks;

    /* Take a random number of chunks from the D. */
    for (int i = 0; i < mini_batch_size; i++)
    {
      int rand_index = torch::randint(0, data_buffer_size, {1}).item<int>();
      chunks.push_back(data_buffer[rand_index]);
    }

    /* Create state tensor for the batch of all sequences. */
    int num_chunks = chunks.size();
    int num_layers = 1;
    torch::Tensor input = torch::zeros({num_chunks,10,input_size}); /* [sequence_len, batch_size, input_size] */
    /* Create hidden state and cell state tensors for all agents. */
    torch::Tensor h0_critic = torch::zeros({num_chunks,1, 1, hidden_size}); /* [num_layers, batch_size, hidden_size]*/
    torch::Tensor h0_policy = torch::zeros({num_chunks,1,amount_of_players_in_team, hidden_size}); /* [num_players, num_layers, batch_size, hidden_size] */
    int num_time_steps =chunks[0].t.size();
    /* For each chunk in the mini batch, update hidden states from first hidden state. */
    for (int c = 0; c < num_chunks; c++)
    {
        auto chunk = chunks[c];

        h0_critic[c] = chunk.t[0].hidden_v.ht_p;

        for (int t = 0; t < num_time_steps; t++)
        {
          input[c][t] = chunk.t[t].state.squeeze();
        }

        for(int agent = 0; agent < amount_of_players_in_team; agent++)
        {
          h0_policy[c][0][agent]= chunk.t[0].hidden_p[agent].ht_p.squeeze();
        }
    }

    /*Policy Network update by batch*/
    for (int c = 0; c < num_chunks; c++)
    {
      auto h0_p = h0_policy[c]; /*Get first hidden state in new chunk*/
    
      for (int t = 0; t < chunks[c].t.size(); t++)
      {
        /*Get observation states for policy network*/
        auto state_action = input[c][t].view({1, 1, input_size});
        auto input_state = torch::empty({1, amount_of_players_in_team, input_size});

        for (int a = 0; a < amount_of_players_in_team; a++)
        {
          state_action[0][0][0] = a;
          input_state[0][a] = state_action[a].squeeze();
        }

        /*Update the policy networks hidden states */
        auto [pred,hx_new] = Models[0].policy_network->Forward(input_state, h0_p);
        h0_p = hx_new;
      }
    }

    /*Critic Network update by batch*/
    /*Read each chunk*/
    for (int c = 0; c < num_chunks; c++) {
      auto h0_c = h0_critic[c]; /*Get first hidden state in new chunk*/
      /*Read each timestep in chunk*/
      for (int t = 0; t < chunks[c].t.size(); t++){
        input[c][t][0] = -1;

        auto[pred,h0_new] =critic.Forward(input[c][t].view({1,1,input_size}),h0_c);
        h0_c = h0_new;
      }
    }
 
    /* Push chunk to min batch. */
    for (int c = 0; c < num_chunks; c++)
    {
      mini_batch.push_back(chunks[c]);
    }
  }

  //std::cout << "Convert mini batch to tensors" << std::endl;

  /*Create the arrays fit update functions. */
  int num_chunks = mini_batch.size();
  int64_t num_time_steps = static_cast<int64_t>(mini_batch[0].t.size()); /*Timesteps in batch*/
  torch::Tensor old_predicts_p = torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps}); /*old network predicts of action of policy network*/
  torch::Tensor new_predicts_p = torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps}); /*new network predicts of action of policy network*/
  torch::Tensor all_actions_probs = torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps, num_actions}); /*predictions of all actions per agent*/

  torch::Tensor old_predicts_c = torch::zeros({num_chunks, num_time_steps});  /*old network predicts of action of critic network*/
  torch::Tensor new_predicts_c = torch::zeros({num_chunks, num_time_steps});  /*old network predicts of action of critic network*/
  torch::Tensor reward_to_go = torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps});
  torch::Tensor gae = torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps});

  std::vector<Agents> old_net; /*Create Models class for each robot.*/
  CriticNetwork old_net_critic;
  old_net = LoadOldAgents(amount_of_players_in_team,old_net_critic);

  /* Assert sizes. */
  assert(old_predicts_p.size(0) == num_chunks);
  assert(old_predicts_p.size(1) == amount_of_players_in_team);
  assert(old_predicts_p.size(2) == num_time_steps);
  assert(old_predicts_c.size(0) == num_chunks);
  assert(old_predicts_c.size(1) == num_time_steps);
  assert(new_predicts_p.size(0) == num_chunks);
  assert(new_predicts_p.size(1) == amount_of_players_in_team);
  assert(new_predicts_p.size(2) == num_time_steps);
  assert(new_predicts_c.size(0) == num_chunks);
  assert(new_predicts_c.size(1) == num_time_steps);
  assert(all_actions_probs.size(0) == num_chunks);
  assert(all_actions_probs.size(1) == amount_of_players_in_team);
  assert(all_actions_probs.size(2) == num_time_steps);
  assert(all_actions_probs.size(3) == num_actions);
  assert(reward_to_go.size(0) == num_chunks);
  assert(reward_to_go.size(1) == amount_of_players_in_team);
  assert(reward_to_go.size(2) == num_time_steps);


  /*Configure arrays from min_batch to fit into later functions*/
  for (int c = 0; c < num_chunks; c++) /* For each chunk in mini batch. */
  {
    auto batch = mini_batch[c];

    //std::cout << "Batch " << c <<  " size: " << batch.t.size() << std::endl;

    for (int32_t j = 0; j < amount_of_players_in_team; j++)
    {
      for (int32_t t = 0; t < num_time_steps; t++)
      {
        auto state = batch.t[t].state; /*Get saved state for critic*/
        auto hs_c = batch.t[t].hidden_v.ht_p; /*Get hidden state for critic*/
        auto mc_c = batch.t[t].hidden_v.ct_p; /*Get memory cell for critic*/
        auto old_ci = old_net_critic.Forward(state,hs_c); /*Get predictions from old critic network*/

        old_predicts_c[c][t] = std::get<0>(old_ci).squeeze(); /*Array needs to be same value for all agents*/
        new_predicts_c[c][t] = batch.t[t].critic_value[0];

        //std::cout << "Reward to go size: " << reward_to_go.sizes() << "Batch R: " << batch.R.sizes() << std::endl;
        assert(reward_to_go.size(1) == batch.R.size(0));
        assert(reward_to_go.size(2) == batch.R.size(1));
        reward_to_go[c][j] = batch.R[j]; /*Store Reward To Go*/

        /* Pass each agent's state at the current timestep through the policy network*/
        auto hs_p = batch.t[t].hidden_p[j].ht_p; /*Get saved hidden state for agent*/
        auto mc_p = batch.t[t].hidden_p[j].ct_p; /*Get saved memory cell for agent*/

        auto agent_state = state.clone();
        agent_state.index({0, 0, 0}) = j; /*Update the first index value to robot ID*/
        auto old_pi = old_net[0].policy_network->Forward(agent_state, hs_p); /*Make prediction from the agents old network*/

        /*Save new and old predicts for each agent*/
        torch::Tensor output_old_p = std::get<0>(old_pi).squeeze(); /*Output from old_net*/
        output_old_p =  torch::softmax(output_old_p,0);
        auto act = batch.t[t].actions[j].item<int>(); /*action did in the recorded timestep*/
        old_predicts_p[c][j][t] = output_old_p[act]; /*Save prediction of the old networks probability of agents done action in the timestep*/
        new_predicts_p[c][j][t] = batch.t[t].actions_prob[j][act]; /*Save stored prediction of the action*/
        all_actions_probs[c][j][t] = batch.t[t].actions_prob[j]; /*Store all predictions the agent did at the timestep*/

        reward_to_go[c][j][t] = batch.R[j][t]; /*Store Reward To Go*/
        gae[c][j][t] = batch.A[j][t]; /*Store General Advantage Estimation*/
      }
    }
  }

  std::cout << "Calculate losses and update networks" << std::endl;

  /*Calculate from configured arrays*/

  /*Save to old network*/
  SaveOldModels(Models, critic);

  /*Verify the old saved networks is the same as the current networks*/
  //CheckModelParametersMatch(old_net, Models, old_net_critic, critic);

  torch::Tensor policy_entropy = ComputePolicyEntropy(all_actions_probs, 0.9);
  torch::Tensor probability_ratios = ComputeProbabilityRatio(new_predicts_p, old_predicts_p);

  torch::Tensor policy_loss = -ComputePolicyLoss(gae, probability_ratios, 0.3, policy_entropy);
  torch::Tensor critic_loss = ComputeCriticLoss(new_predicts_c, old_predicts_c, reward_to_go, 0.3);

  UpdateNets(Models, critic, policy_loss, critic_loss);

  /*save updated models of all networks(Policy and Critic)*/
  SaveModels(Models,critic);


  std::cout << "Old net validation: " << std::endl;
  /*This should be false after updateNets() if networks were updated correctly*/
  //CheckModelParametersMatch(old_net, Models, old_net_critic, critic);

  std::cout << "Training of buffer done! " << std::endl;
  std::cout << "Policy loss: " << policy_loss << std::endl;
  std::cout << "Critic loss: " << critic_loss << std::endl;
  std::cout << "==============================================" << std::endl;
}


}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/