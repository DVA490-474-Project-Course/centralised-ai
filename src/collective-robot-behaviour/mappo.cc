/* Mappo.cc
 * ==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-23
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: Multi Agent Proximal Policy Optimization (Mappo), training
 * algorithm connected into grSim so simulate the robocup competition. License:
 * See LICENSE file for license details.
 * ==============================================================================
 */

#include "mappo.h"
#include "../../src/common_types.h"
#include "../../src/simulation-interface/simulation_interface.h"
#include "chrono"
#include "communication.h"
#include "network.h"
#include "run_state.h"
#include "torch/torch.h"
#include "tuple"
#include "utils.h"
#include "vector"

namespace centralised_ai {
namespace collective_robot_behaviour {

/*
 * Check if the old network have the same as the new network, which is used to
 * verify if the networks are getting updated. Returns true if networks match,
 * False if networks doesnt match
 */
bool CheckModelParametersMatch(const PolicyNetwork& kSavedPolicy,
                               const PolicyNetwork& kLoadedPolicy,
                               const CriticNetwork& kSavedCritic,
                               const CriticNetwork& kLoadedCritic) {

  bool check = true;

  /* Check policy model parameters */
  std::vector<torch::Tensor> saved_policy_params =
      kSavedPolicy.rnn->parameters();
  std::vector<torch::Tensor> loaded_policy_params =
      kLoadedPolicy.rnn->parameters();

  if (saved_policy_params.size() != loaded_policy_params.size()) {
    std::cerr << "Parameter count mismatch in policy network: saved("
              << saved_policy_params.size() << "), loaded("
              << loaded_policy_params.size() << ")" << std::endl;
    check = false;
  }

  for (size_t j = 0; j < saved_policy_params.size(); ++j) {
    if (!saved_policy_params[j].equal(loaded_policy_params[j])) {
      std::cerr << "Parameter mismatch in policy network at parameter " << j
                << std::endl;
      check = false;
    }
  }

  /* Check critic model parameters */
  std::vector<torch::Tensor> saved_critic_params =
      kSavedCritic.rnn->parameters();
  std::vector<torch::Tensor> loaded_critic_params =
      kLoadedCritic.rnn->parameters();

  if (saved_critic_params.size() != loaded_critic_params.size()) {
    std::cerr << "Parameter count mismatch in critic network: saved("
              << saved_critic_params.size() << "), loaded("
              << loaded_critic_params.size() << ")" << std::endl;
    check = false;
  }

  for (size_t j = 0; j < saved_critic_params.size(); ++j) {
    if (!saved_critic_params[j].equal(loaded_critic_params[j])) {
      std::cerr << "Parameter mismatch in critic network at parameter " << j
                << std::endl;
      check = false;
    }
  }

  /* Output result */
  if (check == true) {
    std::cout << "All parameters match successfully!" << std::endl;
    return true;
  } else {
    return false;
  }
}

/* Reset the initalise state of the networks and hidden states.
 * This for timestep 0 used in MappoRun
 */
std::tuple<std::vector<Trajectory>, torch::Tensor, torch::Tensor>
ResetHidden() {
  std::vector<Trajectory> trajectories;
  HiddenStates reset_states;
  reset_states.ct_p = torch::zeros({1, 1, hidden_size});
  reset_states.ht_p = torch::zeros({1, 1, hidden_size});

  Trajectory initial_trajectory;
  for (int i = 0; i < amount_of_players_in_team; i++) {
    initial_trajectory.hidden_states_policy.push_back(reset_states);
  }

  trajectories.push_back(initial_trajectory);

  torch::Tensor action_probabilities = torch::zeros({num_actions});
  torch::Tensor action;

  return std::make_tuple(trajectories, action_probabilities, action);
};

/*
 * Where the agents run and training-data getting received.
 */
std::vector<DataBuffer>
MappoRun(PolicyNetwork& policy, CriticNetwork& critic,
         ssl_interface::AutomatedReferee& referee,
         ssl_interface::VisionClient& vision_client, Team own_team,
         std::vector<simulation_interface::SimulationInterface>
             simulation_interfaces) {

  torch::AutoGradMode enable_grad_mode(false);

  /* Initialise data buffer D */
  std::vector<DataBuffer> data_buffer;

  /* Get opponent team class */
  Team opponent_team = ComputeOpponentTeam(own_team);
  torch::Tensor action_probabilities;
  torch::Tensor action;
  RunState run_state;

  /* Create trajectory vector */
  std::vector<Trajectory> trajectory;

  /* Initialise values */
  Trajectory exp;
  HiddenStates new_states;

  /* Add each Trajectory into dat.t value for all timesteps in chunk */
  DataBuffer chunk;

  /* Gain enough batches for training */
  for (int i = 1; i <= batch_size; i++) {
    /* Clears the trajectory for each new iteration. */
    trajectory.clear();

    std::tie(trajectory, action_probabilities, action) =
        ResetHidden(); /* Reset/initialise hidden states for timestep 0 */
    torch::Tensor state =
        GetGlobalState(referee, vision_client, own_team,
                       opponent_team); /* Get current state as vector */
    state = GetGlobalState(
        referee, vision_client, own_team,
        opponent_team); /* Duplicate get state to avoid wrong initial info */

    /* Loop for amount of timestamps in each batch */
    for (int timestep = 1; timestep < max_timesteps; timestep++) {
      exp.hidden_states_policy.clear();

      torch::Tensor prob_actions_stored =
          torch::zeros({amount_of_players_in_team, num_actions});

      /* Get hidden states and output probabilities for critic network, input is
       * state and previous timestep */
      std::tuple<torch::Tensor, torch::Tensor> critic_value = critic.Forward(
          state, trajectory[timestep - 1].hidden_states_critic.ht_p);

      torch::Tensor critic_output = std::get<0>(critic_value);
      torch::Tensor critic_hx = std::get<1>(critic_value);

      /* For each agent in one timestep, get probabilities and hidden states */
      for (int agent = 0; agent < amount_of_players_in_team; agent++) {
        torch::Tensor agent_state = state.clone();

        /* Update the first index value to robot ID */
        agent_state.index({0, 0, 0}) = agent;

        torch::Tensor local_state =
            GetLocalState(vision_client, own_team, agent);

        /* Get action probabilities and hidden states */
        std::tuple<torch::Tensor, torch::Tensor> policy_value = policy.Forward(
            local_state,
            trajectory[timestep - 1].hidden_states_policy[agent].ht_p);

        prob_actions_stored[agent] = std::get<0>(policy_value)[0][0];
        new_states.ht_p = std::get<1>(policy_value);

        assert(action_probabilities.requires_grad() == 0);

        /* Store hidden states */
        exp.hidden_states_policy.push_back(new_states);
      }

      /* Get the actions with the highest probabilities for each agent */
      torch::Tensor prob_actions_stored_softmax =
          torch::softmax(prob_actions_stored, 1);
      exp.actions = prob_actions_stored_softmax.argmax(1);

      /* Send actions to the simulation.
       * Note that maybe have a delay between sending actions and receiving the
       * new state will let the policy learn much better due to actually see a
       * difference in the environment from the taken actions.
       */
      SendActions(simulation_interfaces, exp.actions);

      /* Update all values */
      exp.actions_prob = prob_actions_stored_softmax;
      exp.state = state.clone();
      exp.critic_value =
          critic_output.squeeze().expand({amount_of_players_in_team});
      exp.hidden_states_critic.ht_p = critic_hx;

      /* Update state and use it for next iteration */
      state = GetGlobalState(referee, vision_client, own_team, opponent_team);

      /* Get rewards from the actions */
      exp.rewards = run_state.ComputeRewards(state.squeeze(0).squeeze(0),
                                             {-0.001, 500, 10, 0.001});
      assert(exp.rewards.size(0) == amount_of_players_in_team);

      /* Store the experience into the trajectory */
      trajectory.push_back(exp);

    } /* end for timestep */

    /* Erase initial trajectory (First index in trajectories) */
    trajectory.erase(trajectory.begin());

    int32_t trajectory_length = trajectory.size();

    /* Calculate Reward to go */
    torch::Tensor reward_to_go =
        torch::zeros({amount_of_players_in_team, trajectory_length});
    torch::Tensor rewards =
        torch::zeros({amount_of_players_in_team, trajectory_length});

    for (int32_t a = 0; a < 2; a++) {
      for (int32_t t = 0; t < trajectory_length; t++) {
        rewards[a][t] = trajectory[t].rewards[a];
      }

      reward_to_go[a] = ComputeRewardToGo(rewards[a], 0.99);
    }

    /* Calculate temporal difference */
    torch::Tensor critic_values = torch::zeros(trajectory_length);
    for (int32_t t = 0; t < trajectory_length; t++) {
      critic_values[t] = trajectory[t].critic_value[0];
    }

    torch::Tensor temporal_difference =
        ComputeTemporalDifference(critic_values, rewards, 0.99);
    torch::Tensor gae =
        ComputeGeneralAdvantageEstimation(temporal_difference, 0.99, 0.95);

    /* Split amount of timesteps in trajectories to length L */
    int32_t l_batch_size = 10;
    int32_t num_chunks = trajectory_length / l_batch_size;

    for (int32_t l = 0; l < num_chunks; l++) {
      int32_t start_index = l * l_batch_size;
      int32_t end_index =
          std::min(start_index + l_batch_size, trajectory_length);

      /* Clear the trajectory data of the chunk. */
      chunk.t.clear();

      /* Add each time step of the slice of the trajectory to the chunk */
      for (int k = start_index; k < end_index; k++) {
        chunk.t.push_back(trajectory[k]);
      }

      chunk.A = gae.slice(1, start_index, end_index);
      chunk.R = reward_to_go.slice(1, start_index, end_index);

      /* Store [t, A, R] in D (DataBuffer) */
      data_buffer.push_back(chunk);
    }
  }

  return data_buffer;
}

/*
 * The full Mappo function for robot decision-making and training.
 * Follows the algorithm from the paper: "The Surprising Effectiveness of PPO in
 * Cooperative Multi-Agent Games" by Yu et al., available at:
 * https://arxiv.org/pdf/2103.01955
 */
void MappoUpdate(PolicyNetwork& policy, CriticNetwork& critic,
                 std::vector<DataBuffer> data_buffer) {
  /* Total number of chunks in D. */
  int data_buffer_size = data_buffer.size();

  /* Mini-batch. */
  std::vector<DataBuffer> mini_batch;

  std::cout << "Updating hidden states" << std::endl;
  policy.train();
  critic.train();
  torch::AutoGradMode enable_grad_mode(true);
  torch::autograd::DetectAnomalyGuard(true);

  std::vector<DataBuffer> chunks;

  /* Create random min batches that the agents network will update on
   * from papaer, should be set to 1
   */
  int num_mini_batch = 1;
  int mini_batch_size = batch_size / num_mini_batch;
  for (int k = 1; k <= num_mini_batch; k++) {

    /* Clears the chunks for each mini batch. */
    chunks.clear();

    /* Take a random number of chunks from the D. */
    for (int i = 0; i < mini_batch_size; i++) {
      int rand_index = torch::randint(0, data_buffer_size, {1}).item<int>();
      chunks.push_back(data_buffer[rand_index]);
    }

    /* Create state tensor for the batch of all sequences. */
    int num_chunks = chunks.size();
    int num_time_steps = chunks[0].t.size();
    int num_layers = 1;
    torch::Tensor input = torch::zeros(
        {num_chunks, 10,
         num_global_states}); /* [batch_size, sequence_len, input_size] */
    /* Create hidden state and cell state tensors for all agents. */
    torch::Tensor h0_critic =
        torch::zeros({num_chunks, 1, 1,
                      hidden_size}); /* [num_layers, batch_size, hidden_size]*/
    torch::Tensor h0_policy = torch::zeros(
        {num_chunks, 1, amount_of_players_in_team,
         hidden_size}); /* [num_players, num_layers, batch_size, hidden_size] */

    /* For each chunk in the mini batch, update hidden states from first hidden
     * state.
     */
    for (int c = 0; c < num_chunks; c++) {
      DataBuffer chunk = chunks[c];

      h0_critic[c] = chunk.t[0].hidden_states_critic.ht_p.clone();

      for (int t = 0; t < num_time_steps; t++) {
        input[c][t] = chunk.t[t].state.squeeze().clone();
      }

      for (int agent = 0; agent < amount_of_players_in_team; agent++) {
        h0_policy[c][0][agent] =
            chunk.t[0].hidden_states_policy[agent].ht_p.squeeze().clone();
      }
    }

    /* Push chunk to min batch. */
    for (int c = 0; c < num_chunks; c++) {
      mini_batch.push_back(chunks[c]);
    }
  }

  /* Create the arrays fit update functions. */
  int num_chunks = mini_batch.size();
  int64_t num_time_steps =
      static_cast<int64_t>(mini_batch[0].t.size()); /* Timesteps in batch */
  torch::Tensor old_policy_probabilities = torch::zeros(
      {num_chunks, amount_of_players_in_team,
       num_time_steps}); /* old network predicts of action of policy network */
  torch::Tensor new_policy_probabilities = torch::zeros(
      {num_chunks, amount_of_players_in_team,
       num_time_steps}); /* new network predicts of action of policy network */
  torch::Tensor all_actions_probs =
      torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps,
                    num_actions}); /* predictions of all actions per agent */

  torch::Tensor old_predicts_c = torch::zeros(
      {num_chunks,
       num_time_steps}); /* old network predicts of action of critic network */
  torch::Tensor new_predicts_c = torch::zeros(
      {num_chunks,
       num_time_steps}); /* old network predicts of action of critic network */
  torch::Tensor reward_to_go =
      torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps});
  torch::Tensor gae =
      torch::zeros({num_chunks, amount_of_players_in_team, num_time_steps});

  /* Loads Models class for all robots */
  PolicyNetwork old_net_policy;
  CriticNetwork old_net_critic;
  LoadOldNetworks(old_net_policy, old_net_critic);

  /* Assert sizes */
  assert(reward_to_go.size(0) == num_chunks);
  assert(reward_to_go.size(1) == amount_of_players_in_team);
  assert(reward_to_go.size(2) == num_time_steps);

  torch::Tensor local_state = torch::zeros({1, 1, num_local_states});
  assert(local_state.size(0) == 1);
  assert(local_state.size(1) == 1);
  assert(local_state.size(2) == num_local_states);

  /* Configure arrays from min_batch to fit into later functions */
  for (int c = 0; c < num_chunks; c++) {
    DataBuffer batch = mini_batch[c];
    torch::Tensor h0_c = batch.t[0].hidden_states_critic.ht_p.clone();

    for (int32_t j = 0; j < amount_of_players_in_team; j++) {
      torch::Tensor h0_p = batch.t[0].hidden_states_policy[j].ht_p.clone();
      torch::Tensor h0_p_old = h0_p.clone();

      for (int32_t t = 0; t < num_time_steps; t++) {
        /* Update critic network from batch */
        /* Old network */
        torch::Tensor state = batch.t[t].state.clone(); /* Get saved state for critic */
        torch::Tensor global_state = state.clone();
        global_state[0][0][0] = -1;

        /* Old network */
        std::tuple<torch::Tensor, torch::Tensor> old_ci =
            old_net_critic.Forward(
                global_state,
                h0_c); /* Get predictions from old critic network */

        old_predicts_c[c][t] =
            std::get<0>(old_ci)
                .squeeze(); /* Array needs to be same value for all agents */

        /* new(current) network */

        std::tuple<torch::Tensor, torch::Tensor> critic_new_value =
              critic.Forward(global_state, h0_c);

        new_predicts_c[c][t] = std::get<0>(critic_new_value).squeeze();
        h0_c = std::get<1>(critic_new_value);

        /* Update Policy network from batch */
        int act = batch.t[t]
                      .actions[j]
                      .item<int>(); /* action did in the recorded timestep */

        /* Create the Local state from the current Global state*/
        local_state[0][0][0] = state[0][0][3 + 2 * j + 0];
        local_state[0][0][1] = state[0][0][3 + 2 * j + 1];
        local_state[0][0][2] = state[0][0][15 + j];
        local_state[0][0][3] = state[0][0][1];
        local_state[0][0][4] = state[0][0][2];

        std::cout << local_state[0][0] << std::endl;
        std::cout << state[0][0] << std::endl;

        /* Make prediction from the agents old network */
        std::tuple<torch::Tensor, torch::Tensor> old_pi =
            old_net_policy.Forward(local_state.clone(), h0_p_old);

        /* Output from old_net */
        h0_p_old = std::get<1>(old_pi);
        torch::Tensor output_old_p = std::get<0>(old_pi).squeeze();
        output_old_p = torch::softmax(output_old_p, -1);


        /* Save prediction of the old networks probability of agents done action
         * in the timestep
         */
        old_policy_probabilities[c][j][t] = output_old_p[act];

        std::tuple<torch::Tensor, torch::Tensor> policy_new_value =
            policy.Forward(local_state.clone(), h0_p);

        h0_p = std::get<1>(policy_new_value);
        torch::Tensor pred_p = std::get<0>(policy_new_value);
        pred_p = torch::softmax(pred_p, -1);

        /* Check if pred_p contains zeros */
        if (pred_p.eq(0).any().item<bool>()) {
          std::cerr << "Error: pred_p contains zero values after softmax!"
                    << std::endl;

          pred_p = torch::clamp(pred_p, 1e-10, 1.0);
          std::cerr << "Corrected pred_p: " << pred_p << std::endl;
        }

        /* Save stored prediction of the action */
        new_policy_probabilities[c][j][t] = pred_p.squeeze()[act];

        /* Store all predictions the agent did at the timestep */
        all_actions_probs[c][j][t] = pred_p.squeeze();

        /* Store Reward To Go */
        assert(reward_to_go.size(1) == batch.R.size(0));
        assert(reward_to_go.size(2) == batch.R.size(1));
        reward_to_go[c][j] = batch.R[j];
        reward_to_go[c][j][t] = batch.R[j][t];

        /* Store General Advantage Estimation */
        gae[c][j][t] = batch.A[j][t];
      }
    }
  }

  std::cout << "Calculate losses and update networks" << std::endl;

  /* Save to old network */
  SaveOldNetworks(policy, critic);

  /* Verify the old saved networks is the same as the current networks
  bool matches =
      CheckModelParametersMatch(old_net_policy, policy, old_net_critic, critic);
  */

  torch::Tensor gae_tensor = gae;
  torch::Tensor reward_to_go_tensor = reward_to_go;

  assert(all_actions_probs.requires_grad() == true);
  assert(new_policy_probabilities.requires_grad() == true);
  assert(old_policy_probabilities.requires_grad() == true);
  assert(old_predicts_c.requires_grad() == true);

  /* Compute policy entropy */
  torch::Tensor policy_entropy =
      ComputePolicyEntropy(all_actions_probs, entropy_coefficient);

  /* Compute probability ratios */
  torch::Tensor probability_ratios = ComputeProbabilityRatio(
      new_policy_probabilities, old_policy_probabilities);

  /* Compute policy loss */
  torch::Tensor policy_loss = -ComputePolicyLoss(gae_tensor, probability_ratios,
                                                 clip_value, policy_entropy);

  /* Compute critic loss */
  torch::Tensor critic_loss = ComputeCriticLoss(
      new_predicts_c, old_predicts_c, reward_to_go_tensor, clip_value);

  assert(policy_loss.requires_grad() && "policy_loss must require gradients");
  assert(!policy_loss.isnan().any().item<bool>() &&
         "critic_loss contains NaNs");

  assert(critic_loss.requires_grad() && "critic_loss must require gradients");
  assert(!critic_loss.isnan().any().item<bool>() &&
         "critic_loss contains NaNs");

  /* Update the networks */
  UpdateNets(policy, critic, policy_loss, critic_loss);

  /* save updated networks to a file */
  SaveNetworks(policy, critic);

  /* This should be false after updateNets() if networks were updated correctly
   */
  /*matches =
      CheckModelParametersMatch(old_net_policy, policy, old_net_critic, critic);
  */

  std::cout << "Training of buffer done! " << std::endl;
  std::cout << "Policy loss: " << policy_loss << std::endl;
  std::cout << "Critic loss: " << critic_loss << std::endl;
  std::cout << "==============================================" << std::endl;
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */