/* network.cc
*==============================================================================
* Author: Viktor Eriksson
* Creation date: 2024-10-04.
* Last modified: 2024-10-24 by Viktor Eriksson
* Description: network functions.
* License: See LICENSE file for license details.
*==============================================================================
*/

#include "filesystem"
#include "torch/script.h"
#include "network.h"
#include "mappo.h"
#include "../common_types.h"

namespace centralised_ai {
namespace collective_robot_behaviour{

DataBuffer::DataBuffer()
    : A(torch::zeros({1, amount_of_players_in_team})),
      R(torch::zeros({1, amount_of_players_in_team})) 
{}

HiddenStates::HiddenStates()
    : ht_p(torch::zeros({1, 1, hidden_size})),  /* Hidden state tensor initialized to zeros*/
      ct_p(torch::zeros({1, 1, hidden_size}))   /*Cell state tensor initialized to zeros*/
{}

PolicyNetwork::PolicyNetwork() : 
  num_layers(1),
  output_size(num_actions),
  layer1(torch::nn::Linear(num_local_states, hidden_size)),
  layer2(torch::nn::Linear(hidden_size, hidden_size)),
  rnn(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(num_layers).batch_first(false)),
  output_layer(torch::nn::Linear(hidden_size, output_size))
{

  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("rnn", rnn);
  register_module("output_layer", output_layer);

  for (const auto& param : rnn->named_parameters()) {
    //std::cout << param.key() << std::endl;  // Print the parameter names
    if (param.key() == "weight_ih_l0") {
      torch::nn::init::orthogonal_(param.value());  // Apply orthogonal initialization
    } else if (param.key() == "weight_hh_l0") {
      torch::nn::init::orthogonal_(param.value());  // Apply orthogonal initialization
    } else if (param.key() == "bias_ih_l0") {
      torch::nn::init::zeros_(param.value());  // Initialize bias to zero
    } else if (param.key() == "bias_hh_l0") {
      torch::nn::init::zeros_(param.value());  // Initialize bias to zero
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> PolicyNetwork::Forward(torch::Tensor input, torch::Tensor hx)
{
  auto layer1_output = layer1->forward(input).tanh();  // Apply linear layer
  auto layer2_output = layer2->forward(layer1_output).tanh();  // Apply linear layer

  auto gru_output = rnn->forward(layer2_output, hx);  // GRU forward pass
  auto h = std::get<1>(gru_output);                 // Extract hidden state
  auto gru_out = std::get<0>(gru_output);         // Extract output

  auto output = output_layer(h);  // Apply linear layer

  return std::make_tuple(output, h);
}


  CriticNetwork::CriticNetwork()
      : 
        layer1(torch::nn::Linear(num_global_states, hidden_size)),
        layer2(torch::nn::Linear(hidden_size, hidden_size)),
        rnn(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1).batch_first(false)),
        output_layer(torch::nn::Linear(hidden_size, 1)) {   // Single output for value function

  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("rnn", rnn);
  register_module("output_layer", output_layer);

  for (const auto& param : rnn->named_parameters()) {
    std::cout << param.key() << std::endl;  // Print the parameter names

    if (param.key().find("weight_ih") != std::string::npos) {
      torch::nn::init::orthogonal_(param.value());  // Apply orthogonal initialization
    } else if (param.key().find("weight_hh") != std::string::npos) {
      torch::nn::init::orthogonal_(param.value());  // Apply orthogonal initialization
    } else if (param.key().find("bias_ih") != std::string::npos) {
      torch::nn::init::zeros_(param.value());  // Initialize bias to zero
    } else if (param.key().find("bias_hh") != std::string::npos) {
      torch::nn::init::zeros_(param.value());  // Initialize bias to zero
    }
  }
}

  std::tuple<torch::Tensor, torch::Tensor> CriticNetwork::Forward(
      torch::Tensor input,
      torch::Tensor hx) {

  // Initialize hidden state to zeros if not provided
  if (hx.sizes().size() == 0) {
    hx = torch::zeros({rnn->options.num_layers(), input.size(0), hidden_size});
  }
  auto layer1_output = layer1->forward(input).relu();  // Apply linear layer
  auto layer2_output = layer2->forward(layer1_output).relu();  // Apply linear layer

  auto gru_output = rnn->forward(layer2_output, hx);  // GRU forward pass
  auto h = std::get<1>(gru_output).tanh();                 // Extract hidden state
  auto lstm_pred = std::get<0>(gru_output);         // Extract output
  auto output = output_layer->forward(h);  // Linear layer to get state value

  // No tanh activation here
  return std::make_tuple(output, h);  // Return value and hidden state
}

PolicyNetwork CreatePolicy()
{
  PolicyNetwork policy;
  policy.rnn->reset_parameters();

  return policy;
}

void SaveNetworks(PolicyNetwork & policy, CriticNetwork & critic)
{
    std::string model_path = "../models/agent_network0.pt";

    try {
      torch::serialize::OutputArchive output_archive;
      policy.save(output_archive);
      output_archive.save_to(model_path);
    }
    catch (const std::exception& e) {
      std::cerr << "Error saving model for agent " <<0 << ": " << e.what() << std::endl;
    }

  /*Save Crtitc network in models folder*/
  try {
    std::string model_path = "../models/critic_network.pt";

    torch::serialize::OutputArchive output_archive;
    critic.save(output_archive);

    output_archive.save_to(model_path);
 }
  catch (const std::exception& e) {
    std::cerr << "Error saving model for critic network!" << std::endl;
  }
}

void LoadNetworks(PolicyNetwork& policy, CriticNetwork& critic)
{
  /* Load the policy network */
  try {
    std::string policy_path = "../models/agent_network0.pt";
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(policy_path);

    policy.load(input_archive);
    std::cout << "Loading policy network from " << policy_path << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error loading model policy network: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  /* Load the critic network */
  try {
    std::string critic_path = "../models/critic_network.pt";
    torch::serialize::InputArchive critic_archive;
    critic_archive.load_from(critic_path);

    critic.load(critic_archive);
    std::cout << "Loading critic network from " << critic_path << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error loading model for critic network: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void SaveOldNetworks(PolicyNetwork & policy, CriticNetwork & critic) {

    std::string model_path = "../models/old_agents/agent_network0.pt";

    try {
      torch::serialize::OutputArchive output_archive;
      policy.save(output_archive);
      output_archive.save_to(model_path);
    }
    catch (const std::exception& e) {
      std::cerr << "Error saving model for agent " << 0 << ": " << e.what() << std::endl;
    }


  /*Save Crtitc network in models folder*/
  try {
    std::string model_path = "../models/old_agents/critic_network.pt";

    torch::serialize::OutputArchive output_archive;
    critic.save(output_archive);

    output_archive.save_to(model_path);
 }
  catch (const std::exception& e) {
    std::cerr << "Error saving model for critic network!" << std::endl;
  }
}

void LoadOldNetworks(PolicyNetwork& policy, CriticNetwork& critic) {

  /* Load the policy network */
  try {
      std::string policy_path = "../models/old_agents/agent_network" + std::to_string(0) + ".pt";
      torch::serialize::InputArchive input_archive;
      input_archive.load_from(policy_path);

      policy.load(input_archive);

      std::cout << "Loading policy network from " << policy_path << std::endl;
    }
    catch (const std::exception& e) {
      std::cerr << "Error loading model for policy network: " << e.what() << std::endl;
    }

  /* Load the critic network */
  try {
    std::string critic_path = "../models/old_agents/critic_network.pt";
    torch::serialize::InputArchive critic_archive;
    critic_archive.load_from(critic_path);

    critic.load(critic_archive);
    std::cout << "Loading critic network from " << critic_path << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error loading model for critic network: " << e.what() << std::endl;
  }
}

  void UpdateNets(PolicyNetwork& policy,
    CriticNetwork& critic,
    torch::Tensor pol_loss,
    torch::Tensor cri_loss) {

  /* Set up Adam options*/
  torch::optim::AdamOptions adam_options;
  adam_options.lr(1e-4);  // Learning rate
  adam_options.eps(1e-5);  // Epsilon
  adam_options.weight_decay(0);  // Weight decay

  torch::optim::Adam opts({policy.parameters()}, adam_options);

  /*Update critic network*/
  torch::optim::Adam critnet({critic.parameters()}, adam_options);


  // Zero the gradients before the backward pass
  opts.zero_grad();
  critnet.zero_grad();

  auto loss = pol_loss + cri_loss;
  loss.backward();

  torch::nn::utils::clip_grad_norm_(policy.parameters(), 0.5);
  torch::nn::utils::clip_grad_norm_(critic.parameters(), 0.5);

  opts.step();
  critnet.step();
}

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
