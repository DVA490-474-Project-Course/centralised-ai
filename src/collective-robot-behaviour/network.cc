/* network.cc
 *==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-04.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: network functions.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#include "network.h"
#include "../../src/common_types.h"
#include "communication.h"
#include "filesystem"
#include "mappo.h"
#include "torch/script.h"
#include "torch/torch.h"

namespace centralised_ai 
{
namespace collective_robot_behaviour
{

DataBuffer::DataBuffer()
    : A(torch::zeros({1, amount_of_players_in_team})),
      R(torch::zeros({1, amount_of_players_in_team})) {}

Trajectory::Trajectory()
    : state(torch::zeros({1, 1, num_global_states})),
      actions_prob(torch::zeros({num_actions})),
      rewards(torch::zeros(amount_of_players_in_team)),
      actions(torch::zeros({amount_of_players_in_team})),
      new_state(torch::zeros({1, 1, num_global_states})) {}

HiddenStates::HiddenStates()
    : ht_p(torch::zeros(
          {1, 1, hidden_size})), /* Hidden state tensor initialized to zeros */
      ct_p(torch::zeros(
          {1, 1, hidden_size})) /* Cell state tensor initialized to zeros */
{}

PolicyNetwork::PolicyNetwork()
    : kNumLayers(1), kOutputSize(num_actions),
      layer1(torch::nn::Linear(num_local_states, hidden_size)),
      layer2(torch::nn::Linear(hidden_size, hidden_size)),
      rnn(torch::nn::GRUOptions(hidden_size, hidden_size)
              .num_layers(kNumLayers)
              .batch_first(false)),
      output_layer(torch::nn::Linear(hidden_size, kOutputSize)) {

  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("rnn", rnn);
  register_module("output_layer", output_layer);

  for (const torch::OrderedDict<std::string, at::Tensor>::Item& kParam : rnn->named_parameters()) {
    if (kParam.key() == "weight_ih_l0") {
      torch::nn::init::orthogonal_(
          kParam.value());
    } else if (kParam.key() == "weight_hh_l0") {
      torch::nn::init::orthogonal_(
          kParam.value());
    } else if (kParam.key() == "bias_ih_l0") {
      torch::nn::init::zeros_(kParam.value());
    } else if (kParam.key() == "bias_hh_l0") {
      torch::nn::init::zeros_(kParam.value());
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
PolicyNetwork::Forward(torch::Tensor input, torch::Tensor hx) {
  torch::Tensor layer1_output = layer1->forward(input).tanh();
  torch::Tensor layer2_output =
      layer2->forward(layer1_output).tanh();

  std::tuple<torch::Tensor, torch::Tensor > gru_output = rnn->forward(layer2_output, hx);
  torch::Tensor gru_hidden_states = std::get<1>(gru_output);
  torch::Tensor gru_out = std::get<0>(gru_output);

  torch::Tensor output = output_layer(gru_hidden_states);

  return std::make_tuple(output, gru_hidden_states);
}

CriticNetwork::CriticNetwork()
    : layer1(torch::nn::Linear(num_global_states, hidden_size)),
      layer2(torch::nn::Linear(hidden_size, hidden_size)),
      rnn(torch::nn::GRUOptions(hidden_size, hidden_size)
              .num_layers(1)
              .batch_first(false)),
      output_layer(torch::nn::Linear(hidden_size,
                                     1)) {

  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("rnn", rnn);
  register_module("output_layer", output_layer);

  for (const torch::OrderedDict<std::string, at::Tensor>::Item& kParam : rnn->named_parameters()) {
    std::cout << kParam.key() << std::endl;

    if (kParam.key().find("weight_ih") != std::string::npos) {
      torch::nn::init::orthogonal_(
          kParam.value());
    } else if (kParam.key().find("weight_hh") != std::string::npos) {
      torch::nn::init::orthogonal_(
          kParam.value());
    } else if (kParam.key().find("bias_ih") != std::string::npos) {
      torch::nn::init::zeros_(kParam.value());
    } else if (kParam.key().find("bias_hh") != std::string::npos) {
      torch::nn::init::zeros_(kParam.value());
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
CriticNetwork::Forward(torch::Tensor input, torch::Tensor hx) {

  /* Initialize hidden state to zeros if not provided */
  if (hx.sizes().size() == 0) {
    hx = torch::zeros({rnn->options.num_layers(), input.size(0), hidden_size});
  }
  torch::Tensor layer1_output = layer1->forward(input).relu();
  torch::Tensor layer2_output =
      layer2->forward(layer1_output).relu();

  std::tuple<torch::Tensor, torch::Tensor> gru_output = rnn->forward(layer2_output, hx);
  torch::Tensor gru_hidden_states = std::get<1>(gru_output).tanh();
  torch::Tensor gru_out = std::get<0>(gru_output);
  torch::Tensor output = output_layer->forward(gru_hidden_states);

  return std::make_tuple(output, gru_hidden_states);
}

PolicyNetwork CreatePolicy() {
  PolicyNetwork policy;
  policy.rnn->reset_parameters();

  return policy;
}

void SaveNetworks(PolicyNetwork& policy, CriticNetwork& critic) {
  
  try {
    std::string model_path = "../models/agent_network0.pt";
    torch::serialize::OutputArchive output_archive;
    
    policy.save(output_archive);
    output_archive.save_to(model_path);
  } catch (const std::exception& kException) {
    std::cerr << "Error saving model for agent " << 0 << ": " << kException.what()
              << std::endl;
  }

  try {
    std::string model_path = "../models/critic_network.pt";
    torch::serialize::OutputArchive output_archive;

    critic.save(output_archive);
    output_archive.save_to(model_path);
  } catch (const std::exception& kException) {
    std::cerr << "Error saving model for critic network!" << std::endl;
  }
}

void LoadNetworks(PolicyNetwork& policy, CriticNetwork& critic) {
  
  /* Load the policy network */
  try {
    std::string policy_path = "../models/agent_network0.pt";
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(policy_path);

    policy.load(input_archive);
    std::cout << "Loading policy network from " << policy_path << std::endl;
  } catch (const std::exception& kException) {
    std::cerr << "Error loading model policy network: " << kException.what()
              << std::endl;

    throw std::runtime_error("Error loading model policy network");
  }

  /* Load the critic network */
  try {
    std::string critic_path = "../models/critic_network.pt";
    torch::serialize::InputArchive critic_archive;
    critic_archive.load_from(critic_path);

    critic.load(critic_archive);
    std::cout << "Loading critic network from " << critic_path << std::endl;
  } catch (const std::exception& kException) {
    std::cerr << "Error loading model for critic network: " << kException.what()
              << std::endl;
    
    throw std::runtime_error("Error loading model critic network");
  }
}

void SaveOldNetworks(PolicyNetwork& policy, CriticNetwork& critic) {

  try {
    std::string model_path = "../models/old_agents/agent_network0.pt";
    torch::serialize::OutputArchive output_archive;

    policy.save(output_archive);
    output_archive.save_to(model_path);
  } catch (const std::exception& kException) {
    std::cerr << "Error saving model for agent " << 0 << ": " << kException.what()
              << std::endl;
  }

  try {
    std::string model_path = "../models/old_agents/critic_network.pt";
    torch::serialize::OutputArchive output_archive;

    critic.save(output_archive);
    output_archive.save_to(model_path);
  } catch (const std::exception& kException) {
    std::cerr << "Error saving model for critic network!" << std::endl;
  }
}

void LoadOldNetworks(PolicyNetwork& policy, CriticNetwork& critic) {

  /* Load the policy network */
  try {
    std::string policy_path =
        "../models/old_agents/agent_network" + std::to_string(0) + ".pt";
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(policy_path);

    policy.load(input_archive);

    std::cout << "Loading policy network from " << policy_path << std::endl;
  } catch (const std::exception& kException) {
    std::cerr << "Error loading model for policy network: " << kException.what()
              << std::endl;
  }

  /* Load the critic network */
  try {
    std::string critic_path = "../models/old_agents/critic_network.pt";
    torch::serialize::InputArchive critic_archive;
    critic_archive.load_from(critic_path);

    critic.load(critic_archive);
    std::cout << "Loading critic network from " << critic_path << std::endl;
  } catch (const std::exception& kException) {
    std::cerr << "Error loading model for critic network: " << kException.what()
              << std::endl;
  }
}

void UpdateNets(PolicyNetwork& policy, CriticNetwork& critic,
                torch::Tensor pol_loss, torch::Tensor cri_loss) {

  /* Set up Adam options*/
  torch::optim::AdamOptions adam_options;
  adam_options.lr(1e-4);
  adam_options.eps(1e-5);
  adam_options.weight_decay(0);

  torch::optim::Adam opts({policy.parameters()}, adam_options);

  /* Update critic network */
  torch::optim::Adam critnet({critic.parameters()}, adam_options);

  /* Zero the gradients before the backward pass */
  opts.zero_grad();
  critnet.zero_grad();

  torch::Tensor loss = pol_loss + cri_loss;
  loss.backward();

  torch::nn::utils::clip_grad_norm_(policy.parameters(), 0.2);
  torch::nn::utils::clip_grad_norm_(critic.parameters(), 0.2);

  opts.step();
  critnet.step();
}

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */