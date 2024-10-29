/* network.cc
*==============================================================================
* Author: Viktor Eriksson
* Creation date: 2024-10-04.
* Last modified: 2024-10-24 by Viktor Eriksson
* Description: network functions.
* License: See LICENSE file for license details.
*==============================================================================
*/

#include <filesystem>
#include <torch/script.h>
#include "network.h"
#include "mappo.h"
/*!
* Configuration values of the networks
*/
extern int input_size; // Number of input features
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;
namespace centralised_ai {
namespace collective_robot_behaviour
{
DataBuffer::DataBuffer()
    : A(torch::zeros({1, hidden_size})),
      R(torch::zeros({1, hidden_size})) {
}

Agents::Agents(int id, PolicyNetwork network)
    : robotId(id), policy_network(std::move(network)),
      random_floats(torch::rand({2})), // Initialize random_floats here
      x_pos(random_floats[0].item<float>()), // Initialize x_pos
      y_pos(random_floats[1].item<float>()){

}


HiddenStates::HiddenStates()
    : ht_p(torch::zeros({1, 1, hidden_size})),  /* Hidden state tensor initialized to zeros*/
      ct_p(torch::zeros({1, 1, hidden_size}))   /*Cell state tensor initialized to zeros*/
{}

PolicyNetwork::PolicyNetwork()
  : num_layers(1),
  output_size(num_actions),
  lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
  output_layer(torch::nn::Linear(hidden_size, output_size)) {

  register_module("lstm", lstm);
  register_module("output_layer", output_layer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PolicyNetwork::Forward(
  torch::Tensor input,
  torch::Tensor hx,
  torch::Tensor cx)
  {
  auto hidden_states = std::make_tuple(hx, cx);
  auto lstm_output = lstm->forward(input, hidden_states);
  auto val = std::get<0>(lstm_output);
  auto hx_new = std::get<0>(std::get<1>(lstm_output));
  auto cx_new = std::get<1>(std::get<1>(lstm_output));

  auto output = val.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
  auto value = output_layer(output);

  return std::make_tuple(value, hx_new, cx_new);
}




CriticNetwork::CriticNetwork():
  num_layers(1),
  output_size(1),
  lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
  value_layer(torch::nn::Linear(hidden_size, output_size)) {

  register_module("lstm", lstm);
  register_module("value_layer", value_layer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CriticNetwork::Forward(
  torch::Tensor input,
  torch::Tensor hx,
  torch::Tensor cx) {

  auto hidden_states = std::make_tuple(hx, cx);
  auto lstm_output = lstm->forward(input, hidden_states);
  auto val = std::get<0>(lstm_output);
  auto hx_new = std::get<0>(std::get<1>(lstm_output));
  auto cx_new = std::get<1>(std::get<1>(lstm_output));

  val = val.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
  auto value = value_layer(val);

  return std::make_tuple(value, hx_new, cx_new);
}


std::vector<Agents> CreateAgents(int amount_of_players_in_team) {
  std::vector<Agents> robots;
  for (int i = 0; i < amount_of_players_in_team; i++) {
    PolicyNetwork model;;
    robots.emplace_back(i, model);
  }
  return robots;
}

void SaveModels(const std::vector<Agents>& models, CriticNetwork& critic) {
  for (const auto& agent : models) {
    std::string model_path = "../models/agent_network" + std::to_string(agent.robotId) + ".pt";

    try {
      torch::serialize::OutputArchive output_archive;
      agent.policy_network.save(output_archive);
      output_archive.save_to(model_path);
    }
    catch (const std::exception& e) {
      std::cerr << "Error saving model for agent " << agent.robotId << ": " << e.what() << std::endl;
    }
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

std::vector<Agents> LoadAgents(int player_count, CriticNetwork& critic) {
  std::vector<Agents> agents;
  PolicyNetwork model;;

  /*Load all policy networks and assign to each agent*/
  for (player_count--; player_count >= 0; player_count--) {
    try {
      std::string model_path = "../models/agent_network" + std::to_string(player_count) + ".pt";
      torch::serialize::InputArchive input_archive;

      input_archive.load_from(model_path);


      model.load(input_archive);
      std::cout << "Loading agent " << player_count << " from " << model_path << std::endl;
      agents.emplace_back(player_count, model);

    }
    catch (const std::exception& e) {
      std::cerr << "Error loading model for agent " << player_count << ": " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  /*Load in crtitic network*/
  try {
    std::string model_path = "../models/critic_network.pt";
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(model_path);

    critic.load(input_archive);
    std::cout << "Loading critic network from " << model_path << std::endl;
  }

  catch (const std::exception& e) {
    std::cerr << "Error loading model for critic network" << std::endl;
    exit(EXIT_FAILURE);
  }

  return agents;
}

void UpdateNets(std::vector<Agents>& agents, CriticNetwork& critic, std::vector<DataBuffer> exper_buff) {

  torch::Tensor targets = torch::tensor({{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, torch::dtype(torch::kFloat));
  torch::Tensor loss = torch::tensor(0.004, torch::requires_grad(true));  // This creates a tensor with gradient tracking enabled

  //update each agent
  for (auto& agent : agents)
  {
    agent.policy_network.train();
    torch::optim::Adam optimizer(agent.policy_network.parameters(), torch::optim::AdamOptions(0.99).eps(1e-5));

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    std::cout << "Loss: " << loss.item<float>() << std::endl;
  }
}
}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
