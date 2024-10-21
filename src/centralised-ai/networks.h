//
// Created by Viktor Eriksson on 2024-10-04.
//
#include <filesystem>
#include <torch/script.h>
#ifndef NETWORKS_H
#define NETWORKS_H
#include "Communication.h"
int input_size = 7; // Number of input features
int num_actions = 3;
int amount_of_players_in_team = 6;
int hidden_size = 5;

struct Policynetwork : torch::nn::Module {
 // Number of features in the hidden state
    const int num_layers = 1; // Number of LSTM layers on top of eachother
    const int output_size = num_actions; // Number of output classes

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_layer{nullptr};

    Policynetwork()
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          output_layer(register_module("output_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
         torch::Tensor input,
         torch::Tensor hx,
         torch::Tensor cx) {

        auto hidden_states = std::make_tuple(hx, cx);
        auto lstm_output = lstm->forward(input, hidden_states);
        auto val = std::get<0>(lstm_output);
        auto hx_new = std::get<0>(std::get<1>(lstm_output)); // Hidden state (hx)
        auto cx_new = std::get<1>(std::get<1>(lstm_output)); // Cell state (cx)

        auto output = val.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // Last time step output
        auto value = output_layer(output);

        // Return value, hx_new, and cx_new as a flat tuple (no nesting)
        return std::make_tuple(value, hx_new, cx_new);
    }

};

struct CriticNetwork : torch::nn::Module {
     // Number of input features

    const int num_layers = 1; // Number of LSTM layers
    const int output_size = 1; // Single value output for critic

    CriticNetwork()
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(false)),
          value_layer(register_module("value_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
         torch::Tensor input,
         torch::Tensor hx,
         torch::Tensor cx) {

        auto hidden_states = std::make_tuple(hx, cx);
        auto lstm_output = lstm->forward(input, hidden_states);
        auto val = std::get<0>(lstm_output);
        auto hx_new = std::get<0>(std::get<1>(lstm_output)); // Hidden state (hx)
        auto cx_new = std::get<1>(std::get<1>(lstm_output)); // Cell state (cx)
        //WRONG DIMENSIONS HERE; DONT KNOW WHY THO! POLICY COULD HAVE SAME PROBLEM!
        auto value = value_layer(val); // Final output (value estimation)

        // Return value, hx_new, and cx_new as a flat tuple (no nesting)
        return std::make_tuple(value, hx_new, cx_new);
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear value_layer{nullptr};
};

struct Agents { // Generate two random floats between 0 and 1
    torch::Tensor random_floats = torch::rand({2});

    // Access and print the random floats
    int robotId;
    float x_pos = random_floats[0].item<float>();
    float y_pos= random_floats[1].item<float>();
    Policynetwork policyNetwork;

    // Constructor
    Agents(int id, Policynetwork network)
        : robotId(id), policyNetwork(std::move(network)) {
    }


};

// Function to create agents
std::vector<Agents> createAgents(int amount_of_players_in_team) {
    std::vector<Agents> robots; // Initialize the vector here
    for (int i = 0; i < amount_of_players_in_team; i++) {
        Policynetwork model; // Create a policy network for each agent
        robots.emplace_back(i, model); // Add new agent to the vector
    }
    return robots; // Return the vector of agents
}

void save_models(const std::vector<Agents>& models, CriticNetwork& critic) {
    for (const auto& agent : models) {
        std::string model_path = "../models/agent_network" + std::to_string(agent.robotId) + ".pt";

        try {
            torch::serialize::OutputArchive output_archive;
            // Save the full model (policy network) to the archive
            agent.policyNetwork.save(output_archive);

            // Save the serialized model to file
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
        // Save the full model (policy network) to the archive
        critic.save(output_archive);

        // Save the serialized model to file
        output_archive.save_to(model_path);
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving model for critic network!" << std::endl;
    }
}

void print_parameters(const Agents& agent) {
    std::vector<torch::Tensor> params = agent.policyNetwork.parameters();

    std::cout << "Policy Network Parameters:" << std::endl;
    for (size_t i = 0; i < params.size(); ++i) {
        std::cout << "Parameter " << i << ": " << params[i] << std::endl;
        std::cout << params[i] << std::endl; // This prints the actual tensor values
    }
}

std::vector<Agents> load_agents(int player_count, CriticNetwork& critic) {
    std::vector<Agents> agents;
    Policynetwork model;

    /*Load all policy networks and assign to each agent*/
    for (player_count--; player_count >= 0; player_count--) {
        try {
            std::string model_path = "../models/agent_network" + std::to_string(player_count) + ".pt";
            torch::serialize::InputArchive input_archive;
            // Load the model from file into the archive
            input_archive.load_from(model_path);

            // Load the full model (policy network) from the archive
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
        // Load the model from file into the archive
        input_archive.load_from(model_path);

        // Load the full model (policy network) from the archive
        critic.load(input_archive);
        std::cout << "Loading critic network from " << model_path << std::endl;


    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model for critic network" << std::endl;
        exit(EXIT_FAILURE);
    }

    //return the loaded agents, critic is assigned
    return agents;
}

void update_nets(std::vector<Agents>& agents, CriticNetwork& critic, std::vector<databuffer> exper_buff) {

    torch::Tensor targets = torch::tensor({{1.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, torch::dtype(torch::kFloat));
    torch::Tensor loss = torch::tensor(0.004, torch::requires_grad(true));  // This creates a tensor with gradient tracking enabled

    //update each agent
    for (auto& agent : agents)
    {
        agent.policyNetwork.train(); // Set the model to training mode
        torch::optim::Adam optimizer(agent.policyNetwork.parameters(), torch::optim::AdamOptions(0.99).eps(1e-5));

        // Backpropagation
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Loss: " << loss.item<float>() << std::endl;
    }
}

#endif //NETWORKS_H
