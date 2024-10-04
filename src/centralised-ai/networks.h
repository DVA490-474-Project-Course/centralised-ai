//
// Created by Viktor Eriksson on 2024-10-04.
//
#include <filesystem>
#include <torch/script.h>
#ifndef NETWORKS_H
#define NETWORKS_H

struct Policynetwork : torch::nn::Module {
    const int input_size = 3; // Number of input features
    const int hidden_size = 4; // Number of LSTM hidden units
    const int num_layers = 1; // Number of LSTM layers
    const int output_size = 6; // Number of output classes

    Policynetwork()
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          output_layer(register_module("output_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_output = lstm->forward(x);
        auto output = std::get<0>(lstm_output); // Output from LSTM
        output = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // Last time step output
        return output_layer(output); // Final output
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_layer{nullptr};
};

struct CriticNetwork : torch::nn::Module {
    const int input_size = 3; // Number of input features
    const int hidden_size = 4; // Number of LSTM hidden units
    const int num_layers = 1; // Number of LSTM layers
    const int output_size = 1; // Single value output for critic

    CriticNetwork()
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          value_layer(register_module("value_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_output = lstm->forward(x);
        auto output = std::get<0>(lstm_output); // Output from LSTM
        output = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // Last time step output
        return value_layer(output); // Final output (value estimation)
    }

private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear value_layer{nullptr};
};

struct Agents {
    // Generate two random floats between 0 and 1
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

    // Function to use the policy network for prediction
    torch::Tensor predict(torch::Tensor input) {
        return policyNetwork.forward(input);
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

void save_models(const std::vector<Agents>& models,CriticNetwork &critic) {
    for (const auto& agent : models) {
        std::string model_path = "../models/agent_network" + std::to_string(agent.robotId) + ".pt";

        try {
            torch::serialize::OutputArchive output_archive;
            agent.policyNetwork.save(output_archive); // Serialize the model
            output_archive.save_to(model_path); // Save the serialized data to the specified file
            //std::cout << "Model saved to " << model_path << std::endl; // Confirmation message
        } catch (const std::exception& e) {
            std::cerr << "Error saving model for Agent " << agent.robotId << ": " << e.what() << std::endl;
        }
    }
    try {
        std::string model_path = "../models/critic_network.pt";
        torch::serialize::OutputArchive output_archive;
        critic.save(output_archive); // Serialize the model
        output_archive.save_to(model_path); // Save the serialized data to the specified file
        //std::cout << "Model saved to " << model_path << std::endl; // Confirmation message
    } catch (const std::exception& e) {
        std::cerr << "Error saving model for critic "  << ": " << e.what() << std::endl;
    }

}

#endif //NETWORKS_H
