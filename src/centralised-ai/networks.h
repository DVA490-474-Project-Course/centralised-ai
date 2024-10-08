//
// Created by Viktor Eriksson on 2024-10-04.
//
#include <filesystem>
#include <torch/script.h>
#ifndef NETWORKS_H
#define NETWORKS_H
#include "Communication.h"
extern int input_size = 6; // Number of input features

struct Policynetwork : torch::nn::Module {
    const int hidden_size = 4; // Number of LSTM hidden units
    const int num_layers = 1; // Number of LSTM layers
    const int output_size = 6; // Number of output classes

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_layer{nullptr};

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
};

struct CriticNetwork : torch::nn::Module {
     // Number of input features
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

Experience step(Agents& agent, CriticNetwork& critic) {

    //get current state
    torch::Tensor state = get_states();

    //get CRITIC predictions of actions
    torch::Tensor outputCritic = critic.forward(state);
    //std::cout << "Critic Output: " << outputCritic << std::endl;

    //get Policy predictions of actions of agent
    torch::Tensor output = agent.policyNetwork.forward(state);
    //std::cout << "Policy Output: " << output << std::endl;

    torch::Tensor action = argmax(output);// Action is the highest prediction
    //std::cout << "Action: " << action << std::endl;

    float reward = get_rewards();  //threshold reward
    // Generate the next state (replace with your own logic)
    torch::Tensor next_state = get_states();  // Replace with actual next state logic
    bool done = false;  //goal, game is done!

    // Return the experience
    return Experience(state, action, reward, next_state, done);

}

// Training function
void training(Policynetwork &model, torch::Tensor inputs, torch::Tensor targets, int epochs, float learning_rate) {
    model.train(); // Set the model to training mode

    // Define loss function and optimizer
    torch::nn::MSELoss loss_fn;
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        torch::Tensor predictions = model.forward(inputs);
        torch::Tensor loss = loss_fn(predictions, targets);

        // Backpropagation
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Print loss every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch [" << epoch << "/" << epochs << "] Loss: " << loss.item<float>() << std::endl;
        }
    }
}

void update_nets(std::vector<Agents>& agents, CriticNetwork& critic, std::vector<Experience> exper_buff) {
    torch::Tensor inputs = torch::tensor({
                                                {{10.0, 11.0, 12.0,10.0, 11.0, 12.0}},
                                                {{10.0, 11.0, 12.0,10.0, 11.0, 12.0}},
                                                {{1.0, 1.0, 1.0,10.0, 11.0, 12.0}}
                                            }, torch::dtype(torch::kFloat));

    torch::Tensor targets = torch::tensor({
                                              {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              {0.0, 0.0, 0.0, 0.0, .0, 1.0}
                                          }, torch::dtype(torch::kFloat));


    for (auto& agent : agents)
    {
        training(agent.policyNetwork, inputs, targets, 1, 1);
    }
}
#endif //NETWORKS_H
