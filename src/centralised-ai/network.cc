//
// Created by Viktor Eriksson on 2024-10-04.
//
#include <filesystem>
#include <torch/script.h>
#include "network.h" //structs

/*!
 * Configuration of the networks
 */
int input_size = 7; // Number of input features
int num_actions = 9;
extern int amount_of_players_in_team;
int hidden_size = 5;

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

