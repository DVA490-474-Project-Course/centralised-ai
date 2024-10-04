#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/serialize.h> // For serialization

int amount_of_players_in_team = 6;

// Define the LSTM network class
#include <torch/torch.h>

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

// Training function
void train(Policynetwork &model, torch::Tensor inputs, torch::Tensor targets, int epochs, float learning_rate) {
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

struct Agents {
    int robotId;
    float x_pos;
    float y_pos;
    Policynetwork policyNetwork;

    // Constructor
    Agents(int id, float x, float y, Policynetwork network)
        : robotId(id), x_pos(x), y_pos(y), policyNetwork(std::move(network)) {
    }

    // Function to use the policy network for prediction
    torch::Tensor predict(torch::Tensor input) {
        return policyNetwork.forward(input);
    }
};

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

int main() {

    // Generate two random floats between 0 and 1
    torch::Tensor random_floats = torch::rand({2});

    // Access and print the random floats
    float x_pos = random_floats[0].item<float>();
    float y_pos = random_floats[1].item<float>();

    // Create a policy networks and local variables for each robot
    std::vector<Agents> robots;
    for (int i = 0; i < amount_of_players_in_team; i++) {
        Policynetwork model;
        robots.emplace_back(i, x_pos, y_pos, model);
    }

    //Create critic network
    CriticNetwork critic;

    // Example input (5 samples, 1 time step, 3 features each)
    torch::Tensor inputs = torch::tensor({
                                             {{10.0, 11.0, 12.0}},
                                             {{10.0, 11.0, 12.0}},
                                             {{1.0, 1.0, 1.0}}
                                         }, torch::dtype(torch::kFloat));

    // Example target (5 samples, 6 outputs each)
    torch::Tensor targets = torch::tensor({
                                              {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              {0.0, 0.0, 0.0, 0.0, .0, 1.0}
                                          }, torch::dtype(torch::kFloat));

    // Train the model
    int epochs = 10;
    float learning_rate = 0.1;

    //learn each agent
    for (int i = 0; i < amount_of_players_in_team; i++) {

        train(robots[i].policyNetwork, inputs, targets, epochs, learning_rate);

        torch::Tensor test_input = torch::tensor({{{10.0, 11.0, 12.0}}}, torch::dtype(torch::kFloat));
        torch::Tensor output = robots[i].policyNetwork.forward(test_input);
        std::cout << "Output: " << output << std::endl;
    }

    save_models(robots,critic);


    return 0;
}
