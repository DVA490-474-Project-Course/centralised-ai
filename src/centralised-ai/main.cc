#include <torch/torch.h>
#include <iostream>
#include <vector>

int amount_of_players_in_team = 6;

// Define the LSTM network class
struct Policynetwork : torch::nn::Module {
    // Configuration parameters
    const int input_size = 3; // Number of input features
    const int hidden_size = 4; // Number of LSTM hidden units
    const int actions = 6; // Number of output classes (0 to 6)
    const int num_layers = 1;
    const int output_size = 6;

    Policynetwork()
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          output_layer(register_module("output_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        // Pass through LSTM
        auto lstm_output = lstm->forward(x);

        // Take the output of the last time step
        auto output = std::get<0>(lstm_output); // output from LSTM
        output = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // Last time step output

        // Pass through the output layer
        output = output_layer(output); // Final output
        return output;
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_layer{nullptr};
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
    Policynetwork critic;

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
    int epochs = 100;
    float learning_rate = 0.1;

    for (int i = 0; i < amount_of_players_in_team; i++) {
        train(robots[i].policyNetwork, inputs, targets, epochs, learning_rate);
    }

    //Test the policy network
    for (int i = 0; i < amount_of_players_in_team; i++) {
        // Test the model with a new input
        torch::Tensor test_input = torch::tensor({{{10.0, 11.0, 12.0}}}, torch::dtype(torch::kFloat));
        torch::Tensor output = robots[i].policyNetwork.forward(test_input);
        std::cout << "Output: " << output << std::endl;
    }


    return 0;
}
