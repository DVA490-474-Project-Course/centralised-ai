//
// Created by viktor on 2024-10-23.

#ifndef NETWORK_H
#define NETWORK_H
#include <torch/torch.h>
#include "Communication.h"

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
        : lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)),
          value_layer(register_module("value_layer", torch::nn::Linear(hidden_size, output_size))) {
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
         torch::Tensor input,
         torch::Tensor hx,
         torch::Tensor cx) {

        auto hidden_states = std::make_tuple(hx, cx);
        auto lstm_output = lstm->forward(input, hidden_states);
        auto val = std::get<0>(lstm_output); //get all timesteps of Output
        auto hx_new = std::get<0>(std::get<1>(lstm_output)); // Hidden state (hx)
        auto cx_new = std::get<1>(std::get<1>(lstm_output)); // Cell state (cx)
        //WRONG DIMENSIONS HERE; DONT KNOW WHY THO! POLICY COULD HAVE SAME PROBLEM!
        val = val.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}); // Last time step output

        // Print the sizes of the tensors for debugging
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
    /*!
     * @brief explain each parameter
     */
    int robotId;
    float x_pos = random_floats[0].item<float>();
    float y_pos= random_floats[1].item<float>();
    Policynetwork policyNetwork;

    // Constructor
    Agents(int id, Policynetwork network)
        : robotId(id), policyNetwork(std::move(network)) {
    }


};


/*!
 * @brief short desciption
 *
 * long description
 *
 * @param[in] amount_of_players_in_team number of players in one team
 * @return Returns amount of policy networks as the amount of players into a vector.
 */
std::vector<Agents> createAgents(int amount_of_players_in_team);

void save_models(const std::vector<Agents>& models, CriticNetwork& critic);

void print_parameters(const Agents& agent);

std::vector<Agents> load_agents(int player_count, CriticNetwork& critic);

void update_nets(std::vector<Agents>& agents, CriticNetwork& critic, std::vector<databuffer> exper_buff);


#endif //NETWORK_H
