//
// Created by viktor on 2024-10-04.
//

#include <torch/torch.h>
#include "networks.h"
#ifndef COMMUNICATION_H
#define COMMUNICATION_H

extern int input_size;
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;
int buffer_a_b_size;

struct hidden_states {
    torch::Tensor ht_p;
    torch::Tensor ct_p;

    hidden_states()
    :
    ht_p(torch::zeros({amount_of_players_in_team, 1, hidden_size})), // Initialize action probabilities with a 1x6 zero tensor
    ct_p(torch::zeros({amount_of_players_in_team, 1, hidden_size})) // Initialize action probabilities with a 1x6 zero tensor

    {}
};

struct trajectory {
    // State, action probabilities, rewards, new state
    int robotID;
    torch::Tensor state;
    torch::Tensor  ht_v;
    torch::Tensor  ct_v;
    torch::Tensor actions;
    float rewards;
    torch::Tensor new_state;
    std::vector<hidden_states> hiddenP;
    hidden_states hiddenV;


    trajectory()
        : robotID(-1),
          state(torch::zeros({1, amount_of_players_in_team})), // Initialize state with a 1x6 zero tensor
          ht_v(torch::zeros(hidden_size)),
          ct_v(torch::zeros(hidden_size)),
          actions(torch::zeros({amount_of_players_in_team,num_actions})),
          rewards(float{1}), // Initialize rewards as an empty vector
          new_state(torch::zeros({1, amount_of_players_in_team})) // Initialize new_state with a 1x6 zero tensor
    {}
};

struct databuffer {
    std::vector<trajectory> t;
    torch::Tensor A = torch::zeros({1,buffer_a_b_size});
    torch::Tensor R = torch::zeros({1,buffer_a_b_size});
    //torch::Tensor<float> R;

};

torch::Tensor get_states() {
    // Example state data stored in a std::vector
    torch::Tensor state_vector = torch::randn({1,1,input_size});
    //torch::Tensor state_vector = torch::randn({1,1,input_size});

    //std::cout << state_vector << std::endl;
    //get 25 values

    // Convert the std::vector to a tensor and reshape if needed
    return state_vector;//change input size
    //                                  {amount to process at time,time steps, dimension input}
}

float get_rewards() {
    float reward = 1.0;
    return reward;
}


#endif //COMMUNICATION_H
