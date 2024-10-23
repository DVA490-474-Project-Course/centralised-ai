//
// Created by viktor on 2024-10-23.
//

#ifndef COMMUNICATION_H
#define COMMUNICATION_H
#include <torch/torch.h>

extern int input_size;
extern int num_actions;
extern int amount_of_players_in_team;
extern int hidden_size;


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
    torch::Tensor A = torch::zeros({1,hidden_size});
    torch::Tensor R = torch::zeros({1,hidden_size});
    //torch::Tensor<float> R;

};

torch::Tensor get_states();


float get_rewards();

#endif //COMMUNICATION_H
