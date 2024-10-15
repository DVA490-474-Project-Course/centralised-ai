//
// Created by viktor on 2024-10-04.
//

#include <torch/torch.h>
#include "networks.h"
#ifndef COMMUNICATION_H
#define COMMUNICATION_H

struct Experience {
    // State, action probabilities, rewards, new state
    torch::Tensor state;
    torch::Tensor act_prob;
    torch::Tensor  val_out;
    std::vector<int> actions;
    float rewards;
    torch::Tensor new_state;


    Experience()
        : state(torch::zeros({1, 6})), // Initialize state with a 1x6 zero tensor
          act_prob(torch::zeros({1, 6})), // Initialize action probabilities with a 1x6 zero tensor
          val_out(torch::zeros(1)),
          actions(std::vector<int>{1,6}),
          rewards(float{1}), // Initialize rewards as an empty vector
          new_state(torch::zeros({1, 6})) // Initialize new_state with a 1x6 zero tensor
    {}
};


torch::Tensor get_states() {
    // Example state data stored in a std::vector
    torch::Tensor state_vector = torch::randn({1,1,6});
    //std::cout << state_vector << std::endl;
    //get 25 values

    // Convert the std::vector to a tensor and reshape if needed
    return state_vector;//change input size
    //                                  {amount to process at time,time steps, dimension input}
}

float get_rewards() {
    float reward = 1.0; //dummy value for now
    return reward;
}


#endif //COMMUNICATION_H
