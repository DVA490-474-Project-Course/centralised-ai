//
// Created by viktor on 2024-10-04.
//

#include <torch/torch.h>
#include "network.h"

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


