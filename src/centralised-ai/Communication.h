//
// Created by viktor on 2024-10-04.
//
#include <torch/torch.h>
#ifndef COMMUNICATION_H
#define COMMUNICATION_H

struct Experience {
    torch::Tensor state;
    torch::Tensor action;
    float reward;
    torch::Tensor next_state;
    bool done;

    Experience(torch::Tensor s, torch::Tensor a, float r, torch::Tensor ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

torch::Tensor get_states() {
    // Example state data stored in a std::vector
    std::vector<float> state_vector = {10.0, 11.0, 12.0,10.0, 11.0, 12.0};
    //get 25 values

    // Convert the std::vector to a tensor and reshape if needed
    return torch::tensor(state_vector).reshape({1, 1, 6}); //change input size
    //                                  {amount to process at time,time steps, dimension input}
}

float get_rewards() {
    float reward = 1.0; //dummy value for now
    return reward;
}


#endif //COMMUNICATION_H
