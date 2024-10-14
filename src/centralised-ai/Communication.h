//
// Created by viktor on 2024-10-04.
//
#include <torch/torch.h>
#ifndef COMMUNICATION_H
#define COMMUNICATION_H

struct Experience {

    torch::Tensor ValueFunc;
    float reward;
    //std::vector<float> actionprobs;
    torch::Tensor act_prob ;

    Experience(torch::Tensor ValueFunc, float reward,torch::Tensor actpr)
        : ValueFunc(ValueFunc), reward(reward), act_prob(actpr) {}



};

torch::Tensor get_states() {
    // Example state data stored in a std::vector
    torch::Tensor state_vector = torch::randn({1,1,6});
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
