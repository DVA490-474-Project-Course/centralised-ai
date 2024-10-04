//
// Created by viktor on 2024-10-04.
//

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


//void get_states{};

//void get_rewards{};

#endif //COMMUNICATION_H
