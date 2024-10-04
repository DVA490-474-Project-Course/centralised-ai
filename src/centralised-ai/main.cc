#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/serialize.h> // For serialization
#include "networks.h"
#include "Communication.h"
int amount_of_players_in_team = 6;

// Define the LSTM network class
#include <torch/torch.h>

/*Get the experience from each action to new states over some time
 Inputs: timesteps, agents, */

std::vector<Experience> experience(std::vector<Agents>& robots) {
    int max_timesteps = 100;
    std:std::vector<Experience> experience_buffer;

    for (int i = 0; i < max_timesteps; i++) {
        for (int id = 0; id < amount_of_players_in_team; ++id) {

            //get state
            torch::Tensor state = torch::tensor({{{10.0, 11.0, 12.0}}}, torch::dtype(torch::kFloat)); //get_state()
            //get action
            torch::Tensor action = robots[id].policyNetwork.forward(state);
            //make action
            //send_action(id,action)

            //get feedback
            //[reward,next_state,done]
            //experience_buffer.push_back({state, action, reward, next_state, done});

        }
    }
    return experience_buffer;
}




int main() {

    //for check amount_of_players_in_team
    //if a agent exist is model, apegnt_network + id then load, else return "models did not load in" and exit
    // Create a policy networks and local variables for each robot

    std::vector<Agents> robots;

    //Create agents
    robots = createAgents(amount_of_players_in_team);
    //loadModels(robots,amount_of_players_in_team);

    //Create critic network
    CriticNetwork critic;

    //Get values in experience buffer
    //std::vector<Experience> buffer = experience(robots);

    for (int i = 0; i < amount_of_players_in_team; i++) {
        torch::Tensor test_input = torch::tensor({{{10.0, 11.0, 12.0}}}, torch::dtype(torch::kFloat));
        torch::Tensor output = robots[i].policyNetwork.forward(test_input);
        std::cout << "Output: " << output << std::endl;
    }

    //save or update
    save_models(robots,critic);


    return 0;
}
