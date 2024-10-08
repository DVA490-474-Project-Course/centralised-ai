#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/serialize.h> // For serialization
#include "networks.h"
#include "Communication.h"
int amount_of_players_in_team = 2;

// Define the LSTM network class
#include <torch/torch.h>

/*Get the experience from each action to new states over some time
 Inputs: timesteps, agents, */

int main() {

    //for check amount_of_players_in_team
    //if a agent exist is model, apegnt_network + id then load, else return "models did not load in" and exit
    // Create a policy networks and local variables for each robot

    std::vector<Agents> Models;
    CriticNetwork critic;

    /*create or load agents*/
    //Models = createAgents(amount_of_players_in_team);
    Models = load_agents(amount_of_players_in_team,critic);

    /*Get values in experience buffer*/
    int max_timesteps = 500;
    std::vector<Experience> experience_buffer;
    int timestep = 0;
    while (timestep < max_timesteps) {
        bool done = false;
        for (auto& agent : Models) {
            agent.policyNetwork.train();
            // Train agents and store experience
            Experience exp = step(agent, critic); //send in agent and critic network
            experience_buffer.push_back(exp); //add to experience buffer
            timestep += 1; //Add timestep

            /*Check if done or timestep, then update val*/
            if (exp.done == true || timestep >= max_timesteps) {
                done = true;
                break;
            }
            //std::cout << "Timestep: " << timestep << std::endl;
        }
    }

    update_nets(Models, critic,experience_buffer);
    /*Save or update the models*/
    save_models(Models,critic);
    return 0;
}
