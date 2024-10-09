#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/serialize.h>
#include "networks.h"
#include "Communication.h"
int amount_of_players_in_team = 6;
#include <torch/torch.h>

int main() {

    std::vector<Agents> Models; //Create Models class for each robot.
    CriticNetwork critic; //Create global critic network

    /*create or load agents*/ //need to configure how we should do this for the user
    Models = createAgents(amount_of_players_in_team);
    //Models = load_agents(amount_of_players_in_team,critic); //Load in the trained model
    // print_parameters(Models[0]); //check parameters if they are the updated or same


    /*Get values in experience buffer*/
    int max_timesteps = 500;
    int runs = 0;
    int max_runs = 100;
    while (runs < max_runs)
    {
        std::vector<Experience> experience_buffer; //reset buffer
        int timestep = 0;
        bool done = false;
        /*Run until timesteps or game is done*/
        while (timestep < max_timesteps) {
            /*Sequentially make each robot do an action*/
            for (auto& agent : Models) {
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

        update_nets(Models, critic,experience_buffer); //update networks
        /*Save or update the models*/
        save_models(Models,critic); //save models of all networks(policy and critic)
        runs++;
        std::cout << "-------------------" << std::endl;

    }
    return 0;
}
