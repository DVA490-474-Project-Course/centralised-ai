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
    int max_timesteps = 5;
    int steps = 0;
    int step_max = 1;

    while (steps < step_max)
    {
        std::vector<Experience> experience_buffer; //set data buffer. reset

        int timestep = 0;
        bool done = false;

        /*Run until timesteps or game is done*/
        while (timestep < max_timesteps) {

            /*Sequentially make each robot do an action*/
            std::vector<int> actions;
            std::vector<float> rewards;
            torch::Tensor act_prob = torch::zeros({max_timesteps, amount_of_players_in_team, 6}); //6 actions for now

            torch::Tensor state = get_states();
            for (auto& agent : Models) {
                //one state, all robots do their policy action
                torch::Tensor output = agent.policyNetwork.forward(state);
                torch::Tensor action = argmax(output);

                //append action probabilities
                actions.push_back(action.item().toInt());
                act_prob.index_put_({timestep, agent.robotId},output);
            }

            //do all actions
                //sen command to do all actions

            torch::Tensor valuefunc = critic.forward(state);
            float reward = get_rewards();  //threshold reward
            Experience exp = {valuefunc,reward,act_prob};//get_expbuff(critic); //send in agent and critic network
            experience_buffer.push_back(exp); //add to experience buffer

            timestep += 1; //Add timestep

        }

        update_nets(Models, critic,experience_buffer); //update networks

        /*Save or update the models*/
        save_models(Models,critic); //save models of all networks(policy and critic)
        steps++;
        std::cout << "-------------------" << std::endl;
    }

    return 0;
}
