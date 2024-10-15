#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <torch/serialize.h>
#include "networks.h"
#include "Communication.h"
extern int amount_of_players_in_team = 6;

#include <torch/torch.h>


int main() {
    std::vector<Agents> Models; //Create Models class for each robot.
    CriticNetwork critic; //Create global critic network

    /*create or load agents*/ //need to configure how we should do this for the user
    Models = createAgents(amount_of_players_in_team);
    //Models = load_agents(amount_of_players_in_team,critic); //Load in the trained model



    /*Get values in experience buffer*/
    int max_timesteps = 5;
    int steps = 0;
    int step_max = 1;
    int batch_size = 10;

    while (steps < step_max) {
        std::vector<Experience> data_buffer; //set data buffer D. reset

        for (int i = 1; i <= batch_size; i++) {
            std::vector<Experience> trajectories;
            Experience exp;

            //reset?
            std::vector<int> actions;
            torch::Tensor act_prob = torch::zeros({amount_of_players_in_team, 6}); //6 actions for now

            for (int timestep = 1; timestep < max_timesteps; timestep++) {
                torch::Tensor state = get_states();
                torch::Tensor valNetOutput = critic.forward(state);
                    for (auto& agent : Models) {
                        //one state, all robots do their policy action
                        torch::Tensor output = agent.policyNetwork.forward(state);
                        torch::Tensor action = argmax(output);

                        //append action and probabilities
                        actions.push_back(action.item().toInt());
                        act_prob.index_put_({agent.robotId},output);
                        //std::cout << act_prob << std::endl;
                    } //end for agent

                //execute actions
                exp.state = state;
                exp.rewards = get_rewards();
                exp.act_prob = act_prob; //rows = robot, columns = actions
                exp.val_out = valNetOutput;
                exp.actions = actions;
                exp.new_state = get_states();

                trajectories.push_back(exp); //save in t
                std::cout << trajectories[timestep-1].act_prob << std::endl;
                std::cout << trajectories[timestep-1].act_prob << std::endl;

            } //end for timestep

            //compute GAE and reward-go-to

            //split to mini batches
            for (int l = 0; l < 10; l++) //K = 10 for now
            {
                //data_buffer[l] = ...

            } //end for
        } //end for batch_size

        /*
        for (minibatch k = 1,...K) {
            b = random from data_buffer;

            for (each data chunk in c in mini batch b) {
                update LSTM hidden states for prolicy and critic network frim first hidden state in data chunk.
            } //end for
        } //end for
        */

        //ADAM UPDATE NETWORKS

        update_nets(Models, critic,data_buffer); //update networks
        save_models(Models,critic); //save models of all networks(policy and critic)

    } //end while
    return 0;

}

