#include <torch/torch.h>
#include <iostream>
#include <valarray>
#include <vector>
#include <torch/serialize.h>
#include "networks.h"
#include "Communication.h"
#include <torch/script.h>


extern int amount_of_players_in_team;
extern int num_actions;
extern int buffer_a_b_size;

int main() {
    std::vector<Agents> Models; //Create Models class for each robot.
    CriticNetwork critic; //Create global critic network

    /*create or load agents*/ //need to configure how we should do this for the user
    Models = createAgents(amount_of_players_in_team);
    //Models = load_agents(amount_of_players_in_team,critic); //Load in the trained model


    /*Get values in experience buffer*/
    int max_timesteps = 50;
    int steps = 0;
    int step_max = 5;
    int batch_size = 10;

    while (steps < step_max) {
        std::vector<databuffer> data_buffer; //set data buffer D. reset

        for (int i = 1; i <= batch_size; i++) {
            std::vector<trajectory> trajectories;
            trajectory exp;

            //reset?
            torch::Tensor act_prob = torch::zeros({amount_of_players_in_team, num_actions});

            for (int timestep = 1; timestep < max_timesteps; timestep++) {
                std::vector<int> actions;
                std::vector<float> rewards;

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

                trajectories.push_back(exp);


            } //end for timestep

            //compute GAE and reward-go-to
            torch::Tensor A = torch::rand({max_timesteps, 1}); // Random values between 0 and 1
            torch::Tensor R = torch::rand({max_timesteps, 1}); // Random values between 0 and 1

            //split to mini batches
            int L = 10;
            buffer_a_b_size = L;
            for (int l = 0; l < max_timesteps/L; l++)
            {
                databuffer dat;
                for (int i = l; i < l+L; i++) {
                    dat.t.push_back(trajectories[i]);
                }
                // Create a tensor slice for A and R for the current mini-batch
                dat.A = A.slice(0, l, std::min(l + L, max_timesteps)); // Get slice of A
                dat.R = R.slice(0, l, std::min(l + L, max_timesteps)); // Get slice of R

                data_buffer.push_back(dat); // Store t, A and R in D
            } //end for
        } //end for batch_size

        //“Batch Size” refers to the number of environment steps collected before updating
        //the policy via gradient descent

        //“Minibatch” refers to the number of mini-batches a batch of data is split into
        //“gain” refers to the weight initialization gain of the last network layer
        int len = data_buffer.size();
        for (int k = 1; k < (len) * 0.5; k++) {
            std::vector<databuffer> min_batch;
            int rand_index = torch::randint(0, len, {1}).item<int>();
            min_batch.push_back(data_buffer[rand_index]);

            for (auto it = min_batch.begin(); it != min_batch.end(); it++) {
            std::cout << "aaa" << std::endl;
                //update LSTM hidden states for prolicy and critic network frim first hidden state in data chunk.
            } //end for

        } //end for


        //ADAM UPDATE NETWORKS

        update_nets(Models, critic,data_buffer); //update networks
        save_models(Models,critic); //save models of all networks(policy and critic)

        steps++;
    } //end while
    return 0;

}

