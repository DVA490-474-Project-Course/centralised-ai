//
// Created by viktor on 2024-10-23.
//
#include <vector>
#include "network.h"
#include "Communication.h"
#include <tuple>
#include <torch/torch.h>

std::tuple<std::vector<trajectory>, torch::Tensor, torch::Tensor> reset_hidden()
{
    std::vector<trajectory> trajectories;
    hidden_states reset_states;
    trajectory initial_trajectory;
    for (int i = 0; i < amount_of_players_in_team; i++) {
        initial_trajectory.hiddenP.push_back(reset_states);
    }

    trajectories.push_back(initial_trajectory);

    torch::Tensor act_prob = torch::zeros({amount_of_players_in_team, num_actions});
    torch::Tensor action;

    return std::make_tuple(trajectories, act_prob, action);
};

/*Configuration values*/
int max_timesteps = 50;
int steps = 0;
int step_max = 5;
int batch_size = 10;
int amount_of_players_in_team = 6;

void MAPPO(std::vector<Agents> Models,CriticNetwork critic ) {

    while (steps < step_max) {
        std::vector<databuffer> data_buffer; //set data buffer D. reset

        for (int i = 1; i <= batch_size; i++) {

            //Reset initalise
            auto[trajectories,act_prob,action] = reset_hidden();

            for (int timestep = 1; timestep < max_timesteps; timestep++) {

                std::vector<int> actions_agents;
                std::vector<float> rewards;
                trajectory exp;
                hidden_states new_states;


                torch::Tensor state = get_states(); //get the state function
                std::cout << trajectories[timestep-1].hiddenP[0].ht_p << std::endl;
                auto [valNetOutput, V_hx, V_cx] = critic.forward(state, trajectories[timestep-1].hiddenP[0].ht_p, trajectories[timestep-1].hiddenP[0].ct_p);
                    for (auto& agent : Models) {
                        auto [act_prob, hx_new, ct_new] = agent.policyNetwork.forward(state,trajectories[timestep-1].hiddenP[agent.robotId].ht_p,trajectories[timestep-1].hiddenP[agent.robotId].ct_p);
                        //one state, all robots do their policy action
                        exp.actions.index_put_({agent.robotId}, act_prob.squeeze());
                        //append action and probabilities
                        actions_agents.push_back(torch::argmax(act_prob).item().toInt());

                        //Save new hidden states
                        new_states.ht_p = hx_new;
                        new_states.ct_p = ct_new;
                        exp.hiddenP.push_back(new_states); //saved for timestep

                        //FIX SO EACH ROBOT SAVE EACH HIDDEN STATE SO IT IS USED FOR FORWARD FUNCTION OF ITS AGENT

                    } //end for agent

                //execute actions
                exp.state = state;
                exp.rewards = get_rewards();
                exp.ht_v = valNetOutput[0]; //why 0
                exp.new_state = get_states();
                exp.hiddenV.ht_p = V_hx;
                exp.hiddenV.ct_p = V_cx;
                trajectories.push_back(exp);


           } //end for timestep

            //compute GAE and reward-go-to
            torch::Tensor A = torch::rand({max_timesteps, 1}); // Random values between 0 and 1
            torch::Tensor R = torch::rand({max_timesteps, 1}); // Random values between 0 and 1

            //split to chunks, databuffer[timestep] = {t,A,R}
            int L = 10;
            for (int l = 0; l < max_timesteps/L; l++) //T/L
            {
                //Add each trajectory into dat.t value for all timesteps in chunk
                databuffer dat;
                for (int i = l; i < l+L; i++) {
                    dat.t.push_back(trajectories[i]);
                }

                //A and R is callculated in each
                // Create a tensor slice for A and R for the current mini-batch
                dat.A = A.slice(0, l, std::min(l + L, max_timesteps)); // Get slice of A
                dat.R = R.slice(0, l, std::min(l + L, max_timesteps)); // Get slice of R

                data_buffer.push_back(dat); // Store t, A and R in D
            } //end for
        } //end for batch_size

        //“Batch Size” refers to the number of environment steps collected before updating the policy via gradient descent
        //“Minibatch” refers to the number of mini-batches a batch of data is split into
        //“gain” refers to the weight initialization gain of the last network layer

        int len = data_buffer.size();
        std::vector<databuffer> min_batch;
        for (int k = 0; k <= 5; k++) {
            int rand_index = torch::randint(0, len, {1}).item<int>();
            min_batch.push_back(data_buffer[rand_index]); //Take random saved chunks
            //Send in the minibatch and update the hidden states by the saved stateds and hidden values.
                //for each timestep in batch
                for (int i = 0; i < min_batch[k].t.size(); i++) {
                    auto state_read = min_batch[k].t[i].state; //get state for teh timestep in minibatch

                    //For each agents actions in one timestep //UPDATE SO ALL TIMESTEPS GET SENT AS ONE VECTOR AS STATE
                    for (int agent = 0; agent < amount_of_players_in_team; agent++)
                    {
                        auto hx_read = min_batch[k].t[i].hiddenP[agent].ht_p;
                        auto ct_read = min_batch[k].t[i].hiddenP[agent].ct_p;
                        std::cout << state_read << std::endl;
                        //Update all policynetworks
                        Models[agent].policyNetwork.forward(state_read,hx_read,ct_read);
                    }
                    auto hv_read = min_batch[k].t[i].hiddenV.ht_p;
                    auto cv_read = min_batch[k].t[i].hiddenV.ct_p;
                    critic.forward(state_read,hv_read,cv_read);
                } //update LSTM hidden states for policy and critic network from first hidden state in data chunk.
        } //end for


        //ADAM UPDATE NETWORKS

        update_nets(Models, critic,data_buffer); //update networks
        save_models(Models,critic); //save models of all networks(policy and critic)

        steps++;
    } //end while
}