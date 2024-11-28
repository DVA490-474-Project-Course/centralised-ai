/* network.h
*==============================================================================
* Author: Viktor Eriksson
* Creation date: 2024-10-04.
* Last modified: 2024-10-24 by Viktor Eriksson
* Description: network header files.
* License: See LICENSE file for license details.
*==============================================================================
*/

#ifndef NETWORK_H
#define NETWORK_H
/* PyTorch C++ API library  */
#include <torch/torch.h>
/* Projects .h files for communication functions. */
#include "communication.h"

namespace centralised_ai {
namespace collective_robot_behaviour
{
/*!
* @brief Struct is the hidden states used for the networks
*
* Struct contains the hidden state (ht_p) and the cell state (ct_p)
*
* Initalised, hidden state and cell state is 3 dim of zeroes in range of the hidden_size
* hidden state array: (2 if bidirectional=True otherwise 1, batch size , if proj size > 0 otherwise hidden size )
* cell state array: (2 if bidirectional=True otherwise 1 * num layers, batch size , hidden size)
*
* @note PyTorch LSTM instructions from https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
*/
  struct HiddenStates {
    torch::Tensor ht_p;
    torch::Tensor ct_p;

    HiddenStates();
  };

  /*!
   * @brief Struct representing a trajectory array used during training.
   *
   * This struct contains state, action probabilities, rewards, new state,
   * policy hidden states (HiddenStates struct), and critic hidden states (HiddenStates struct).
   *
   * Initalised, state,actions and new_state is zeroes. Rewards is empty float
   *
   * @note The concept of a trajectory array is detailed in:
   * "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" - https://arxiv.org/pdf/2103.01955
   */
  struct Trajectory {
    // State, action probabilities, rewards, new state
    /*int robotID;*/
    torch::Tensor state;
    torch::Tensor actions_prob;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor new_state;
    std::vector<HiddenStates> hidden_p;
    HiddenStates hidden_v;
    torch::Tensor critic_value;


    Trajectory()
    : /*robotID(-1),*/
      state(torch::zeros({1, 1,input_size})), //Previous error wrong array size
      actions_prob(torch::zeros({num_actions})),
      rewards(torch::zeros(amount_of_players_in_team)),
      actions(torch::zeros({amount_of_players_in_team})),
      new_state(torch::zeros({1, 1, input_size})) // New state, wrote to same as state dimension

    {}
  };

/*!
 * @brief DataBuffer Struct representing a data buffer for storing data in chunks during training.
 *
 * This struct organizes stored trajectories in chunks.
 * It contains a vector of Trajectory struct, A (GAE) and R (Compute reward-to-go)
 * Additional tensors (A and R) represent accumulated advantages and rewards used in training updates.
 *
 * @note Referred to as "D" in the paper, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" - https://arxiv.org/pdf/2103.01955
 */
struct DataBuffer {
  std::vector<Trajectory> t;
  torch::Tensor A; /*CHANGE NAME WHEN IMPLEMENTING*/
  torch::Tensor R;

  // Constructor
  DataBuffer();
};

/*!
 * @brief PolicyNetwork struct creates a LSTM network along with a forward function.
 *
 *Contains num_layers, output_size, input_size, hidden_size.
 *Batch first = true, linear output_layer.
 *
 */
struct PolicyNetwork : torch::nn::Module {
  const int num_layers;
  const int output_size;

  torch::nn::GRU rnn{nullptr};
  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Linear output_layer{nullptr};

  PolicyNetwork();
  /*!
  * @brief Forward function for the LSTM Policy Network
  *
  * @param[in] input state size [samples,1,states]
  * @param[in] hx hidden state
  * @param[in] cx memory cell
  *
  *@param[out] (Predicted actions, hx new, cx new)
  */
  std::tuple<torch::Tensor, torch::Tensor> Forward(
    torch::Tensor input,
    torch::Tensor hx
  );
};


/*!
 * @brief Struct of the LSTM Critic Network
 *
 * @info  ,Linear output.
 *
 *Contains robotid, x_pos, y_pos and poliycnetwork
 */
struct CriticNetwork : torch::nn::Module {
  torch::nn::GRU rnn{nullptr};
  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Linear value_layer{nullptr};

  CriticNetwork();

  /*!
  * @brief Forward function for the LSTM Critic Network
  *
  * @param[in] input state size [samples,1,states]
  * @param[in] hx hidden state
  * @param[in] cx memory cell
  *
  *@param[out] (Predicted actions, hx new, cx new)
  */
  std::tuple<torch::Tensor, torch::Tensor> Forward(
    torch::Tensor input,
    torch::Tensor hx
  );
};


/*!
 * @brief Agents struct creates a struct of a policy network and robot id to define the robots configuration.
 *
 *Contains robotid, x_pos, y_pos and poliycnetwork
 */
struct Agents {
  int robotId;
  torch::Tensor random_floats; // Declaration only
  float x_pos;
  float y_pos;
  std::shared_ptr<PolicyNetwork> policy_network;

  // Constructor
  Agents(int id, std::shared_ptr<PolicyNetwork> model);
};

/*!
 * @brief short desciption
 *
 * long description
 *
 * @param[in] amount_of_players_in_team number of players in one team
 * @return Returns amount of policy networks as the amount of players into a vector.
 */
std::vector<Agents> CreateAgents(int amount_of_players_in_team);

/*!
 * @brief Save the agents models and the critic network in models folder.
 *
 * This function serializes the parameters of the given agents' policy networks
 * and the critic network into a file. This allows for the preservation of the
 * trained models' weights, enabling later recovery or continuation of training.
 *
 * @param[in] models A constant reference to a vector of Agents containing the
 *               individual agent models to be saved.
 * @param[in] critic A reference to the CriticNetwork instance that will also
 *               be saved along with the agents' models.
 */
void SaveModels(const std::vector<Agents>& models, CriticNetwork& critic);

/*!
 * @brief Load network models via the /models folder.
 *
 * @param[in]  player_count amount of players to load in
 * @param[in] critic A reference to the CriticNetwork
 * @param[out] Returns Agent vector of all policy networks for each robot.
 */
std::vector<Agents> LoadAgents(int player_count, CriticNetwork& critic);


std::vector<Agents> LoadOldAgents(int player_count, CriticNetwork& critic);

void SaveOldModels(const std::vector<Agents>& models, CriticNetwork& critic);


/*!
 * @brief Update all network weights
 *
 * @param[in]  agents policy networks of all agents/robots.
 * @param[in] critic A reference to the CriticNetwork
 * @param[in] exper_buff experience buffer vector.
 */
void UpdateNets(std::vector<Agents>& agents, CriticNetwork& critic,torch::Tensor policy_loss,torch::Tensor critic_loss);
}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //NETWORK_H