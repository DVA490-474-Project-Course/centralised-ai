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
      state(torch::zeros({1, 1,num_global_states})), //Previous error wrong array size
      actions_prob(torch::zeros({num_actions})),
      rewards(torch::zeros(amount_of_players_in_team)),
      actions(torch::zeros({amount_of_players_in_team})),
      new_state(torch::zeros({1, 1, num_global_states})) // New state, wrote to same as state dimension

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
 * @brief Struct of the policy network based on the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" - https://arxiv.org/pdf/2103.01955".
 *
 *Contains num_layers, output_size, input_size, hidden_size.
 *Batch first = true, linear output_layer.
 *
 */
struct PolicyNetwork : torch::nn::Module {
  const int num_layers;
  const int output_size;
  
  torch::nn::Linear layer1{nullptr};
  torch::nn::Linear layer2{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear output_layer{nullptr};

  /*!
  * @brief Forward function for the LSTM Policy Network
  *
  * @param[in] input state size [samples,1,states]
  * @param[in] hx hidden state
  * @param[in] cx memory cell
  *
  *@param[out] (Predicted actions, hx new, cx new)
  */
  PolicyNetwork();
  std::tuple<torch::Tensor, torch::Tensor> Forward(
    torch::Tensor input,
    torch::Tensor hx
  );
};


/*!
 * @brief Struct of the Critic Network based on the paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" - https://arxiv.org/pdf/2103.01955".
 */
struct CriticNetwork : torch::nn::Module {
  torch::nn::Linear layer1{nullptr};
  torch::nn::Linear layer2{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear output_layer{nullptr};

  /*!
  * @brief Forward function for the LSTM Critic Network
  *
  * @param[in] input state size [samples,1,states]
  * @param[in] hx hidden state
  * @param[in] cx memory cell
  *
  *@param[out] (Predicted actions, hx new, cx new)
  */
  CriticNetwork();
  std::tuple<torch::Tensor, torch::Tensor> Forward(
    torch::Tensor input,
    torch::Tensor hx
  );
};

/*!
 * @brief Create a new policy network with reset hidden states.
 * @return Returns a new PolicyNetwork instance.
 */
PolicyNetwork CreatePolicy();

/*!
 * @brief Save the agents models and the critic network in models folder.
 *
 * This function serializes the parameters of the given agents' policy networks
 * and the critic network into a file. This allows for the preservation of the
 * trained models' weights, enabling later recovery or continuation of training.
 *
 * @param[in] policy A reference to the policy network instance.
 * @param[in] critic A reference to the critic network instance.
*/
void SaveNetworks(PolicyNetwork & policy, CriticNetwork & critic);

/*!
 * @brief Load network models via the /models folder.
 *
 * @param[in] policy A reference to the PolicyNetwork
 * @param[in] critic A reference to the CriticNetwork
 */
void LoadNetworks(PolicyNetwork& policy, CriticNetwork& critic);

/*!
 * @brief Load old network models via the /models/old_agents folder.
 *
 * @param[in] policy A reference to the PolicyNetwork
 * @param[in] critic A reference to the CriticNetwork
 */
void LoadOldNetworks(PolicyNetwork& policy, CriticNetwork& critic);

  /*!
 * @brief Save the old policy and the critic network in models/old_agents folder.
 *
 * This function serializes the parameters of the given agents' policy networks
 * and the critic network into a file. This allows for the preservation of the
 * trained models' weights, enabling later recovery or continuation of training.
 *
 * @param[in] policy A reference to the old policy network instance.
 * @param[in] critic A reference to the old critic network instance.
 */
void SaveOldNetworks(PolicyNetwork & policy, CriticNetwork & critic);

/*!
 * @brief Update all network weights from loss functions
 *
 * @param[in] policy A reference to the policy network.
 * @param[in] critic A reference to the critic network.
 * @param[in] policy_loss A tensor value representing the policy loss.
 * @param[in] critic_loss A tensor value representing the critic loss.
 */
void UpdateNets(PolicyNetwork& policy, CriticNetwork& critic, torch::Tensor policy_loss, torch::Tensor critic_loss);

}/*namespace centralised_ai*/
}/*namespace collective_robot_behaviour*/
#endif //NETWORK_H