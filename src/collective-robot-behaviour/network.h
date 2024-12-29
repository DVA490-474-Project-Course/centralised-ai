/* network.h
 *==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-10-04.
 * Last modified: 2024-10-24 by Viktor Eriksson
 * Description: network header files.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_NETWORK_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_NETWORK_H_

#include "../../src/common_types.h"
#include "communication.h"
#include "filesystem"
#include "torch/script.h"
#include "torch/torch.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Struct is the hidden states used for the networks
 *
 * Struct contains the hidden state (ht_p) and the cell state (ct_p)
 *
 * Initalised, hidden state and cell state is 3 dim of zeroes in range of the
 * hidden_size hidden state array: (2 if bidirectional=True otherwise 1, batch
 * size , if proj size > 0 otherwise hidden size ) cell state array: (2 if
 * bidirectional=True otherwise 1 * num layers, batch size , hidden size)
 *
 * @note PyTorch LSTM instructions from
 * https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
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
 * policy hidden states (HiddenStates struct), and critic hidden states
 * (HiddenStates struct).
 *
 * Initalised, state,actions and new_state is zeroes. Rewards is empty float
 *
 * @note The concept of a trajectory array is detailed in:
 * "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" -
 * https://arxiv.org/pdf/2103.01955
 */
struct Trajectory {
  torch::Tensor state;
  torch::Tensor actions_prob;
  torch::Tensor actions;
  torch::Tensor rewards;
  torch::Tensor new_state;
  std::vector<HiddenStates> hidden_states_policy;
  HiddenStates hidden_states_critic;
  torch::Tensor critic_value;

  Trajectory();
};

/*!
 * @brief DataBuffer Struct representing a data buffer for storing data in
 * chunks during training.
 *
 * This struct organizes stored trajectories in chunks.
 * It contains a vector of Trajectory struct, A (GAE) and R (Compute
 * reward-to-go) Additional tensors (A and R) represent accumulated advantages
 * and rewards used in training updates.
 *
 * @note Referred to as "D" in the paper, "The Surprising Effectiveness of PPO
 * in Cooperative Multi-Agent Games" - https://arxiv.org/pdf/2103.01955
 */
struct DataBuffer {
  std::vector<Trajectory> t;
  torch::Tensor A;
  torch::Tensor R;

  DataBuffer();
};

/*!
 * @brief Struct of the policy network based on the paper "The Surprising
 * Effectiveness of PPO in Cooperative Multi-Agent Games" -
 * https://arxiv.org/pdf/2103.01955".
 *
 * Contains kNumLayers,,, kOutputSize, input_size, hidden_size.
 * Batch first = true, linear output_layer.
 *
 */
struct PolicyNetwork : torch::nn::Module {
  const int kNumLayers;
  const int kOutputSize;

  torch::nn::Linear layer1{nullptr};
  torch::nn::Linear layer2{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear output_layer{nullptr};

  PolicyNetwork();

  /*!
   * @brief Forward function for the LSTM Policy Network
   *
   * @param[in] input state size [samples,1,states]
   * @param[in] hx hidden state
   * @param[in] cx memory cell
   *
   * @returns A tuple with the values,,, (Predicted actions, hx new, cx new)
   */
  std::tuple<torch::Tensor, torch::Tensor> Forward(torch::Tensor input,
                                                   torch::Tensor hx);
};

/*!
 * @brief Struct of the Critic Network based on the paper "The Surprising
 * Effectiveness of PPO in Cooperative Multi-Agent Games" -
 * https://arxiv.org/pdf/2103.01955".
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
   * @returns A tuple with the values (Predicted actions, hx new, cx new)
   */
  CriticNetwork();
  std::tuple<torch::Tensor, torch::Tensor> Forward(torch::Tensor input,
                                                   torch::Tensor hx);
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
void SaveNetworks(PolicyNetwork& policy, CriticNetwork& critic);

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
 * @brief Save the old policy and the critic network in models/old_agents
 * folder.
 *
 * This function serializes the parameters of the given agents' policy networks
 * and the critic network into a file. This allows for the preservation of the
 * trained models' weights, enabling later recovery or continuation of training.
 *
 * @param[in] policy A reference to the old policy network instance.
 * @param[in] critic A reference to the old critic network instance.
 */
void SaveOldNetworks(PolicyNetwork& policy, CriticNetwork& critic);

/*!
 * @brief Update all network weights from loss functions
 *
 * @param[in] policy A reference to the policy network.
 * @param[in] critic A reference to the critic network.
 * @param[in] policy_loss A tensor value representing the policy loss.
 * @param[in] critic_loss A tensor value representing the critic loss.
 */
void UpdateNets(PolicyNetwork& policy, CriticNetwork& critic,
                torch::Tensor policy_loss, torch::Tensor critic_loss);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_NETWORK_H_ */