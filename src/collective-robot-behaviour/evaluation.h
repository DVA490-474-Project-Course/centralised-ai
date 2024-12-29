/* evaluation.h
 * ==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-12-05.
 * Last modified: 2024-12-06 by Jacob Johansson
 * Description: Header file for showing evaluated results from training the
 * mappo networks. License: See LICENSE file for license details.
 * ==============================================================================
 */

#ifndef CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_EVALUATION_H_
#define CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_EVALUATION_H_

#include "fstream"
#include "iostream"
#include "matplotlibcpp.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "sstream"
#include "torch/torch.h"
#include "vector"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

/*!
 * @brief Saves the reward to a file with the specified file name.
 * @param[in] mean_reward: The mean reward tensor, with the shape [1].
 * @param[in] episode: The episode number.
 * @param[in] file_name: The filename to save the reward to go to.
 */
void SaveRewardToFile(const torch::Tensor& mean_reward, int32_t episode,
                      const std::string& filename);

/*!
 * @brief Saves the losses to a file with the specified file name.
 * @param[in] kLosses: The losses tensor, with the shape [policy_loss,
 * critic_loss].
 * @param[in] kFileName: The filename to save the losses to.
 */
void SaveLossesToFile(const torch::Tensor& kLosses,
                      const std::string& kFileName);

/*!
 * @brief Loads the reward from a file.
 * @param[in] file_name: The filename to load the reward to go from.
 * @returns The reward tensor, with the shape [num_episodes, 1].
 * @note: Num batches is the total number of batches that are in the file, i.e.
 * batches from all episodes.
 */
torch::Tensor LoadRewardFromFile(const std::string& file_name);

/*!
 * @brief Plots the mean reward for each episode, with reward on the y-axis and
 *        episode index on the x-axis.
 * @param[in] reward: The reward tensor, with the shape [num_episodes, 1].
 * @pre You must call matplotlibcpp::figure(); once before calling this function
 *      in order show the plot.
 */
void PlotReward(const torch::Tensor& reward);

} /* namespace collective_robot_behaviour */
} /* namespace centralised_ai */

#endif /* CENTRALISEDAI_COLLECTIVEROBOTBEHAVIOUR_EVALUATION_H_ */