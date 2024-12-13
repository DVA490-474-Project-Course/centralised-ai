

#ifndef EVALUATION_H_H
#define EVALUATION_H_H

#include <torch/torch.h>


namespace centralised_ai
{
namespace collective_robot_behaviour
{
    /*!
    *@brief Saves the reward to a file.
    * @param reward_to_go: The mean reward tensor, with the shape [1].
    * @param episode: The episode number.
    * @param filename: The filename to save the reward to go to.
    */
    void SaveRewardToFile(const torch::Tensor & mean_reward_to_go, int32_t episode, const std::string & filename);

    /*!
    *@brief Loads the reward from a file.
    * @param filename: The filename to load the reward to go from.
    * @returns The reward tensor, with the shape [num_episodes, 1].
    * NOTE: Num batches is the total number of batches that are in the file, i.e. batches from all episodes.
    */
    torch::Tensor LoadRewardFromFile(const std::string & filename);

    /*!
    * @brief Plots the mean reward for each episode, with reward on the y-axis and episode index on the x-axis.
    * @param reward: The reward tensor, with the shape [num_episodes, 1].
    * @pre You must call matplotlibcpp::figure(); once before calling this function in order show the plot.
    */
    void PlotReward(const torch::Tensor & reward);
}
}

#endif