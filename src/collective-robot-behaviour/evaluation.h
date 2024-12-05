

#ifndef EVALUATION_H_H
#define EVALUATION_H_H

#include <torch/torch.h>


namespace centralised_ai
{
namespace collective_robot_behaviour
{
    /*!
    *@brief Saves the reward_to_go to a file.
    * @param reward_to_go: The reward to go tensor, with the shape [batch_size, num_time_steps].
    * @param filename: The filename to save the reward to go to.
    */
    void SaveRewardToGoToFile(const torch::Tensor & reward_to_go, const std::string & filename);

    /*!
    *@brief Loads the reward to go from a file.
    * @param filename: The filename to load the reward to go from.
    * @returns The reward to go tensor, with the shape [batch_size * num_batches, num_time_steps].
    * NOTE: Num batches is the total number of batches that are in the file, i.e. batches from all episodes.
    */
    torch::Tensor LoadRewardToGoFromFile(const std::string & filename);
}
}

#endif