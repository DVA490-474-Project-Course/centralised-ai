
#include "evaluation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    void SaveRewardToFile(const torch::Tensor & mean_reward, int32_t episode, const std::string & filename)
    {
        /* Save the reward tensor to a file. */
        std::ofstream file(filename, std::ios::app);

        if(!file.is_open())
        {
            std::cerr << "Could not open file: " << filename << std::endl;
            return;
        }

        /* Write the episode index as the first column. */
        file << episode << ",";

        /* Write the mean reward. */
        file << mean_reward.item<float>() << std::endl;


        file.close();
    }

    torch::Tensor LoadRewardFromFile(const std::string & filename)
    {
        /* Load the reward tensor from a file. */
        std::ifstream file(filename, std::ios::in);

        if(!file.is_open())
        {
            std::cerr << "Could not open file: " << filename << std::endl;
            return torch::zeros({0, 0});
        }

        /* Create tensor with the shape [num_columns, num_rows]. */
        std::vector<std::vector<float>> reward_data;
        std::string line;
        
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::vector<float> reward_row;
            std::string value;
            while (std::getline(iss, value, ','))
            {
                reward_row.push_back(std::stof(value));
            }

            reward_data.push_back(reward_row);
        }

        file.close();

        /* Create a new tensor from the vector of vectors. */
        int32_t num_columns = reward_data.size();
        int32_t num_rows = reward_data[0].size();

        torch::Tensor reward = torch::zeros({num_columns, num_rows});

        for (int32_t c = 0; c < num_columns; c++)
        {
            for (int32_t r = 0; r < num_rows; r++)
            {
                reward[c][r] = reward_data[c][r];
            }
        }

        return reward;
    }
}
}