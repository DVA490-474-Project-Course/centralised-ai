
#include "evaluation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace centralised_ai
{
namespace collective_robot_behaviour
{
    void SaveRewardToGoToFile(const torch::Tensor & reward_to_go, const std::string & filename)
    {
        /* Save the reward to go tensor to a file. */
        std::ofstream file(filename, std::ios::app);

        if(!file.is_open())
        {
            std::cerr << "Could not open file: " << filename << std::endl;
            return;
        }

        int32_t batch_size = reward_to_go.size(0);
        int32_t num_time_steps = reward_to_go.size(1);

        for (int32_t b = 0; b < batch_size; b++)
        {
            /* Write the batch index as first column. */
            file << b << ",";
            
            for (int32_t t = 0; t < num_time_steps; t++)
            {
                /* Write the reward to go for each time step. */
                file << reward_to_go[b][t].item<float>() << ",";
            }

            file << std::endl;
        }

        file.close();
    }

    torch::Tensor LoadRewardToGoFromFile(const std::string & filename)
    {
        /* Load the reward to go tensor from a file. */
        std::ifstream file(filename, std::ios::in);

        if(!file.is_open())
        {
            std::cerr << "Could not open file: " << filename << std::endl;
            return torch::zeros({0, 0});
        }

        /* Create tensor with the shape [num_columns, num_rows]. */
        std::vector<std::vector<float>> reward_to_go_data;
        std::string line;
        
        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::vector<float> reward_to_go_row;
            std::string value;
            while (std::getline(iss, value, ','))
            {
                reward_to_go_row.push_back(std::stof(value));
            }

            reward_to_go_data.push_back(reward_to_go_row);
        }

        file.close();

        /* Create a new tensor from the vector of vectors. */
        int32_t num_columns = reward_to_go_data.size();
        int32_t num_rows = reward_to_go_data[0].size();

        torch::Tensor reward_to_go = torch::zeros({num_columns, num_rows});

        for (int32_t c = 0; c < num_columns; c++)
        {
            for (int32_t r = 0; r < num_rows; r++)
            {
                reward_to_go[c][r] = reward_to_go_data[c][r];
            }
        }

        return reward_to_go;
    }
}
}