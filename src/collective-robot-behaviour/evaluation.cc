
#include "evaluation.h"
#include <iostream>
#include <fstream>

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
}
}