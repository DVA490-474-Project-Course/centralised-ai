/* evaluation.cc
 * ==============================================================================
 * Author: Viktor Eriksson, Jacob Johansson
 * Creation date: 2024-12-05.
 * Last modified: 2024-12-06 by Jacob Johansson
 * Description: Source file for showing evaluated results from training the
 * mappo networks. License: See LICENSE file for license details.
 * ==============================================================================
 */

#include "evaluation.h"
#include "fstream"
#include "iostream"
#include "matplotlibcpp.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "sstream"
#include "torch/torch.h"
#include "vector"

namespace centralised_ai {
namespace collective_robot_behaviour {

void SaveLossesToFile(const torch::Tensor& kLosses,
                      const std::string& kFileName) {
  std::ofstream file(kFileName, std::ios::app);

  if (!file.is_open()) {
    std::cerr << "Could not open file: " << kFileName << std::endl;
    return;
  }

  file << kLosses[0].item<float>() << "," << kLosses[1].item<float>()
       << std::endl;
  file.close();
}

void SaveRewardToFile(const torch::Tensor& mean_reward, int32_t episode,
                      const std::string& file_name) {
  /* Save the reward tensor to a file. */
  std::ofstream file(file_name, std::ios::app);

  if (!file.is_open()) {
    std::cerr << "Could not open file: " << file_name << std::endl;
    return;
  }

  /* Write the episode index as the first column. */
  file << episode << ",";

  /* Write the mean reward. */
  file << mean_reward.item<float>() << std::endl;

  file.close();
}

torch::Tensor LoadRewardFromFile(const std::string& filename) {
  /* Load the reward tensor from a file. */
  std::ifstream file(filename, std::ios::in);

  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return torch::zeros({0, 0});
  }

  /* Create tensor with the shape [num_columns, num_rows]. */
  std::vector<std::vector<float>> reward_data;
  std::string line;

  std::istringstream reader(line);
  std::vector<float> reward_row;
  std::string value;

  while (std::getline(file, line)) {
    while (std::getline(reader, value, ',')) {
      reward_row.push_back(std::stof(value));
    }

    reward_data.push_back(reward_row);

    reward_row.clear();
    reward_row.clear();
  }

  file.close();

  /* Create a new tensor from the vector of vectors. */
  int32_t num_columns = reward_data.size();
  int32_t num_rows = reward_data[0].size();

  torch::Tensor reward = torch::zeros({num_columns, num_rows});

  for (int32_t c = 0; c < num_columns; c++) {
    for (int32_t r = 0; r < num_rows; r++) {
      reward[c][r] = reward_data[c][r];
    }
  }

  return reward;
}

void PlotReward(const torch::Tensor& reward) {
  /* Convert the tensor to a x- and y vector */
  std::vector<float> x;
  std::vector<float> y;
  for (int32_t i = 0; i < reward.size(0); i++) {
    x.push_back(i);
    y.push_back(reward[i].item<float>());
  }

  /* Plot the data. */
  matplotlibcpp::plot(x, y, "-k");

  matplotlibcpp::grid(true);

  matplotlibcpp::title("Mean Reward per Episode");
  matplotlibcpp::xlabel("Episode");
  matplotlibcpp::ylabel("Mean Reward");

  matplotlibcpp::pause(0.1);
}

} // namespace collective_robot_behaviour
} // namespace centralised_ai