//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-10-29 by Jacob Johansson
// Description: Source file for all code related to the world representation.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include "world.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor compute_average_distance_reward(torch::Tensor positions, float max_distance, float max_reward)
{
	/* Calculate the average position of all the positions. */
	torch::Tensor average_position = positions.mean(1, true);
	torch::Tensor distances = (positions - average_position).pow(2).sum(0); /* [num_agents]. */

	torch::Tensor rewards = (-1/max_distance) * distances + 1; /* Linear function for calculating the reward. */

	return torch::clamp(rewards, 0, 1) * max_reward;
}

}
}