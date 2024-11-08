//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-11-01 by Jacob Johansson
// Description: Source file for all code related to the world representation.
// License: See LICENSE file for license details.
//==============================================================================

#include <torch/torch.h>
#include "world.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor ComputeAverageDistanceReward(torch::Tensor & positions, float max_distance, float max_reward)
{
	/* Calculate the average position of all the positions. */
	torch::Tensor average_position = positions.mean(1, true);
	torch::Tensor distances = (positions - average_position).pow(2).sum(0); /* [num_agents]. */

	torch::Tensor rewards = (-1/pow(max_distance, 2)) * distances + 1; /* Linear function for calculating the reward. */

	return torch::clamp(rewards, 0, 1) * max_reward;
}

torch::Tensor ComputeDistanceToBallReward(torch::Tensor & positions, torch::Tensor & ball_position, float reward)
{
	torch::Tensor distances = (positions - ball_position).pow(2).sum(0); /* [num_agents]. */
	return -torch::sqrt(distances) * reward;
}

torch::Tensor ComputeHaveBallReward(torch::Tensor & have_ball_flags, float reward)
{
	torch::Tensor rewards = torch::empty(have_ball_flags.size(0));
	for (int32_t i = 0; i < have_ball_flags.size(0); i++)
	{
		int32_t flag = have_ball_flags[i].item<int>();
		if(flag > 0)
		{
			rewards[i] = reward;
		}
		else
		{
			rewards[i] = -reward;
		}
	}
	return rewards;
}

}
}