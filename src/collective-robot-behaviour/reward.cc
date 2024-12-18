//==============================================================================
// Author: Jacob Johansson
// Creation date: 2024-10-01
// Last modified: 2024-12-12 by Jacob Johansson
// Description: Source file for all code related to the reward functions.
// License: See LICENSE file for license details.
//==============================================================================

#include "torch/torch.h"
#include "reward.h"

namespace centralised_ai
{
namespace collective_robot_behaviour
{

torch::Tensor ComputeAngleToBallReward(const torch::Tensor & orientations, const torch::Tensor & positions, const torch::Tensor & ball_position)
{
	torch::Tensor angles_to_ball = torch::empty(orientations.size(0));
	torch::Tensor robots_to_ball = ball_position - positions;

	for (int32_t i = 0; i < orientations.size(0); i++)
	{
		torch::Tensor world_forward = torch::zeros(2);
		world_forward[0] = orientations[i].cos();
		world_forward[1] = orientations[i].sin();

		torch::Tensor robot_to_ball = torch::zeros(2);
		robot_to_ball[0] = robots_to_ball[0][i];
		robot_to_ball[1] = robots_to_ball[1][i];

		torch::Tensor robot_to_ball_normalized = robot_to_ball.div(robot_to_ball.norm());
		torch::Tensor ball_product = robot_to_ball_normalized.dot(world_forward);

		angles_to_ball[i] = ball_product;
	}

	return angles_to_ball;
}

torch::Tensor ComputeAverageDistanceReward(torch::Tensor & positions, float max_distance, float max_reward)
{
	/* Calculate the average position of all the positions. */
	torch::Tensor average_position = positions.mean(1, true);
	torch::Tensor distances = (positions - average_position).pow(2).sum(0); /* [num_agents]. */

	torch::Tensor rewards = (-1/pow(max_distance, 2)) * distances + 1;

	return torch::clamp(rewards, 0, 1) * max_reward;
}

torch::Tensor ComputeDistanceToBallReward(torch::Tensor & positions, torch::Tensor & ball_position, float reward)
{
	torch::Tensor distances = (positions - ball_position).pow(2).sum(0); /* [num_agents]. */
	return -torch::sqrt(distances) * reward;
}

torch::Tensor ComputeHaveBallReward(torch::Tensor & have_ball_flags, float reward)
{
	torch::Tensor rewards = torch::zeros(have_ball_flags.size(0));

	for (int32_t i = 0; i < have_ball_flags.size(0); i++)
	{
		if(have_ball_flags[i].item<int>() > 0)
		{
			rewards[i] += reward;
		}
		else
		{
			rewards[i] -= reward;
		}
	}
	
	return rewards;
}

}
}