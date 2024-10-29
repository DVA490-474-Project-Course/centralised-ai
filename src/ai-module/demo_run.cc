
#include <vector>
#include "network.h"
#include "mappo.h"

/*Configuration values*/
int max_timesteps = 100;
int steps = 0; /*move into mappo------------------------*/
int step_max = 5;
int batch_size = 10;
int amount_of_players_in_team = 6;
int input_size = 7; // Number of input features
int num_actions = 9;
int hidden_size = 5;

int main() {
  std::vector<centralised_ai::collective_robot_behaviour::Agents> models; /*Create Models class for each robot.*/
  centralised_ai::collective_robot_behaviour::CriticNetwork critic; /*Create global critic network*/

  /*Comment out if want to create new agents, otherwise load in saved models*/
  models = centralised_ai::collective_robot_behaviour::CreateAgents(amount_of_players_in_team);
  //Models = LoadAgents(amount_of_players_in_team,critic); //Load in the trained model

  /*Run Mappo Agent algorithm by Policy Models and critic network*/
  centralised_ai::collective_robot_behaviour::Mappo(models,critic);

  return 0;
}
