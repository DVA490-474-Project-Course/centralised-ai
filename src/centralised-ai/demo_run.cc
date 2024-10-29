
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
    std::vector<Agents> Models; /*Create Models class for each robot.*/
    CriticNetwork critic; /*Create global critic network*/

    /*Comment out if want to create new agents, otherwise load in saved models*/
    Models = createAgents(amount_of_players_in_team);
    //Models = load_agents(amount_of_players_in_team,critic); //Load in the trained model

    /*Run MAPPO Agent algorithm by Policy Models and critic network*/
    MAPPO(Models,critic);



    return 0;
}
