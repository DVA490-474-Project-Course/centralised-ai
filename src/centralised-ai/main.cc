
#include <vector>
#include "network.h"
#include "MAPPO.h"

extern int amount_of_players_in_team;
extern int num_actions;


int main() {
    std::vector<Agents> Models; //Create Models class for each robot.
    CriticNetwork critic; //Create global critic network

    /*Comment out if want to create new agents, otherwise load in saved models*/
    Models = createAgents(amount_of_players_in_team);
    //Models = load_agents(amount_of_players_in_team,critic); //Load in the trained model

    /*Run MAPPO Agent algorithm by Policy Models and critic network*/
    MAPPO(Models,critic);



    return 0;
}
