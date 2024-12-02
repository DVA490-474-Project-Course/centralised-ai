/* config.h
*==============================================================================
 * Author: Viktor Eriksson
 * Creation date: 2024-11-30.
 * Last modified: 2024-11-30 by Viktor Eriksson
 * Description: Configuration values.
 * License: See LICENSE file for license details.
 *==============================================================================
 */

#ifndef CONFIG_H
#define CONFIG_H

/*Configuration values*/
const int max_timesteps = 201;
const int steps = 0; /*move into mappo------------------------*/
const int step_max = 0;
const int buffer_length = 2;
const int amount_of_players_in_team = 2;
const int batch_size = buffer_length * amount_of_players_in_team;
const int input_size = 9;
const int num_actions = 3;
const int hidden_size = 64;

#endif //CONFIG_H
