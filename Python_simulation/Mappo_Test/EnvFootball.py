import gymnasium as gymnasium
from gym import spaces
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import Model, DeterministicMixin
from skrl.models.torch import Model, MultiCategoricalMixin
from Player_class import Player


from typing import Union, List, Optional
import copy
from skrl.envs.wrappers.torch import Wrapper
from skrl.agents.torch import Agent
from skrl.trainers.torch import Trainer

# Constants
FIELD_LENGTH = 120  # Length in arbitrary units
FIELD_WIDTH = 80    # Width in arbitrary units
GOAL_WIDTH = 10     # Width of the goal area


class FootballEnv(gymnasium.Env):
    def __init__(self):
        #super(FootballEnv, self).__init__()

        super().__init__()

        self.field_width = FIELD_WIDTH
        self.field_length = FIELD_LENGTH
        self.num_envs = 1
        # Goal positions
        self.goal_positions = [(0, FIELD_WIDTH / 2), (FIELD_LENGTH, FIELD_WIDTH / 2)]
        # Number of players per team
        self.num_players_per_team = 6  # Example value
        self.possible_agents = [f"agent_{i}" for i in range(self.num_players_per_team)]  # Adjust accordingly
        # Action space: Each player can move (left, right, up, down), shoot, or pass
        self.action_space = spaces.Discrete(6)
        # Actions: 0 = stay, 1 = move left, 2 = move right, 3 = move up, 4 = move down, 5 = shoot, 6 = pass
        self.num_agents = self.num_players_per_team # Adjust accordingly
        # Observation space: Positions of all players and ball on the field
        # Each player has an (x, y) position, and the ball has an (x, y) position
        self.observation_spaces = {
            agent_name: spaces.Box(low=0, high=100, shape=(self.num_players_per_team * 2 * 2 + 2,), dtype=np.float32)
            for agent_name in self.possible_agents
        }  

        self.device = "cpu"
        self.memory ={agent_name: [] for agent_name in self.possible_agents}
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_players_per_team * 2 * 2 + 2,), dtype=np.float32)
        # Define shared observation space for all agents
        self.shared_observation_spaces = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_players_per_team * 6,),  # 6 inputs per agent
            dtype=np.float32
        )

        # Initialize the state
        self.state = np.zeros(self.num_players_per_team * 2 * 2 + 2)  # Player positions + ball position
        self.reset()

    def state(self):
        return self._get_state()

    def reset(self):
        # Reset player positions
        # Create players with specific roles and positions
        self.players = [
        Player(index=0, position=[15, 40], role='goalkeeper',haveball=0, team = 1),
        Player(index=1, position=[30, 25], role='defender',haveball=0, team = 1),
        Player(index=2, position=[30, 55], role='defender',haveball=0, team = 1),
        Player(index=3, position=[25, 40], role='defender',haveball=0, team = 1),
        Player(index=4, position=[50, 50], role='striker',haveball=0, team = 1),
        Player(index=5, position=[50, 30], role='striker',haveball=0, team = 1),
        
        Player(index=0, position=[105, 40], role='goalkeeper',haveball=0, team = 2),
        Player(index=1, position=[90, 25], role='defender',haveball=0, team = 2),
        Player(index=2, position=[90, 55], role='defender',haveball=0, team = 2),
        Player(index=3, position=[95, 40], role='defender',haveball=0, team = 2),
        Player(index=4, position=[70, 50], role='striker',haveball=0, team = 2),
        Player(index=5, position=[70, 30], role='striker',haveball=0, team = 2)
        ]
        

        # Reset ball position randomly
        self.ball_position = np.array([FIELD_LENGTH / 2, FIELD_WIDTH / 2], dtype=float)
   
        # Random player has the ball

        return self._get_state()
    
    def step(self,player, action):

        # Execute the action based on your custom action logic
        self.execute_action(action, player)

        # Check if the game is over
        done = self._check_done()

        # Calculate reward
        reward = self._get_reward()

        return self._get_state(), reward, done, {}

    def _check_done(self):
        if (self.ball_position[0] <= GOAL_WIDTH and 
            FIELD_WIDTH / 2 - 10 <= self.ball_position[1] <= FIELD_WIDTH / 2 + 10) or \
           (self.ball_position[0] >= FIELD_LENGTH - GOAL_WIDTH and 
            FIELD_WIDTH / 2 - 10 <= self.ball_position[1] <= FIELD_WIDTH / 2 + 10):
            return True
        return False

    def check_who_has_ball(self, ball_position):
        for player in self.players:
            index = -1
            distance_to_ball = np.linalg.norm(player.position - ball_position)
            if distance_to_ball < 3:
                player.haveball = True
                print(f"Team: {player.team} Player: {player.index + 1} has the ball!")
                index = player.index
            else:
                player.haveball = False
        return index

    def _get_reward(self):
        if self._check_done():
            return 1  # Reward for scoring a goal
        return 0

    def _get_state(self):
        player_positions_flat = np.concatenate([player.position for player in self.players])
        ballindex = self.check_who_has_ball(self.ball_position)
        ballindex = np.array([ballindex])
        state = np.concatenate([player_positions_flat, self.ball_position, ballindex])
        return state

    def render(self):
        pass

    def close(self):
        # Optional: Clean up any resources
        pass

    def execute_action(self, action, player):
        if action == 0:  # Move right
            player.move(player.position + [1, 0], self.ball_position)
        elif action == 1:  # Move left
            player.move(player.position + [-1, 0], self.ball_position)
        elif action == 2:  # Move up
            player.move(player.position + [0, 1], self.ball_position)
        elif action == 3:  # Move down
            player.move(player.position + [0, -1], self.ball_position)
        elif action == 4:  # Shoot
            player.shoot(self.ball_position)
        elif action == 5:  # Pass
            player.pass_ball(self.ball_position, self.players)

# define the model
class PolicyNetwork(MultiCategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=27, sequence_length=1):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = 1 #amount oto check on
        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]
        self.training = True
        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"rnn": [hidden_states]}
   
# define the model
class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=64, sequence_length=10):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size + self.num_actions, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def get_specification(self):
        # batch size (N) is the number of envs during rollout
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # critic models are only used during training
        rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
        # get the hidden states corresponding to the initial sequence
        sequence_index = 1 if role == "target_critic" else 0  # target networks act on the next state of the environment
        hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                hidden_states[:, (terminated[:,i1-1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_output = torch.cat(rnn_outputs, dim=1)
        # no need to reset the RNN state in the sequence
        else:
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)), {"rnn": [hidden_states]}

