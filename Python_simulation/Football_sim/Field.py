import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import random
import math

from EnvFootball import FootballEnv
from Player_class import Player
from Ai_model import FootballNN  # Import AI model
import EnvFootball
#from EnvFootball import PolicyNetwork
#from EnvFootball import ValueNetwork

from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
# Initialize field dimensions
FIELD_LENGTH = 120  # Length in arbitrary units
FIELD_WIDTH = 80    # Width in arbitrary units
GOAL_WIDTH = 10     # Width of the goal area

# Define a class for Player
    
def reset_game():
    global players, ball_position

    # Create players with specific roles and positions
    players = [
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

    ball_position = np.array([FIELD_LENGTH / 2, FIELD_WIDTH / 2], dtype=float)

# Function to check if a goal is scored
def check_goal(ball_position):
    if (ball_position[0] <= GOAL_WIDTH and 30 <= ball_position[1] <= 50) or \
       (ball_position[0] >= FIELD_LENGTH - GOAL_WIDTH and 30 <= ball_position[1] <= 50):
        return True
    return False

# Function to draw the football field
def draw_field(ax):
    ax.clear()
    #ax.add_patch(patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, fill=False, color='green'))
    ax.plot([FIELD_LENGTH / 2, FIELD_LENGTH / 2], [0, FIELD_WIDTH], 'k--')
    ax.plot([10, 10], [30, 50], 'black') #create goals
    ax.plot([110, 110], [30, 50], 'black') #create goals
    center_circle = plt.Circle((FIELD_LENGTH / 2, FIELD_WIDTH / 2), 10, color='black', fill=False)
    ax.add_patch(center_circle)
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Field Length')
    ax.set_ylabel('Field Width')
    ax.grid(True)

## Function to update the animation
def update(frame):

    
    action = 1#Mappo_act()
    
    #action = agent.act(env.state,0,0)
    agent.models['agent_0']['policy'].compute(env.state)

    Train_network()

    player = env.players[0]
    env.execute_action(action, player)
    
    
    # Check if a goal is scored
    #if check_goal(ball_position):
    #    print("Goal! Resetting game.")
    #    reset_game()
        

    # Draw the field
    draw_field(ax)
    
    # Plot players
    for player in env.players:
        if player.team == 1:
            ax.plot(player.position[0], player.position[1], 'bo', markersize=10)
            ax.text(player.position[0], player.position[1] + 2, f'Player {player.index + 1}', color='blue', fontsize=9)
        elif player.team == 2:
            ax.plot(player.position[0], player.position[1], 'go', markersize=10)
            ax.text(player.position[0], player.position[1] + 2, f'Player {player.index + 1}', color='green', fontsize=9)
    
    # Plot ball
    
    ax.plot(env.ball_position[0], env.ball_position[1], 'ro', markersize=8, label='Ball')
    ax.legend()

#Create network, policy and actor, envirionemnt
def Create_networks(env):
    models = {}
    
    for agent_name in env.possible_agents:
        models[agent_name] = {}
        models[agent_name]["policy"] = EnvFootball.PolicyNetwork(observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device,
             unnormalized_log_prob=True,
             reduction="sum",
             num_envs=env.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10) #input, output
    
        #One value network!
        models[agent_name]["value"] = EnvFootball.ValueNetwork(observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            clip_actions=False,
            num_envs=env.num_envs,
            num_layers=1,
            hidden_size=64,
            sequence_length=10)
    
    

    memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=env.device, replacement=False)
    cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
    cfg_agent["rollouts"] = 1000  # adjust rollout timesteps as needed
    cfg_agent["learning_rate"] = 0.0003  # typical learning rate for PPO
    cfg_agent["entropy_coefficient"] = 0.01  # encourage exploration

    # instantiate the agent
    agentMappo = MAPPO(possible_agents=env.possible_agents,
                models=models,
                memories=memory,  # only required during training
                cfg=cfg_agent,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_space,
                device=env.device,
                shared_observation_spaces=env.shared_observation_spaces)
    
    return agentMappo
    
#Train networks, loss function and global buffer
def Train_network():
    None

env = FootballEnv()

agent = Create_networks(env)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
draw_field(ax)

# Initialize the game
env.reset()

# Create the animation
ani = FuncAnimation(fig, update, frames=range(1000), repeat=True, interval=100)

# Display the final plot
plt.show()
