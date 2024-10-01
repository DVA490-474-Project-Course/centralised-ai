
import numpy as np
from EnvFootball import FootballEnv
from EnvFootball import PolicyNetwork
from EnvFootball import ValueNetwork
from Player_class import Player

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import torch
# import the agent and its default configuration
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import SequentialTrainer

FIELD_LENGTH = 120  # Length in arbitrary units
FIELD_WIDTH = 80    # Width in arbitrary units
GOAL_WIDTH = 10     # Width of the goal area

def setup_agent():
    envF = FootballEnv()
    print(type(envF)) 
    #envF = Wrapper(env)
    #print(type(env)) 

    # instantiate the agent's models
    models = {}
    for agent_name in envF.possible_agents:
        models[agent_name] = {}
        models[agent_name]["policy"] = PolicyNetwork(observation_space=envF.observation_space,
             action_space=envF.action_space,
             device=envF.device,
             unnormalized_log_prob=True,
             reduction="sum",
             num_envs=envF.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10) #input, output
        
        models[agent_name]["value"] = ValueNetwork(observation_space=envF.observation_space,
             action_space=envF.action_space,
             device=envF.device,
             clip_actions=False,
             num_envs=envF.num_envs,
             num_layers=1,
             hidden_size=64,
             sequence_length=10)
    
    # Set up agent configuration
    cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
    cfg_agent["rollouts"] = 1000  # adjust rollout timesteps as needed
    cfg_agent["learning_rate"] = 0.0003  # typical learning rate for PPO
    cfg_agent["entropy_coefficient"] = 0.01  # encourage exploration

    device = envF.device
    memory = RandomMemory(memory_size=20000, num_envs=envF.num_envs, device=device, replacement=False)

    # instantiate the agent
    agent = MAPPO(possible_agents=envF.possible_agents,
                models=models,
                memories=memory,  # only required during training
                cfg=cfg_agent,
                observation_spaces=envF.observation_spaces,
                action_spaces=envF.action_space,
                device=envF.device,
                shared_observation_spaces=envF.shared_observation_spaces)
    
    
    cfg_trainer = {"timesteps": 1000, "headless": False}
    trainer = SequentialTrainer(env = envF, agents=agent,cfg=cfg_trainer)
        
    # train the agent(s)
    trainer.train()


    return envF

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

def plot_windows():
    
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
    plt.pause(0.0001)  # Pause to make the update visible

def Animate_Plot(frame):
    #done = False
    #states = env._get_state()
    #while done != True:

        #ax.clear()
    actions = {}

    state = env._get_state()
    state = state.reshape(1, 1, 27)  # (N=1, L=1, Hin=2
    inputs = {
    "states": state ,    # The state of the environment
    "rnn" : torch.zeros(2,1,64), #fix so it reads from self
    "terminated": None # Previous actions if needed
    }
    for models in agents.models:
        #actions = env.action_space.sample()  # Take a random action
        actions, log_prob, outputs  = agents.policies[models].act(inputs)
        
        #next_states, rewards, done, info = env.step(1,0)
        
        #plot_windows()
    
    env.reset()


#--------------------------------------------------------------------------
env = setup_agent()

# create a sequential trainer


# evaluate the agent(s)
#trainer.eval()

#Create animation of the game
#fig, ax = plt.subplots(figsize=(10, 6))
env.reset()
#draw_field(ax)
# Create the animation

#while True:
Animate_Plot(0)

#ani = FuncAnimation(fig, Animate_Plot, frames=range(1000), repeat=True,interval=0.01)
# Display the final plot

plt.show()

