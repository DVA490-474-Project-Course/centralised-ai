import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the neural network class
class FootballNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FootballNN, self).__init__()
        
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)          # Hidden layer
        self.fc3 = nn.Linear(64, output_size) # Output layer

    def forward(self, x):
        # Define the forward pass through the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on the output layer
        return x
    
    def predict_action(self):
        return 1
    
    def construct_state(players, ball_position):
        # Flatten the player positions and ball position into a single array
        state = np.concatenate([player.position for player in players] + [ball_position])
        return torch.tensor(state, dtype=torch.float32)

    # Example of selecting an action based on the NN output
    def select_action(state, model):
        # Forward pass through the network
        with torch.no_grad():
            action_logits = model(state)
        
        # Choose the action with the highest score (logit)
        action = torch.argmax(action_logits).item()
        
        return action
    

# Example AI class
class AIModel:
    def __init__(self):
        #self.action_space = ['pass', 'forward', 'reverse', 'right', 'left', 'shoot']
        self.action_space = [0,1,2,3,4,5,6]
        #self.action_space = ['left']
    def get_action(self, player,players, ball_position, attack):

        if player.team == 1:
            action = random.choice(self.action_space)

        elif player.team == 2:
            action = random.choice(self.action_space)
        
        
        # Construct state vector from players' positions and ball position
        state = self._construct_state(players, ball_position)
        
        return action
    
    def _construct_state(self, players, ball_position):
        # Create a 2D state array with player positions and ball position
        state = np.array([player.position for player in players] + [ball_position])
        return state

# Initialize AI model
ai_model = AIModel()
