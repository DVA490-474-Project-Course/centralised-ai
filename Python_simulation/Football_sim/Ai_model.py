import numpy as np
import random

# Example AI class
class AIModel:
    def __init__(self):
        self.action_space = ['pass', 'forward', 'reverse', 'right', 'left', 'shoot']
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
