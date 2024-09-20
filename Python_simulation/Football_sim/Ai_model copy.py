# AI_model.py

import numpy as np
import random
# Example AI class
class AIModel:
    def __init__(self):
        self.action_space = ['pass', 'forward', 'reverse', 'right', 'left', 'shoot']
        pass

    def get_action(self, players, ball_position,Attack):
        actions = self.action_space
        # Random actions for now until AI
        action = random.choice(actions)
        return action
    

# Initialize AI model
ai_model = AIModel()