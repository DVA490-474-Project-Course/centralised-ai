
import numpy as np

class Player:
    def __init__(self, index, position, role, haveball, team):
        self.index = index
        self.position = np.array(position, dtype=float)
        self.role = role
        self.haveball = haveball
        self.team = team
        self.start_position = np.array(position, dtype=float)  # Save the starting position

    