import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import random
import math

from Ai_model import ai_model  # Import AI model
# Initialize field dimensions
FIELD_LENGTH = 120  # Length in arbitrary units
FIELD_WIDTH = 80    # Width in arbitrary units
GOAL_WIDTH = 10     # Width of the goal area

# Define a class for Player
class Player:
    def __init__(self, index, position, role, haveball, team):
        self.index = index
        self.position = np.array(position, dtype=float)
        self.role = role
        self.haveball = haveball
        self.team = team
        self.start_position = np.array(position, dtype=float)  # Save the starting position

    def move(self, direction,ball_position): 
        # Move to the starting position
        distance = direction- self.position

        if np.linalg.norm(distance) > 0:
            direction_to_target = distance / np.linalg.norm(distance)
            self.position += direction_to_target * 1
            self.position = np.clip(self.position, [0, 0], [FIELD_LENGTH-10, FIELD_WIDTH])
            self.position = np.clip(self.position, [10, 0], [FIELD_LENGTH-10, FIELD_WIDTH])
        
        
        #if touched ball    
        distance_to_ball = np.linalg.norm(self.position - ball_position)
        if distance_to_ball < 1: 
            direction = ball_position- self.position
            # Normalize the direction vector manually
            norm = np.linalg.norm(direction)
            if norm != 0:
                # If norm is zero, the player and ball are at the same position, so no movement needed
                direction = direction / norm
            
            # Move the ball along this direction vector to keep it close to the player
            ball_position += direction * 1
        ball_position = self.check_out_ball(ball_position)

    def pass_ball(self, ball_position, players):
        distance_to_ball = np.linalg.norm(self.position - ball_position)
        if distance_to_ball < 2:
            # Calculate distances to all players except the current player
            distances = [np.linalg.norm(np.array(player.position) - np.array(ball_position)) for player in players if player != self]
            
            # Find the index of the nearest player
            nearest_player_index = np.argmin(distances)
            
            # Get the nearest player (excluding self)
            nearest_player = players[nearest_player_index if self.index < nearest_player_index else nearest_player_index + 1]
            
            # Calculate the direction vector to the nearest player
            direction = np.array(nearest_player.position) - np.array(ball_position)
            distance_to_nearest_player = np.linalg.norm(direction)
            
            if distance_to_nearest_player > 0:
                # Normalize the direction vector
                direction /= distance_to_nearest_player
                
                # Move the ball towards the nearest player by a fixed step size
                ball_position += direction * 10  # Move the ball towards the target player
                
                # Ensure the ball stays within the field boundaries
                ball_position = np.clip(ball_position, [0, 0], [FIELD_LENGTH, FIELD_WIDTH])
            
        return ball_position

    def shoot(self, ball_position):
        distance_to_ball = np.linalg.norm(self.position - ball_position)
        goal_pos = [110,40]
        if distance_to_ball < 2:  # Check if the player is close enough to shoot
            # Calculate the vector pointing from the player to the ball's current direction
            distance = goal_pos - self.position

            # Calculate the norm (magnitude) of the direction vector
            norm = np.linalg.norm(distance)

            # Check if norm is zero to avoid division by zero
            if norm == 0:
                # If norm is zero, the player and ball are at the same position, so no movement needed
                return ball_position
            
            # Normalize the direction vector manually
            direction = distance / norm

            # Move the ball along this direction vector
            ball_position += direction * 10  # Scale to control shot power

            # Ensure the ball stays within the field boundaries
            ball_position = np.clip(ball_position, [0, 0], [FIELD_LENGTH, FIELD_WIDTH])
            
        return ball_position
          
    def check_out_ball(self, ball_position):
        if check_goal:
            return
        
        #Y-axis
        if ball_position[0] <= 10:
            ball_position[0] = 11 
        if ball_position[0] >= 110:
            ball_position[0] = 109 
        
        #X_axis
        if ball_position[1] <= 0:
            ball_position[1] = 1 
        if ball_position[1] >= 80:
            ball_position[1] = 79 
        return ball_position

    def check_who_has_ball(self,ball_position):
        distance_to_ball = math.dist(self.position,ball_position)
        if distance_to_ball < 3:  # Check if the player is close enough to shoot
            self.haveball = True
            print("Team: " +str(self.team) + " Player: " + str(self.index + 1) +" have ball!")
        else:
            self.haveball = False
      
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

# Function to execute action
def execute_action(action, player):
    if action == 'forward':
        direction = player.position + [10,0]
        player.move(direction,ball_position)

    elif action == 'reverse':
        direction = player.position + [-10,0]
        player.move(direction,ball_position)

    elif action == 'right':
        direction = player.position + [0,-10]
        player.move(direction,ball_position)

    elif action == 'left':
        direction = player.position + [0,10]
        player.move(direction,ball_position)

    elif action == 'pass':
        ball_position[:] = player.pass_ball(ball_position, players)

    elif action == 'shoot':
        ball_position[:] = player.shoot(ball_position)

    elif action == 'move_to_ball':
        player.move_to_ball(ball_position)

    elif action == 'go_to_position':
        player.go_to_position()

## Function to update the animation
def update(frame):
    # Check which player has the ball

    # Iterate through the strategy buffer and execute actions for each player
    for player in players:

        haveball = player.check_who_has_ball(ball_position)

        # Get the strategy buffer, which should contain actions for specific players
        action = ai_model.get_action(player, players, ball_position,haveball)

        # Execute the action for the corresponding player
        execute_action(action, player)
        
        # Check if a goal is scored
        if check_goal(ball_position):
            print("Goal! Resetting game.")
            reset_game()
            break  # Exit the loop if a goal is scored
    
    # Draw the field
    draw_field(ax)
    
    # Plot players
    for player in players:
        if player.team == 1:
            ax.plot(player.position[0], player.position[1], 'bo', markersize=10)
            ax.text(player.position[0], player.position[1] + 2, f'Player {player.index + 1}', color='blue', fontsize=9)
        elif player.team == 2:
            ax.plot(player.position[0], player.position[1], 'go', markersize=10)
            ax.text(player.position[0], player.position[1] + 2, f'Player {player.index + 1}', color='green', fontsize=9)
    
    # Plot ball
    
    ax.plot(ball_position[0], ball_position[1], 'ro', markersize=8, label='Ball')
    ax.legend()

#train AI_model
def Train_Model():
    None

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
draw_field(ax)

# Initialize the game
reset_game()

# Create the animation
ani = FuncAnimation(fig, update, frames=range(1000), repeat=True, interval=100)

# Display the final plot
plt.show()
