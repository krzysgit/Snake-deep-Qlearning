import pygame
import random
import numpy as np
import gymnasium
from gymnasium import spaces
from collections import deque
from environment_utils import increment
from environment_utils import render_cells
from environment_utils import get_area

class SnakeGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake_positions = np.ones((self.grid_size+2 , self.grid_size+2))
        self.snake_body = deque()
        self.snake_direction = 0
        self.x_food = -1
        self.y_food = -1

        self.new_round()

    def new_round(self):

        self.snake_positions = np.ones((self.grid_size+2, self.grid_size+2))
        self.snake_positions[1:self.grid_size+1,1:self.grid_size+1] = 0
        self.snake_body.clear()
        x_start, y_start = random.randint(1,self.grid_size), random.randint(1,self.grid_size)
        self.snake_body.appendleft((x_start, y_start))
        self.snake_positions[x_start, y_start] = 1
        #format of the direction 0: down, 1: right, 2: up, 3: left
        self.snake_direction = random.randint(0,3)
        self.spawn_food()

    def next_position(self, direction):
        x_head, y_head = self.snake_body[0]
        if direction % 2 == 0:
            y_head -= direction - 1
        elif direction % 2 == 1:
            x_head -= direction - 2
        return x_head, y_head
    

    def is_danger(self, x, y):
        w, h = self.snake_positions.shape
        if x <= 0 or x >= w-1 or y <= 0 or y >= h-1:
            return True
        is_tail_there = (self.snake_body[-1] == (x,y))
        is_head_there = (self.snake_body[0] == (x,y))
        if is_tail_there:
            return False
        elif is_head_there:
            return self.snake_positions[x,y] == 2
        else:
            return self.snake_positions[x,y] == 1

    def step(self, action):
        # I assume that the action is encoded with left -> 0 forward -> 1 right -> 2
        self.snake_direction = (self.snake_direction - action + 1) % 4
        x_head, y_head = self.next_position(self.snake_direction)
        if x_head == self.x_food and y_head == self.y_food:
            self.snake_body.appendleft((x_head , y_head))
            self.snake_positions[x_head, y_head] += 1
            self.spawn_food()
            return True
        else:
            x_tail, y_tail = self.snake_body.pop()
            self.snake_positions[x_tail, y_tail] -= 1  
            self.snake_body.appendleft((x_head , y_head))
            self.snake_positions[x_head, y_head] += 1
            return False

    def spawn_food(self):
        num_occupied = np.sum(self.snake_positions)
        where_apple = random.randint(1, self.grid_size**2 - num_occupied)

        zero_idx = 1
        x_current = 1
        y_current = 1
        while True:
            if self.snake_positions[x_current, y_current] == 0:
                if zero_idx == where_apple:
                    self.x_food = x_current
                    self.y_food = y_current
                    break
                else:
                    zero_idx += 1
                    x_current, y_current = increment(x_current, y_current, self.grid_size)
            else:
                x_current, y_current = increment(x_current, y_current, self.grid_size)               

    def get_state(self):
        state_arr = np.zeros(11)
        # Direction (4)
        state_arr[self.snake_direction] = 1
        # Food direction (4)
        x_head, y_head = self.snake_body[0]
        x_food, y_food = self.x_food, self.y_food
        food_down = (y_head < y_food)
        food_up = (y_head > y_food)
        food_left = (x_head > x_food)
        food_right = (x_head < x_food)
        state_arr[4:8] = [food_up, food_down, food_right, food_left]
        # Danger (3)
        state_arr[8:11] = [self.is_danger(*self.next_position((self.snake_direction - i + 1) % 4)) for i in range(3)]
        #state_arr[8:17] = get_area(self.next_position(self.snake_direction),1,self.snake_positions)
        return state_arr

    def is_collision(self):
        return self.is_danger(*self.snake_body[0])

    def get_length(self):
        return len(self.snake_body)

class SnakeEnv(gymnasium.Env):
    metadata = {
        "fps": ["30"]
    }

    def __init__(self, size=15, cell_size=40, max_steps=200):
        self.size = size
        self.CELL = cell_size

        self.Engine = SnakeGame(size)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(11)

        # Rendering configuration
        self.window = None
        self.clock = None

        # Additional metrics
        self.steps_since_apple = 0      # Step count since last apple
        self.max_steps = max_steps      # Max steps between eating apples
        self.apples_eaten = 0           # Total number of apples eaten

    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        # Additional metrics
        self.steps_since_apple = 0
        self.apples_eaten = 0

        self.Engine.new_round()
        return self.Engine.get_state()

    
    def step(self, direction):
        has_eaten_apple = self.Engine.step(direction)
        observation = self.Engine.get_state()
        terminated = self.Engine.is_collision()

        if has_eaten_apple:
            reward = 10
            self.steps_since_apple = 0
            self.apples_eaten += 1
        else:
            reward = 0

        if terminated:
            reward = -20

        self.steps_since_apple += 1

        if self.steps_since_apple >= self.max_steps:
            truncated = True    # True if "run out of time"
        else:
            truncated = False

        info = {
            "steps": self.steps_since_apple,
            "apples": self.apples_eaten
        }
        return observation, reward, terminated, truncated, info

    
    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode((self.size*self.CELL, self.size * self.CELL))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

        self.window.fill("black")

        render_cells(self.Engine.snake_body, self.window, self.CELL, (self.Engine.x_food, self.Engine.y_food))
    
    def close(self):
        pygame.quit()

    # Retrieves the current board state
    def get_positions(self):
        return self.Engine.snake_body.copy(), (self.Engine.x_food, self.Engine.y_food), self.CELL, self.size

# Renders a custom board position
def render_custom(snake_body, apple_position, CELL, size):
    window = pygame.display.set_mode((CELL*size, CELL*size))
    pygame.display.set_caption("Snake AI")
    window.fill("black")

    render_cells(snake_body, window, CELL, apple_position)