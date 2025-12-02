import pygame
import random
import numpy as np
import gymnasium
from gymnasium import spaces
from collections import deque

def increment(xy_list, size):
    #x = xy_list[0], y = xy_list[0]
    #całe zamieszanie z listą jest po to aby modyfikować zmienne, a nie ich kopie
    if xy_list[0] < size:
        xy_list[0] += 1
    elif xy_list[0] == size:
        xy_list[0] = 1
        xy_list[1] += 1

class SnakeGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake_positions = np.zeros((self.grid_size+2 , self.grid_size+2))
        self.snake_body = deque()
        self.snake_direction = 0
        self.x_food = -1
        self.y_food = -1

        self.new_round()

    def new_round(self):

        self.snake_positions = np.zeros((self.grid_size+2, self.grid_size+2))
        self.snake_body.clear()
        x_start, y_start = random.randint(1,self.grid_size), random.randint(1,self.grid_size)
        self.snake_body.appendleft((x_start, y_start))
        self.snake_positions[x_start, y_start] = 1  
        snake_direction = random.randint(0,3)
        #format of the direction 0: down, 1: left, 2: up, 3: right
        self.snake_direction = np.zeros((4))
        self.snake_direction[snake_direction] = 1
        self.spawn_food()

    def step(self, action):
        #I assume that the action is one hot encoded left -> 0 forward -> 1 right -> 2
        self.snake_direction = (self.snake_direction + action - 1) % 4
        x_head, y_head = self.snake_body[0]
        if self.snake_direction % 2 == 0:
            y_head += self.snake_direction -1
        elif self.snake_direction % 2 == 1:
            x_head += self.snake_direction - 2
        if x_head == self.x_food and y_head == self.y_food:
            self.snake_body.appendleft((x_head , y_head))
            self.snake_positions[x_head, y_head] += 1
            self.spawn_food()
        else:
            x_tail, y_tail = self.snake_body.pop()
            self.snake_positions[x_tail, y_tail] -= 1  
            self.snake_body.appendleft((x_head , y_head))
            self.snake_positions[x_head, y_head] += 1
                
    def spawn_food(self):
        num_ocupied = np.sum(self.snake_positions)
        where_apple = random.randint(1, self.grid_size**2 - num_ocupied)

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
                    increment([x_current, y_current], self.grid_size)
            else:
                increment([x_current, y_current], self.grid_size)               

    def get_state(self):
        pass
    def is_collision(self):
        x_head, y_head = self.snake_body[0]
        p_min, p_max = min(x_head, y_head), min(x_head, y_head)
        if self.snake_positions[x_head, y_head] == 2:
            return True
        elif p_min == 0 or p_max == self.grid_size +1:
            return True
        else:
            return False

class SnakeEnv(gymnasium.Env):
    metadata = {
        "render_modes": ["human", "none"],
        "fps": ["30"],
    }

    def __init__(self, render_mode, size):
        self.size = size
        self.render_mode = render_mode

        self.Engine = SnakeGame(size)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                'direction': spaces.MultiBinary(4),
                'danger': spaces.MultiBinary(3),
                'food': spaces.MultiBinary(4),
            }
        )

