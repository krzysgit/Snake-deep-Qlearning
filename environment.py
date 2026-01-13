import pygame
import random
import numpy as np
import gymnasium
from gymnasium import spaces
from collections import deque

def increment(x, y, size):
    if x < size:
        x += 1
    elif x == size:
        x = 1
        y += 1
    return x, y

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
        is_head_there = (self.snake_body[0] == (x,y))
        is_tail_there = (self.snake_body[-1] == (x, y))
        xy_danger = self.snake_positions[x, y]
        if is_head_there:
            return 1 == (xy_danger - is_head_there)
        elif is_tail_there:
            return False
        else:
            return xy_danger

    def step(self, action):
        #I assume that the action is encoded with left -> 0 forward -> 1 right -> 2
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
                    x_current, y_current = increment(x_current, y_current, self.grid_size)
            else:
                x_current, y_current = increment(x_current, y_current, self.grid_size)               

    def get_state(self):
        #state is described by [is_down, is_right, is_up, is_left,
        #                       danger_left?, danger_forward?, danger_right?,
        #                       food_up?, food_down?, food_right? food_left?]
        state_arr = np.zeros((11))
        state_arr[self.snake_direction] = 1
        state_arr[4:7] = [self.is_danger(*self.next_position((self.snake_direction - i + 1) % 4)) for i in range(3)]
        x_head,y_head = self.snake_body[0]
        x_food, y_food = self.x_food, self.y_food
        food_down = (y_head < y_food)
        food_up = (y_head > y_food)
        food_left = (x_head > x_food)
        food_right = (x_head < x_food)
        state_arr[7:11] = [food_up, food_down, food_right, food_left]
        return state_arr

    def is_collision(self):
        return self.is_danger(*self.snake_body[0])

class SnakeEnv(gymnasium.Env):
    metadata = {
        "render_modes": ["human", "none"],
        "fps": ["30"],
    }

    def __init__(self, render_mode, size=15, cell_size=40, max_steps=40):
        self.size = size
        self.CELL = cell_size
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
        #rendering configuration
        self.window = None
        self.clock = None
        self.window_size = self.CELL * self.size

        self.steps_since_apple = 0
        self.max_steps = max_steps

    
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.Engine.new_round()
        return self.Engine.get_state(), {}

    
    def step(self, action):
        #input shold be of the type [left, forward, right],
        #which we transform to 0, 1, 2 and feed to the engine
        action_dict = {
            (1,0,0): 0,
            (0,1,0): 1,
            (0,0,1): 2,
        }
        direction = action_dict[tuple(action)]
        has_eaten_apple = self.Engine.step(direction)
        observation = self.Engine.get_state()
        terminated = self.Engine.is_collision()
        if (terminated):
            reward = -10
        elif (has_eaten_apple):
            reward = 10
            self.steps_since_apple = 0
        else:
            reward = 0
        self.steps_since_apple += 1
        if(self.steps_since_apple >= self.max_steps):
            truncated = True
        else:
            truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    
    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode((self.size*self.CELL, self.size * self.CELL))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

        self.window.fill("black")

        for x, y in self.Engine.snake_body:
            x -= 1
            y -= 1
            pygame.draw.rect(
                self.window,
                (0, 255, 0),
                (x*self.CELL, y*self.CELL , self.CELL, self.CELL)
            )
        
        x, y = self.Engine.x_food, self.Engine.y_food
        x -= 1
        y -= 1

        pygame.draw.rect(self.window,
                         (255,0,0),
                         (x*self.CELL, y*self.CELL, self.CELL, self.CELL))
        pygame.display.flip()
        pygame.event.pump()
    
    
    def close(self):
        pygame.quit()

