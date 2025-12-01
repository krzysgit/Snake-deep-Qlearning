from collections import deque
import random
import numpy as np

#logika generowania nowego jedzenia mogłaby być obsługiwana również przez strukture danych set
#która to jest średnio znacznie szybsza niż mapa zapisująca wszystkie pozycje węża


class SnakeGame:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.snake_position = None
        self.snake_body = deque()
        self.snake_direction = None

        self.new_round()

    def new_round(self):

        self.snake_position = np.zeros((self.grid_size, self.grid_size))
        self.snake_body.clear()
        x_start, y_start = random.randint(0,self.grid_size), random.randint(0,self.grid_size)
        self.snake_body.appendleft((x_start, y_start))
        self.snake_position[x_start, y_start] = 1  
        snake_direction = random.randint(0,3)
        self.snake_direction = np.zeros((4))
        self.snake_direction[snake_direction] = 1


    def step(self, action):
        pass
    def spawn_food(self):
        pass
    def get_state(self):
        pass
    def is_collision(self):
        x_head, y_head = self.snake_body[0]
        
        