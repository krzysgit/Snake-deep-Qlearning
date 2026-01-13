from collections import deque

import numpy as np
import torch
import torch.nn as nn
import random

class DQN:
    def __init__(self, env, discount_factor=0.95,
            epsilon_greedy=1.0, epsilon_min=0.01,
            epsilon_decay=0.995, learning_rate=1e-3,
            max_memory_size=2000):
        # Agent setup
        self.env = env
        self.df = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.memory = deque(maxlen=max_memory_size)
        self.action_size = env.action_space.n
        # Torch NN setup
        self.model = nn.Sequential(nn.Linear(env.observation_space.shape[0], 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.action_size))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration
            return np.random.choice(self.action_size)

        with torch.no_grad():
            # Exploitation
            output = self.model(torch.tensor(state,
                                               dtype=torch.float32))[0]
        return torch.argmax(output).item()

    def remember(self, transition):
        self.memory.append(transition)

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []

        for transition in batch_samples:
            s, a, r, next_s, done = transition
            with torch.no_grad():
                if done:
                    target = r
                else:
                    pred = self.model(torch.tensor(next_s,
                                                   dtype=torch.float32))[0]
                    target = r + self.df * pred.max()
            target_all = self.model(torch.tensor(s,
                                                 dtype=torch.float32))[0]
            target_all[a] = target

            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()

        self.optimizer.zero_grad()
        pred = self.model(torch.tensor(batch_states,
                                            dtype=torch.float32))
        loss = self.loss_fn(pred, torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)