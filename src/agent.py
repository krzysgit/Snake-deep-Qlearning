from collections import deque
import numpy as np
import torch
import torch.nn as nn
import random
from collections import namedtuple
import copy

Transition = namedtuple(
'Transition', ('state', 'action', 'reward',
'next_state', 'done'))

class DQN:
    def __init__(self, env, discount_factor=0.95,
            epsilon_greedy=1.0, epsilon_min=0.01,
            epsilon_decay=0.995, learning_rate=1e-3,
            max_memory_size=2000, double_dqn=False):
        # Agent setup
        self.env = env
        self.df = discount_factor
        self.epsilon = epsilon_greedy       # Initial exploration probability
        self.epsilon_min = epsilon_min     # Minimum exploration probability
        self.epsilon_decay = epsilon_decay  # Decay of exploration probability
        self.lr = learning_rate
        self.memory = deque(maxlen=max_memory_size)
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.double_dqn = double_dqn
        # Torch NN setup
        self.online_model = nn.Sequential(nn.Linear(self.state_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.action_size))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(), self.lr)

        if double_dqn:
            self.target_model = copy.deepcopy(self.online_model)

            self.target_update_freq = 1000
            self.target_update_steps = 0

            for p in self.target_model.parameters():
                p.requires_grad = False
            self.target_model.eval()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration
            return np.random.choice(self.action_size)

        with torch.no_grad():
            # Exploitation
            output = self.online_model(torch.tensor(state,
                                               dtype=torch.float32))[0]
        return torch.argmax(output).item()

    def remember(self, transition):
        self.memory.append(transition)

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []

        for transition in batch_samples:
            s, a, r, next_s, done = transition
            if done:
                target = r
            else:
                target = self._q_predict(next_s, r)

            target_all = self.online_model(torch.tensor(s,
                                                 dtype=torch.float32))[0]
            target_all[a] = target

            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()

        self.optimizer.zero_grad()
        pred = self.online_model(torch.tensor(batch_states,
                                            dtype=torch.float32))
        loss = self.loss_fn(pred, torch.stack(batch_targets))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _q_predict(self, next_s, r):
        with torch.no_grad():
            pred_online = self.online_model(torch.tensor(next_s,
                                                         dtype=torch.float32))[0]
            if not self.double_dqn:
                target = r + self.df * pred_online.max()
            else:
                a_max = torch.argmax(pred_online)
                pred_target = self.target_model(torch.tensor(next_s,
                                                         dtype=torch.float32))[0]
                q_pred = pred_target[a_max].item()
                target = r + self.df*q_pred
                self._update_target()
        return target


    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)

    def _update_target(self):
        self.target_update_steps += 1
        if self.target_update_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())