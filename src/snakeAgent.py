import torch

from environment import SnakeEnv
from agent import DQN
import numpy as np
from agent import Transition
# General settings

EPISODES = 200
batch_size = 32
init_replay_memory_size = 500

env =  SnakeEnv("human")
agent = DQN(env)
state = env.reset()
state = np.reshape(state, [1, agent.state_size])

for i in range(init_replay_memory_size):
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state = np.reshape(next_state, [1, agent.state_size])
    agent.remember(Transition(state, action, reward,
                              next_state, done))
    if done:
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
    else:
        state = next_state

total_moves, losses, snake_max_lengths, death_states, truncated_array = [], [], [], [], []

for e in range(EPISODES):
    state = env.reset()
    if e % 1 == 0:
        env.render()
    state = np.reshape(state, [1, agent.state_size])
    for i in range(500):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward,
                                  next_state, done))
        state = next_state
        if e % 1 == 0:
            env.render()
        if done:
            death_states.append(env.get_positions())
            truncated_array.append(truncated)
            total_moves.append(i)
            snake_max_lengths.append(info["apples"]+1)
            print(f'Episode: {e}/{EPISODES}, Moves: {i}, Snake length: {info["apples"]+1}, Truncated: {truncated}')
            np.savez("../performance/training_stats.npz",
                     total_moves=total_moves,
                     snake_max_lengths=snake_max_lengths,
                     truncated_array=truncated_array,
                     death_states=np.array(death_states, dtype=object))
            break
        loss = agent.replay(batch_size)
        losses.append(loss)

torch.save(agent.model.state_dict(), "dqn_1.pt")