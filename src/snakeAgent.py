import torch
import pygame
from environment import SnakeEnv
from agent import DQN
import numpy as np
from agent import Transition

# General settings
episodes = 200      # Number of episodes
batch_size = 32    # Training batch for SGD
init_replay_memory_size = 500   # Initial memory size
render_rate = 1             # The rate of rendering games to watch

# Setup
env =  SnakeEnv()
agent = DQN(env, double_dqn=True)
state = env.reset()
state = np.reshape(state, [1, agent.state_size])

# Filling up replay memory
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

# Metric arrays
total_moves, losses, snake_max_lengths, death_states, truncated_array = [], [], [], [], []

# Episode loop
for e in range(episodes):
    state = env.reset()
    if e % render_rate == 0:
        env.render()
    state = np.reshape(state, [1, agent.state_size])
    i = 0
    while True:
        i += 1
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward,
                                  next_state, done))
        state = next_state
        if e % render_rate == 0:
            env.render()

        if done:
            # Updating metrics
            death_states.append(env.get_positions())
            truncated_array.append(truncated)
            total_moves.append(i)
            snake_max_lengths.append(info["apples"]+1)

            print(f'Episode: {e}/{episodes}, Moves: {i}, Snake length: {info["apples"]+1}, Truncated: {truncated}')

            if info["apples"]>20:
                pygame.time.delay(2000)  # 2000 ms
                pygame.event.pump()  # keeps OS event queue from freezing (optional but good)

            # Saving metrics
            np.savez("../performance/training_stats.npz",
                     total_moves=total_moves,
                     snake_max_lengths=snake_max_lengths,
                     truncated_array=truncated_array,
                     death_states=np.array(death_states, dtype=object),
                     losses=np.array(losses, dtype=object))

            # Save the model if it finished training
            torch.save(agent.online_model.state_dict(), "../models/dqn_2.pt")
            break
        loss = agent.replay(batch_size)
        losses.append(loss)
