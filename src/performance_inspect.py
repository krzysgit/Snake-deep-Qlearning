import numpy as np
from environment import render_custom
import torch
import matplotlib.pyplot as plt

def inspect_performance(path):
    # Reads the file
    data = np.load(path, allow_pickle=True)

    total_moves = data["total_moves"].tolist()      # Number of moves
    snake_max_lengths = data["snake_max_lengths"].tolist()      # Snake length
    truncated_array = data["truncated_array"]           # Whether the episode ended because of truncated
    death_states = data["death_states"].tolist()        # Positions rigth before deaths

    length = len(total_moves)
    print(f'Total episodes recorded: {length}')
    while True:
        x = int(input())        # Choose episode number
        if isinstance(x, int) and 0 <= int(x) < length:
            # Logs metrics
            print(f'Episode: {x}, Moves: {total_moves[x]},'
                  f' Snake length: {snake_max_lengths[x]},'
                  f' Truncated: {truncated_array[x]}')
            # Renders a custom board positon
            render_custom(*death_states[x])



def plot_losses(path):
    # Reads the file
    data = np.load(path, allow_pickle=True)
    losses = data["losses"].tolist()

    loss_vals = [
        (l.detach().cpu().item() if torch.is_tensor(l) else float(l))
        for l in losses
    ]

    plt.plot(loss_vals)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.show()

# Read the standard training stats
#inspect_performance("../performance/training_stats.npz")
# Plot losses
plot_losses("../performance/training_stats.npz")

