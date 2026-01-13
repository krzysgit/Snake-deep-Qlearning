import numpy as np
from environment import render_custom

def inspect_performance(path):
    data = np.load(path, allow_pickle=True)

    total_moves = data["total_moves"].tolist()
    snake_max_lengths = data["snake_max_lengths"].tolist()
    truncated_array = data["truncated_array"]
    death_states = data["death_states"].tolist()

    length = len(total_moves)
    print(f'Total episodes recorded: {length}')
    while True:
        x = int(input())
        if isinstance(x, int) and 0 <= int(x) < length:
            print(f'Episode: {x}, Moves: {total_moves[x]},'
                  f' Snake length: {snake_max_lengths[x]},'
                  f' Truncated: {truncated_array[x]}')
            render_custom(*death_states[x])

inspect_performance("../performance/training_stats.npz")

