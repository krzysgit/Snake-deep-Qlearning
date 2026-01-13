import numpy as np
from environment import render_custom

def inspectPerformance(path):
    data = np.load(path, allow_pickle=True)

    total_moves = data["total_moves"].tolist()
    snake_max_lengths = data["snake_max_lengths"].tolist()
    death_states = data["death_states"].tolist()

    length = len(total_moves)
    print(f'Total episodes recorder: {length}')
    while True:
        x = int(input())
        if isinstance(x, int) and 0 <= int(x) < length:
            print(f'Episode: {x}, Moves: {total_moves[x]},'
                  f' Snake length: {snake_max_lengths[x]}')
            render_custom(*death_states[x])

inspectPerformance("performance/training_stats.npz")

