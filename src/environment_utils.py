import pygame

def increment(x, y, size):
    if x < size:
        x += 1
    elif x == size:
        x = 1
        y += 1
    return x, y

def render_cells(snake_body, window, CELL, apple_position):
    for x, y in snake_body:
        x -= 1
        y -= 1
        pygame.draw.rect(
            window,
            (0, 255, 0),
            (x * CELL, y * CELL, CELL, CELL)
        )
    x,y = snake_body[0]
    x -= 1
    y -= 1
    pygame.draw.rect(
        window,
        (0, 150, 0),
        (x * CELL, y * CELL, CELL, CELL)
    )

    x, y = apple_position
    x -= 1
    y -= 1
    pygame.draw.rect(window,
                     (255, 0, 0),
                     (x * CELL, y * CELL, CELL, CELL))
    pygame.display.flip()
    pygame.event.pump()

# This bfs assumes grid borders with 1s
# Breath-First-Search
def bfs(grid, start, end):
    increments = [(1,0),(0,1),(0,-1),(-1,0)]
    wave = {start}
    reachable = {start}
    while True:
        new_wave = set()
        for cell in wave:
            for dx, dy in increments:
                new_cell = (cell[0] + dx, cell[1] + dy)

                x, y = new_cell
                if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                    continue

                if grid[new_cell]==0 and new_cell not in reachable: new_wave.add(new_cell)
        reachable.update(new_wave)
        wave = new_wave

        if end in reachable:
            return True
        elif not wave:
            return False
