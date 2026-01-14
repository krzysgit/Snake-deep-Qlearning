import pygame

def increment(x, y, size):
    if x < size:
        x += 1
    elif x == size:
        x = 1
        y += 1
    return x, y

# Render cells for the snake game
def render_cells(snake_body, window, CELL, apple_position):
    # BODY
    for x, y in snake_body:
        x -= 1
        y -= 1
        pygame.draw.rect(
            window,
            (0, 255, 0),
            (x * CELL, y * CELL, CELL, CELL)
        )

    # HEAD
    x,y = snake_body[0]
    x -= 1
    y -= 1
    pygame.draw.rect(
        window,
        (131, 39, 222),
        (x * CELL, y * CELL, CELL, CELL)
    )
    # APPLE
    x, y = apple_position
    x -= 1
    y -= 1
    pygame.draw.rect(window,
                     (255, 0, 0),
                     (x * CELL, y * CELL, CELL, CELL))
    pygame.display.flip()
    pygame.event.pump()


# Breath-First-Search - algorithm for checking whether there is a path between start and end (avoiding 1s)
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

def get_area(center, radius, grid):
    cells = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            cell = (center[0] + x, center[1] + y)
            if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]:
                if grid[cell]==0:
                    cells.append("0")
                    continue
            cells.append("1")
    return cells