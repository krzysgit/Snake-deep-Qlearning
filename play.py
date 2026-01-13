from environment import SnakeEnv
import pygame

snakeEnv = SnakeEnv(render_mode="human", size=10, cell_size=40)
snakeEnv.render()

clock = pygame.time.Clock()

actionReverseDict = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
}

keyMoveDict = {
    pygame.K_RIGHT: 2,
    pygame.K_UP: 1,
    pygame.K_LEFT: 0
}


running = True
while running:
    next_action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key in keyMoveDict:
                next_action = keyMoveDict[event.key]

    if next_action is not None:
        obs, reward, terminated, truncated, info = snakeEnv.step(actionReverseDict[next_action])
        snakeEnv.render()
        pending_action = None
        if terminated or truncated:
            print("INFO: Game over")
            running = False

    clock.tick(60)