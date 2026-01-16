from environment import SnakeEnv
import pygame

snakeEnv = SnakeEnv()
snakeEnv.render()

clock = pygame.time.Clock()
clock.tick(60)

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
        obs, reward, terminated, truncated, info = snakeEnv.step(next_action)
        print(obs,"\n" ,reward,"\n", terminated,"\n", truncated,"\n", info)
        snakeEnv.render()
        pending_action = None
        if terminated or truncated:
            print("INFO: Game over")
            running = False

"""
Run this file if you want to play manually.
This hasn't been updated in a while so there are limited logging options.
"""