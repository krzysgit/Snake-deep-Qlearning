# Snake deep-Q learning

#### Description of contents:
- *./src*:
- *environment.py* - contains the snake engine (```snakeGame```) and snake env build on top of gymnasium (```snakeEnv```)
- *environment_utils.py* - contains helper methods for rendering and calculating more complicated states values
- *play.py* - simple algorithm allowing you to play using key.left, key.up, key.right
- *agent.py* - deep q learning feedforward NN built with PyTorch
- *snakeAgent.py* - current implementation of the agent in snake game with the learning loop
- *performance_inspect.py* - allows for inspecting death_states, snake lengths etc. of past episodes
- *./test*:
- tests for utilities functions
- *./models*:
- saved parameters of trained NNs
- *./performance*:
- .npz files with metrics from past episodes
