from unittest import TestCase
import numpy as np
from environment_utils import bfs


class Test(TestCase):
    def test_simple_path(self):
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        assert bfs(grid, (1, 1), (3, 3)) is True

    def test_start_equals_end(self):
        grid = np.ones((5, 5))
        grid[2, 2] = 0
        assert bfs(grid, (2, 2), (2, 2)) is True

    def test_no_free_space(self):
        grid = np.ones((5, 5))
        grid[1, 1] = 0
        assert bfs(grid, (1, 1), (3, 3)) is False

    def test_narrow_corridor(self):
        grid = np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1]
        ])
        assert bfs(grid, (1, 1), (4, 4)) is True

    def test_edge_start(self):
        grid = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1]
        ])
        assert bfs(grid, (1, 1), (2, 2)) is True

    def test_enclosed_target(self):
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        assert bfs(grid, (1, 1), (2, 2)) is False

    


