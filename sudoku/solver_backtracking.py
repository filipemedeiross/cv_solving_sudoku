import numpy as np
from .utils import objective_grid,      \
                   first_available_pos, \
                   available_nums


def solver_backtracking(sudoku_grid):
    grid = sudoku_grid.copy()
    step = np.count_nonzero(grid == 0)

    moves = [None] * step
    actns = [None] * step

    size = 0
    while not objective_grid(grid):
        move = first_available_pos(grid)
        nums = available_nums(grid, move[1], move[0])

        if nums:
            grid[move] = nums.pop()
            moves[size] = move
            actns[size] = nums
            size += 1
        else:
            size -= 1

            while not actns[size]:
                # The instance has no solution
                if size == 0:
                    return None

                # Undoing each modification to generate the next successor
                grid[moves[size]] = 0
                size -= 1

            move = moves[size]
            num  = actns[size].pop()

            grid[move] = num
            size += 1

    return grid
