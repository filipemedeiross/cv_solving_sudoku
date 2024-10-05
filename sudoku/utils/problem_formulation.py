import numpy as np
from .constants import N,    \
                       V,    \
                       STEP, \
                       SQR


def flatten_position(pos, inv=False):
    if inv:
        return pos // N, pos % N  # (y, x)
    else:
        return pos % N, pos // N  # (x, y)

def square_loc(p):
    """
    This function delimits the square grid in which a certain position is located

    :param pos:
    x or y position of a specific location

    :return ([x, y]_initial, [x, y]_final):
    Tuple containing the initial and final coordinates of the square,
    on the axis specified by the parameter
    """
    return SQR[p]

def number_constraint(grid, x, y, num):
    return num not in grid[y]    and \
           num not in grid[:, x] and \
           num not in grid[slice(*square_loc(y)), slice(*square_loc(x))]

def satisfies_constraints(grid_slice):
    return np.array_equal(np.sort(grid_slice), V)

def check_rows(grid):
    for row in grid:
        if not satisfies_constraints(row):
            return False

    return True

def check_cols(grid):
    for col in grid.T:
        if not satisfies_constraints(col):
            return False

    return True

def check_sqrs(grid):
    for y in range(0, N, STEP):
        for x in range(0, N, STEP):
            sqr = grid[y : y + STEP, x : x + STEP]

            if not satisfies_constraints(sqr.flatten()):
                return False

    return True

def objective_grid(grid):
    return check_rows(grid) and \
           check_cols(grid) and \
           check_sqrs(grid)

def first_available_pos(grid):
    for y in range(N):
        for x in range(N):
            if not grid[y, x]:
                return y, x

    return None

def available_pos(grid):
    return [(x, y) for y, x in zip(*np.where(grid == 0))]

def available_nums(grid, x, y):
    return [num
            for num in range(1, N + 1)
            if number_constraint(grid, x, y, num)]
