import numpy as np
from .constants import N


def assign(assignment, csp, var, value):
    assignment[var] = value

    # Making inferences
    for neighbor in csp.neighbors[var]:
        if value in csp.domains[neighbor]:
            csp.domains[neighbor].remove(value)
            csp.cuts[var].append((neighbor, value))

def unassign(assignment, csp, var):
    del assignment[var]

    # Unmaking inferences
    for neighbor, value in csp.cuts[var]:
        csp.domains[neighbor].append(value)

    csp.cuts[var].clear()

def assignment_in_grid(assignment):
    return np.array([[assignment[y, x]
                      for x in range(N)]
                      for y in range(N)], dtype='uint8')

def process_grid(grid):
    return [(j, i, grid[j, i])
            for j in range(N)
            for i in range(N)]
