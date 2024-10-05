import numpy as np
from collections import deque
from .constants import  N
from .problem_formulation import square_loc,    \
                                 objective_grid


class SudokuCSP:
    def __init__(self, grid):
        self.domains     = self.generate_domains(grid)
        self.neighbors   = self.generate_neighbors()
        self.constraints = self.generate_constraints()
        self.cuts        = self.generate_cuts()

    def ac_3(self):
        queue = deque([(xi, xj)
                       for xi, xj in self.constraints
                       if self.assigned(xj)])

        while queue:
            xi, xj = queue.popleft()

            if self.revise(xi, xj):
                if self.assigned(xi):
                    queue.extend([(xk, xi)
                                  for xk in self.neighbors[xi] - {xj}])
                elif self.unfeasible(xi):
                    return False

        return True

    def revise(self, xi, xj):
        vj = self.domains[xj][0]

        for vi in self.domains[xi]:
            if vi == vj:
                self.domains[xi].remove(vi)
                return True

        return False

    def generate_domains(self, grid):
        return {(y, x) : self.get_domain(grid, y, x)
                for y in range(N)
                for x in range(N)}

    def generate_neighbors(self):
        return {pos : self.get_neighbors(*pos)
                for pos in self.vars}

    def generate_constraints(self):
        return [(v, n)
                for v in self.vars
                for n in self.neighbors[v]]

    def generate_cuts(self):
        return {v : []
                for v in self.vars}

    @staticmethod
    def get_domain(grid, y, x):
        return [grid[y, x]] if grid[y, x] else list(range(1, N + 1))

    @staticmethod
    def get_row(y, x):
        return [(y, i)
                for i in range(N)
                if i != x]

    @staticmethod
    def get_col(y, x):
        return [(j, x)
                for j in range(N)
                if j != y]

    @staticmethod
    def get_sqr(y, x):
        return [(j, i)
                for j in range(*square_loc(y))
                for i in range(*square_loc(x))
                if j != y and i != x]

    def get_neighbors(self, y, x):
        return set(self.get_row(y, x) +
                   self.get_col(y, x) +
                   self.get_sqr(y, x))

    def unfeasible(self, xi):
        return len(self.domains[xi]) == 0

    def assigned(self, xi):
        return len(self.domains[xi]) == 1

    @property
    def vars(self):
        return self.domains.keys()

    @property
    def won(self):
        grid = np.zeros((N, N), dtype='uint8')

        for pos, domain in self.domains.items():
            if len(domain) > 1:
                return False

            grid[pos] = domain[0]

        return objective_grid(grid)
