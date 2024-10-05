from .utils import SudokuCSP, \
                   assign, unassign, assignment_in_grid, \
                   select_unassigned_variable, order_domain_values


def backtracking_search(csp):
    assignment = {var : domain[0]
                  for var, domain in csp.domains.items()
                  if csp.assigned(var)}

    return backtrack(assignment, csp, len(csp.vars))

def backtrack(assignment, csp, vars):
    if len(assignment) == vars:
        return assignment

    var = select_unassigned_variable(assignment, csp)

    for value in order_domain_values(assignment, csp, var):
        assign(assignment, csp, var, value)

        result = backtrack(assignment, csp, vars)
        if result:
            return result

        unassign(assignment, csp, var)

    return None

def solver_csp(csp):
    csp.ac_3()

    assg = backtracking_search(csp)
    grid = assignment_in_grid(assg)

    return grid

def solver_backtracking_for_csp(sudoku_grid):
    return solver_csp(SudokuCSP(sudoku_grid))
