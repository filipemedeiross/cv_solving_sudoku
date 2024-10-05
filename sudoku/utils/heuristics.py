# Implementing the functions that apply the heuristics


def select_unassigned_variable(assignment, csp):
    available_vars = [var
                      for var in csp.vars
                      if var not in assignment.keys()]

    return min(available_vars, key=lambda var: len(csp.domains[var]))

def order_domain_values(assignment, csp, var):
    lcv = lambda value: len([neighbor
                             for neighbor in csp.neighbors[var]
                             if neighbor not in assignment and value in csp.domains[neighbor]])

    return sorted(csp.domains[var], key=lcv)
