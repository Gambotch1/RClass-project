# src/rclass/solver_gurobi.py

import numpy as np
from gurobipy import Model, GRB

def is_feasible_for_z_gurobi(p_Phi: np.ndarray,
                             q_Phi: np.ndarray,
                             f_c: np.ndarray,
                             z_current: float,
                             delta: float = 1e-6) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Check feasibility via Gurobi: enforce error bound and positivity of q(x) >= delta.
    Returns (feasible, p_coeffs, q_coeffs).
    """
    N, p_terms = p_Phi.shape
    _, q_terms = q_Phi.shape

    m = Model("rational_approx")
    m.setParam("OutputFlag", 0)

    p_vars = m.addVars(p_terms, lb=-GRB.INFINITY, name="p")
    q_vars = m.addVars(q_terms, lb=-GRB.INFINITY, name="q")

    # Normalize denominator
    m.addConstr(q_vars[0] == delta)

    for i in range(N):
        p_xi = sum(p_Phi[i, j] * p_vars[j] for j in range(p_terms))
        q_xi = sum(q_Phi[i, k] * q_vars[k] for k in range(q_terms))
        m.addConstr(f_c[i] * q_xi - p_xi <= z_current * q_xi)
        m.addConstr(p_xi - f_c[i] * q_xi <= z_current * q_xi)
        m.addConstr(q_xi >= delta)

    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        p_coeffs = np.array([p_vars[j].X for j in range(p_terms)])
        q_coeffs = np.array([q_vars[k].X for k in range(q_terms)])
        return True, p_coeffs, q_coeffs
    else:
        return False, None, None
