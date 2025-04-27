# src/rclass/bisection.py

import numpy as np
from typing import Callable, Tuple
from rclass.solver_sage import is_feasible_for_z_sage
from rclass.solver_gurobi import is_feasible_for_z_gurobi

SolverFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[bool, np.ndarray, np.ndarray]]

def solve_rational_approx_bisection(
    p_Phi: np.ndarray,
    q_Phi: np.ndarray,
    f_c: np.ndarray,
    solver: SolverFn = is_feasible_for_z_gurobi,
    z_init_high: float = 10.0,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Use bisection on z to find the minimal feasible error bound.

    Parameters:
      p_Phi, q_Phi: feature matrices
      f_c: binary label vector
      solver: one of is_feasible_for_z_sage or is_feasible_for_z_gurobi
      z_init_high: starting upper bound for z
      tolerance: stopping threshold for (z_high - z_low)

    Returns:
      (p_coeffs, q_coeffs, z_approx) or (None, None, None) if no solution found.
    """
    z_low, z_high = 0.0, z_init_high
    best_p = best_q = None

    while (z_high - z_low) > tolerance:
        z_mid = 0.5 * (z_low + z_high)
        feasible, p_coeffs, q_coeffs = solver(p_Phi, q_Phi, f_c, z_mid)
        if feasible:
            z_high, best_p, best_q = z_mid, p_coeffs, q_coeffs
        else:
            z_low = z_mid

    if best_p is None:
        return None, None, None

    z_approx = 0.5 * (z_low + z_high)
    return best_p, best_q, z_approx


def solve_for_class(
    c: int,
    p_Phi: np.ndarray,
    q_Phi: np.ndarray,
    f_values: dict[int, np.ndarray],
    solver: SolverFn = is_feasible_for_z_gurobi
) -> Tuple[int, np.ndarray, np.ndarray, float]:
    """
    Wrapper to solve one-vs-rest rational approximation for a given class c.

    Returns: (c, p_coeffs, q_coeffs, z_approx).
    """
    f_c = f_values[c]
    p_coeffs, q_coeffs, z_approx = solve_rational_approx_bisection(
        p_Phi, q_Phi, f_c, solver=solver
    )
    return (c, p_coeffs, q_coeffs, z_approx)
