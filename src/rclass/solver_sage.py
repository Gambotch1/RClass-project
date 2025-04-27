import numpy as np
from sage.all import *
from sage.numerical.mip import MIPSolverException, MixedIntegerLinearProgram

def is_feasible_for_z_sage(p_Phi: np.ndarray,
                           q_Phi: np.ndarray,
                           f_c: np.ndarray,
                           z_current: float) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Check feasibility of a rational classifier with error bound z_current
    via SageMath's MixedIntegerLinearProgram (GLPK).

    Returns (feasible, p_coeffs, q_coeffs).
    """
    N, p_terms = p_Phi.shape
    _, q_terms = q_Phi.shape

    lp = MixedIntegerLinearProgram(maximization=False, solver="GLPK")
    p_vars = lp.new_variable(real=True, name="p", indices=range(p_terms))
    q_vars = lp.new_variable(real=True, name="q", indices=range(q_terms))

    # Normalize denominator to avoid trivial zeros
    lp.add_constraint(q_vars[0] == 1)

    # Error-bound constraints for each sample
    for i in range(N):
        p_xi = sum(p_Phi[i, j] * p_vars[j] for j in range(p_terms))
        q_xi = sum(q_Phi[i, k] * q_vars[k] for k in range(q_terms))
        lp.add_constraint(f_c[i] * q_xi - p_xi <= z_current * q_xi)
        lp.add_constraint(p_xi - f_c[i] * q_xi <= z_current * q_xi)

    lp.set_objective(0)
    try:
        lp.solve()
        p_coeffs = np.array([lp.get_values(p_vars[j]) for j in range(p_terms)])
        q_coeffs = np.array([lp.get_values(q_vars[k]) for k in range(q_terms)])
        return True, p_coeffs, q_coeffs
    except MIPSolverException:
        return False, None, None
