import numpy as np
from rclass.solver_sage import is_feasible_for_z_sage
from rclass.bisection import solve_rational_approx_bisection

X = np.random.rand(5, 2)
p_Phi = np.hstack([np.ones((5, 1)), X])
q_Phi = p_Phi
f_c = np.array([1, 0, 1, 0, 1])
solve_rational_approx_bisection(p_Phi, q_Phi, f_c,
                                solver=is_feasible_for_z_sage,
                                tolerance=1e-1)
print("Smoke test passed")