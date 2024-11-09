import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate
from RClass.Chebeval_scalars import *
from scipy.optimize import linprog

def LpRat(f, z, n, m, T, Tn, Tm, DLB, DUB, vrb=False, uH=1e6, uL=1e-6):
    # Condition calculations
    f1 = f(T) + z
    f2 = f(T) - z
    cond1 = (f1[:, None] * Tm)
    cond2 = (f2[:, None] * Tm)

    # Construct constraint matrix A
    A_upper = np.hstack((Tn, -cond1, -np.ones((len(T), 1))))
    A_lower = np.hstack((-Tn, cond2, -np.ones((len(T), 1))))
    A_bounds_lower = np.hstack((np.zeros_like(Tn), -Tm, np.zeros((len(T), 1))))
    A_bounds_upper = np.hstack((np.zeros_like(Tn), Tm, np.zeros((len(T), 1))))

    A = np.vstack([A_upper, A_lower, A_bounds_lower, A_bounds_upper])

    # Construct constraint vector b
    b = np.concatenate([np.zeros(2 * Tn.shape[0]), -DLB * np.ones(len(T)), DUB * np.ones(len(T))])

    # Define bounds for each variable
    lb = [-np.inf] * n + [-np.inf] + [-np.inf] * (m - 1) + [uL]
    ub = [np.inf] * n + [np.inf] + [np.inf] * (m - 1) + [uH]
    bounds = [(low, high) for low, high in zip(lb, ub)]

    # Objective function: minimize the dummy variable
    obj = [0] * (n + m) + [1]

    # Define options for linprog
    # options = {
    #     "disp": vrb,
    #     "tol": 1e-9
    # }

    # Solve the linear program
    # result = linprog(c=obj, A_ub=A, b_ub=b, bounds=bounds, method='highs', options=options)
    result = linprog(c=obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # Check if a solution was found
    if result.success:
        x = result.x
        p = x[:n]
        q = x[n:n+m]
        u = x[n+m]
        exitflag = result.status
    else:
        p = np.zeros(n)
        q = np.zeros(m)
        u = np.inf
        exitflag = result.status

    return p, q, u, exitflag


def checkVal(f, z, n, m, T, Tn, Tm, DLB, DUB, vrb, uH, uL):
    _, _, u, _ = LpRat(f, z, n, m, T, Tn, Tm, DLB, DUB, vrb, uH, uL)
    return u <= 1e-15

def RationalMinMaxOpt(f, n, m, pts, LB, UB=None, a=-1, b=1, prc=10^-15, vrb=0, *args, **kwargs):
    """
    Calculating the uniform best rationl approx of type (n, m)
    via optimization with deviation precision 'eps'.
    
    Parameters:
        f :the function to be approximated. A function handler.
        n,m (int): the rational approx parameters = maximum degree (numer.,deno.)
        pts (int) : discretization points
        LB (int)  : lower bound on the denominator (away from zero)
        UB (int)  : upper bound on the denominator
	    prc (float) : precision of the bisection (maximum deviation accuracy)
        vrb : flag for verbose run
        
    Returns:
        p,q (int): the rational approx coefficients
        z ()the maximal deviation
    """
 
    # Upper bound on denominator (not necessary in general)
    if UB is None:
        UB = 1000 * LB

    # "Chebyshev" Vandermonde matrix for Tn
    I = np.eye(n)
    I[0] = 2
    Tn = np.zeros((len(pts), n))
    for deg in range(n):
        Tn[:, deg] = chebeval_scalars(I[deg, :], pts, deg + 1 , a, b)

    # "Chebyshev" Vandermonde matrix for Tm
    I = np.eye(m)
    I[0] = 2
    Tm = np.zeros((len(pts), m))
    for deg in range(m):
        Tm[:, deg] = chebeval_scalars(I[deg, :], pts, deg + 1, a, b)

    # Lower bound for deviation
    uL = 0

    # Change of variable for polynomial interpolation
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)

    # Upper bound using polynomial interpolation (degree n+m, Chebyshev points)
    numpts = n + m + 1
    int_pts = bpa + bma * np.cos(np.pi * (2 * np.arange(numpts, 0, -1) - 1) / (2 * numpts))
    uH = max(np.abs(f(pts) - barycentric_interpolate(int_pts, f(int_pts), pts)))

    # Bisection method for precision
    while (uH - uL) > prc:
        z = (uH + uL) / 2
        if checkVal(f, z, n, m, pts, Tn, Tm, LB, UB, vrb, np.inf, -np.inf):
            uH = z
        else:
            uL = z

    # Calculate optimal p, q using LpRat
    [p, q, zval,_] = LpRat(f, uH, n, m, pts, Tn, Tm, LB, UB, vrb, np.inf, -np.inf)
    return p, q, zval

if __name__ == "__main__":
    # Define the function f(x) to be approximated
    f = lambda x: np.abs(x - 0.1)

    # Parameters
    a = -0.5
    b = 0.5
    n = 4  # degree of the numerator
    m = 4  # degree of the denominator
    n_coefs = n+1;  # number of coeffs - numerator
    m_coefs = m+1;  # number of coeffs - denominator
    pts = np.linspace(-0.5, 0.5, 50)  # discretization points
    LB = 0.1  # lower bound on the denominator (away from zero)
    UB = 50  # upper bound on the denominator
    prc = 1e-14  # precision of the bisection (maximum deviation accuracy)
    vrb = 0  # flag for verbose run

    # Call the RationalMinMaxOpt function
    [p, q, _] = RationalMinMaxOpt(f, n_coefs, m_coefs, pts, LB, UB, a, b, prc=prc, vrb=vrb)

    # Print the results
    print("p (numerator coefficients):", p)
    print("q (denominator coefficients):", q)

    p[0] = 2*p[0]
    q[0] = 2*q[0]
    Tp   = chebeval_scalars(p, pts ,n_coefs, a, b)
    Tq   = chebeval_scalars(q, pts ,m_coefs, a, b)
    rat_app = Tp/ Tq

    # Generate data for the plots
    x_vals = np.linspace(-0.5, 0.5, 200)  # Points for plotting
    f_vals = f(x_vals)               # Original function values

    # Plot the original function and its rational approximation
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, f_vals, label="Original Function $f(x) = |x|$", color="blue")
    plt.plot(pts, rat_app, label="Rational Approximation", color="red", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Comparison of Original Function and Rational Approximation")
    plt.grid(True)
    plt.show()