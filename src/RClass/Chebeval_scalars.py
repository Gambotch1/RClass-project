import numpy as np

def chebeval_scalars(coef, pts, m, a=-1, b=1):
    """
    Chebyshev evaluation: The Chebyshev polynomial
    Σ(k=0 to m-1) c_k * T_k(x) - c_0/2, x ∈ pts
    
    Translated from Numerical Recipes, Third edition, Section 5.8, pp. 237.
    
    Parameters:
        coef: array-like, Chebyshev coefficients
        pts: array-like, points at which to evaluate the polynomial
        m: int, order of approximation
        a: float, lower bound of interval (default -1)
        b: float, upper bound of interval (default 1)
    
    Returns:
        array-like, evaluated polynomial values
    """
    # Convert inputs to numpy arrays if they aren't already
    pts = np.asarray(pts)
    coef = np.asarray(coef)
    
    # Input validation
    if np.max(pts) > b or np.min(pts) < a:
        raise ValueError('Numbers outside the segment [a,b]')
    
    if m > len(coef):
        raise ValueError('Approximation order is too high for the precomputed C')
    
    if m < 0:
        raise ValueError('Approximation order must be greater than 1')
    
    # Initialize arrays with same shape as pts
    d = np.zeros_like(pts)
    dd = np.zeros_like(pts)
    
    # Change of variable
    newpts = (2 * pts - a - b) / (b - a)
    y2 = 2 * newpts
    
    # Clenshaw's recurrence
    for j in range(m-1, 0, -1):
        sv = d.copy()  # Need to copy because of array reference
        d = y2 * d - dd + coef[j]
        dd = sv
    
    # Final computation
    fv = newpts * d - dd + 0.5 * coef[0]
    
    return fv