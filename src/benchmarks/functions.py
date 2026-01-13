
import numpy as np


def rastrigin(x, grad=False):
    """
    Rastrigin function.
    Global minimum at f(0, ..., 0) = 0.
    """
    x = np.array(x)
    A = 10
    d = len(x)
    val = A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    if grad:
        # Gradient: 2x + 2*pi*A*sin(2*pi*x)
        g = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
        return val, g
    return val

def rosenbrock(x, grad=False):
    """
    Rosenbrock function.
    Global minimum at f(1, ..., 1) = 0.
    """
    x = np.array(x)
    
    # Value
    t1 = 100 * (x[1:] - x[:-1]**2)**2
    t2 = (1 - x[:-1])**2
    val = np.sum(t1 + t2)
    
    if grad:
        d = len(x)
        g = np.zeros(d)
        
        # Vectorized gradient calculation
        # Middle terms are tricky, let's use explicit loop or careful indexing if generalized
        # dL/dx_i involves term i (x_i in quadratic) and term i-1 (x_i in next square)
        
        # Term i contribution: -2(1-x_i) - 400x_i(x_{i+1} - x_i^2)
        # Term i-1 contribution: 200(x_i - x_{i-1}^2)
        
        g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        g[1:] += 200 * (x[1:] - x[:-1]**2)
        
        return val, g
        
    return val

def constrained_rosenbrock(x):
    # For constrained optimization with simple penalties, we might not need gradients 
    # if we only use BFGS on unconstrained. But if we do, it's complex.
    # Task 4 emphasizes "investigate different techniques" - we will stick to unconstrained for BFGS first.
    x = np.array(x)
    obj = rosenbrock(x) # Call unconstrained version
    violation = np.maximum(0, np.sum(x**2) - 1.0)
    return obj, violation
    
# Helper for legacy calls (if any) or simple wrapper
def penalty_function(x, func, penalty_factor=1000):
   # ... (Implementation unchanged, just ensuring signature match)
   res = func(x)
   if isinstance(res, tuple):
       val, violation = res
       return val + penalty_factor * (violation ** 2)
   return res

# Dictionary to easily retrieve bounds and optimal info
PROBLEM_CONFIG = {
    "rastrigin": {
        "func": rastrigin,
        "bounds": [-5.12, 5.12],
        "dim": 2, 
        "optimal_val": 0.0,
        "is_constrained": False
    },
    "rosenbrock": {
        "func": rosenbrock,
        "bounds": [-2.048, 2.048],
        "dim": 2, 
        "optimal_val": 0.0,
        "is_constrained": False
    },
    "constrained_rosenbrock": {
        "func": constrained_rosenbrock,
        "bounds": [-1.5, 1.5], # Confined tightly around the unit disk (radius 1)
        "dim": 2,
        "optimal_val": None, # Non-trivial to calculate analytically, somewhere on the boundary
        "is_constrained": True
    }
}
