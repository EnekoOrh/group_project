
import numpy as np

def rastrigin(x):
    """
    Rastrigin function.
    Global minimum at f(0, ..., 0) = 0.
    Bounds: usually [-5.12, 5.12]
    """
    x = np.array(x)
    A = 10
    d = len(x)
    return A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """
    Rosenbrock function.
    Global minimum at f(1, ..., 1) = 0.
    Bounds: usually [-5, 10] or [-2.048, 2.048]
    """
    x = np.array(x)
    t1 = 100 * (x[1:] - x[:-1]**2)**2
    t2 = (1 - x[:-1])**2
    return np.sum(t1 + t2)

def constrained_rosenbrock(x):
    """
    Rosenbrock function with a Unit Disk constraint: x^2 + y^2 <= 1.0.
    
    ACADEMIC NOTE:
    The unconstrained global minimum of Rosenbrock is at (1, 1).
    The distance of (1, 1) from origin is sqrt(2) approx 1.414.
    By setting the constraint to radius 1.0, we EXCLUDE the global minimum.
    This forces the optimizer to find the best compromise solution on the boundary.
    
    Returns: (objective_value, constraint_violation)
    Constraint violation is > 0 if constraint is violated.
    """
    x = np.array(x)
    obj = rosenbrock(x)
    
    # Constraint: sum(x^2) <= 1.0
    # Violation = max(0, sum(x^2) - 1.0)
    # Square root form: max(0, norm(x) - 1.0) might be smoother, 
    # but squared form is standard for penalty methods.
    violation = np.maximum(0, np.sum(x**2) - 1.0)
    
    return obj, violation

def penalty_function(x, func, penalty_factor=1000):
    """
    Generic wrapper for handling constraints via penalty.
    If 'func' returns a tuple, it's treated as (value, constraint_violation).
    """
    result = func(x)
    if isinstance(result, tuple):
        val, violation = result
        return val + penalty_factor * (violation ** 2)
    return result

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
