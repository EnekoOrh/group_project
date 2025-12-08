
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
    Rosenbrock function with a disk constraint: x^2 + y^2 <= 2.
    Returns: (objective_value, constraint_violation)
    Constraint violation is > 0 if constraint is violated.
    """
    x = np.array(x)
    obj = rosenbrock(x)
    
    # Constraint: sum(x^2) <= 2
    # Violation = max(0, sum(x^2) - 2)
    violation = np.maximum(0, np.sum(x**2) - 2)
    
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
        "dim": 2, # Can technically be N-dim
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
        "bounds": [-2.0, 2.0], # Confined roughly to the constraint area for init
        "dim": 2,
        "optimal_val": 0.0, # Not strictly 0 unless (1,1) is inside. 1^2+1^2=2 which is <= 2. So yes, (1,1) is on boundary.
        "is_constrained": True
    }
}
