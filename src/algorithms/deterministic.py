
import numpy as np
import time

class OptimizationAlgorithm:
    def __init__(self, func, bounds, max_iter=None, max_evals=None, seed=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.max_iter = max_iter if max_iter is not None else float('inf')
        self.max_evals = max_evals if max_evals is not None else float('inf')
        self.n_evals = 0

class BFGS(OptimizationAlgorithm):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton Method.
    
    Iterative update:
    x_{k+1} = x_k - alpha_k * B_k^{-1} * grad(f(x_k))
    
    Where B_k is the approximate Hessian.
    Inverse Hessian approximation H_k = B_k^{-1} is updated directly avoiding matrix inversion.
    """
    def __init__(self, func, bounds, dim, max_iter=100, max_evals=None, tol=1e-5, step_size=1.0, seed=None):
        super().__init__(func, bounds, max_iter, max_evals, seed)
        self.dim = dim
        self.tol = tol
        # Initial guess bounds (can be same as search space)
        if len(self.bounds) == 2 and isinstance(self.bounds[0], (int, float)):
             self.lower = np.full(dim, self.bounds[0])
             self.upper = np.full(dim, self.bounds[1])
        else:
             self.lower = np.array([b[0] for b in self.bounds])
             self.upper = np.array([b[1] for b in self.bounds])

    def _eval(self, x, grad=False):
        self.n_evals += 1
        return self.func(x, grad=grad)

    def solve(self):
        start_time = time.time()
        
        # Random initial point
        x_k = np.random.uniform(self.lower, self.upper, self.dim)
        
        # Initial evaluation
        f_k, g_k = self._eval(x_k, grad=True)
        
        # Initial Inverse Hessian Approximation (Identity Matrix)
        I = np.eye(self.dim)
        H_k = I
        
        history = [(self.n_evals, f_k)]
        trajectory = [x_k.copy()] # Store initial point
        
        iteration = 0
        while self.n_evals < self.max_evals and iteration < self.max_iter:
            iteration += 1
            
            # ... (convergence check) ...
            # Check convergence (Gradient Norm)
            gnorm = np.linalg.norm(g_k)
            if gnorm < self.tol:
                break
            
            # ... (rest of loop) ...
            
            # 1. Determine Search Direction: p_k = -H_k * g_k
            p_k = -np.dot(H_k, g_k)
            
            # 2. Line Search (Backtracking / Armijo)
            # Find alpha such that f(x + alpha*p) satisfies sufficient decrease
            alpha = 1.0
            c = 1e-4
            rho = 0.9
            
            while True and self.n_evals < self.max_evals:
                x_next = x_k + alpha * p_k
                f_next = self._eval(x_next) # No grad needed yet
                
                if f_next <= f_k + c * alpha * np.dot(g_k, p_k):
                    break # Armijo condition satisfied
                alpha *= rho
                
                if alpha < 1e-10: # Safety break
                    break
            
            # 3. Update
            x_next = x_k + alpha * p_k
            f_next, g_next = self._eval(x_next, grad=True)
            
            s_k = x_next - x_k
            y_k = g_next - g_k
            
            # Check division by zero (if no movement)
            rho_k = 1.0 / (np.dot(y_k, s_k) + 1e-10)
            
            # BFGS Update Formula for Inverse Hessian H_{k+1}
            A1 = I - rho_k * np.outer(s_k, y_k)
            A2 = I - rho_k * np.outer(y_k, s_k)
            H_k = np.dot(A1, np.dot(H_k, A2)) + rho_k * np.outer(s_k, s_k)
            
            # Move to next step
            x_k = x_next
            f_k = f_next
            g_k = g_next
            
            history.append((self.n_evals, f_k))
            trajectory.append(x_k.copy())
            
        end_time = time.time()
        return {
            "best_x": x_k,
            "best_val": f_k,
            "history": history,
            "trajectory": np.array(trajectory),
            "time": end_time - start_time,
            "n_evals": self.n_evals,
            "iterations": iteration
        }
