
import numpy as np
import time

class OptimizationAlgorithm:
    def __init__(self, func, bounds, max_iter=1000, seed=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        if seed is not None:
            np.random.seed(seed)
        
        self.dim = len(bounds) if isinstance(bounds[0], (list, tuple, np.ndarray)) else 2 # Assuming 2D if single range given
        if isinstance(bounds[0], (int, float)):
             # Convert simple [min, max] to [[min, max], [min, max], ...]
             # But usually bounds are passed as a list of ranges or a single range.
             # Let's standardize in solve method or assume bounds is a single [min, max] applied to all dims based on problem config
             pass
        self.n_evals = 0 # Initialize counter

    def solve(self):
        raise NotImplementedError

class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, func, bounds, dim, max_iter=1000, temp_init=100, cooling_rate=0.95, step_size=0.1, penalty_factor=1000, seed=None):
        super().__init__(func, bounds, max_iter, seed)
        self.dim = dim
        self.temp_init = temp_init
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.penalty_factor = penalty_factor
        
        # Determine actual bounds for each dimension
        if len(self.bounds) == 2 and isinstance(self.bounds[0], (int, float)):
             self.lower = np.full(dim, self.bounds[0])
             self.upper = np.full(dim, self.bounds[1])
        else:
             # Assume bounds is list of [min, max] per dim
             self.lower = np.array([b[0] for b in self.bounds])
             self.upper = np.array([b[1] for b in self.bounds])

    def _eval(self, x):
        self.n_evals += 1
        res = self.func(x)
        if isinstance(res, tuple):
             val, violation = res
             return val + self.penalty_factor * (violation ** 2)
        return res

    def solve(self):
        start_time = time.time()
        
        # Random initial solution
        current_x = np.random.uniform(self.lower, self.upper, self.dim)
        current_val = self._eval(current_x)
        
        best_x = current_x.copy()
        best_val = current_val
        
        history = [best_val]
        temp = self.temp_init
        
        for i in range(self.max_iter):
            # Generate neighbor
            neighbor_x = current_x + np.random.uniform(-self.step_size, self.step_size, self.dim)
            # Clip to bounds
            neighbor_x = np.clip(neighbor_x, self.lower, self.upper)
            
            neighbor_val = self._eval(neighbor_x)
            
            # Acceptance prob
            delta = neighbor_val - current_val
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                current_x = neighbor_x
                current_val = neighbor_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x
            
            history.append(best_val)
            temp *= self.cooling_rate
            
        end_time = time.time()
        return {
            "best_x": best_x,
            "best_val": best_val,
            "history": history,
            "time": end_time - start_time,
            "n_evals": self.n_evals
        }

class ParticleSwarm(OptimizationAlgorithm):
    def __init__(self, func, bounds, dim, max_iter=1000, num_particles=30, w=0.5, c1=1.5, c2=1.5, penalty_factor=1000, seed=None):
        super().__init__(func, bounds, max_iter, seed)
        self.dim = dim
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.penalty_factor = penalty_factor

        if len(self.bounds) == 2 and isinstance(self.bounds[0], (int, float)):
             self.lower = np.full(dim, self.bounds[0])
             self.upper = np.full(dim, self.bounds[1])
        else:
             self.lower = np.array([b[0] for b in self.bounds])
             self.upper = np.array([b[1] for b in self.bounds])

    def _eval(self, x):
        self.n_evals += 1
        res = self.func(x)
        if isinstance(res, tuple):
             val, violation = res
             return val + self.penalty_factor * (violation ** 2)
        return res

    def solve(self):
        start_time = time.time()
        
        # Init particles
        particles_x = np.random.uniform(self.lower, self.upper, (self.num_particles, self.dim))
        particles_v = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        
        personal_best_x = particles_x.copy()
        personal_best_val = np.array([self._eval(x) for x in particles_x]) # n_evals incremented inside _eval
        
        global_best_idx = np.argmin(personal_best_val)
        global_best_x = personal_best_x[global_best_idx].copy()
        global_best_val = personal_best_val[global_best_idx]
        
        history = [global_best_val]
        
        for i in range(self.max_iter):
            for p in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                
                # Update velocity
                particles_v[p] = (self.w * particles_v[p] + 
                                  self.c1 * r1 * (personal_best_x[p] - particles_x[p]) + 
                                  self.c2 * r2 * (global_best_x - particles_x[p]))
                
                # Update position
                particles_x[p] += particles_v[p]
                particles_x[p] = np.clip(particles_x[p], self.lower, self.upper)
                
                # Evaluate
                val = self._eval(particles_x[p])
                
                if val < personal_best_val[p]:
                    personal_best_val[p] = val
                    personal_best_x[p] = particles_x[p]
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = particles_x[p]
            
            history.append(global_best_val)
            
        end_time = time.time()
        return {
            "best_x": global_best_x,
            "best_val": global_best_val,
            "history": history,
            "time": end_time - start_time,
            "n_evals": self.n_evals
        }
