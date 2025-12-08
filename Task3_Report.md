# Task 3 Group Report: Optimization Techniques

**Date:** 2025-12-08

## 1. Introduction
This report investigates and compares two optimization techniques: **Particle Swarm Optimization (PSO)** and **Simulated Annealing (SA)**. We evaluate their performance on three optimization problems:
1.  **Rastrigin Function** (Unconstrained)
2.  **Rosenbrock Function** (Unconstrained)
3.  **Constrained Rosenbrock Function** (Constrained to $x^2 + y^2 \le 2$)

## 2. Methodology
### 2.1 Algorithms
*   **Simulated Annealing (SA)**: implemented with a geometric cooling schedule (`temp *= 0.95`).
*   **Particle Swarm Optimization (PSO)**: implemented with standard velocity update equations. Inertia `w=0.5`.
*   **Constraints**: Handled using a quadratic penalty function: $F(x) = f(x) + \lambda \cdot (\max(0, g(x)))^2$.

### 2.2 Problems
*   **Rastrigin**: Highly multimodal, testing global search capability.
*   **Rosenbrock**: "Banana function", testing convergence in narrow valleys.
*   **Constrained Rosenbrock**: Tests ability to adhere to boundaries.

### 2.3 Experimental Setup
*   Evaluated over **10 runs** per algorithm/problem.
*   Max iterations: 500.
*   Metrics: Mean Best Objective Value, Standard Deviation, Execution Time, and **Number of Function Evaluations**.

## 3. Results

### 3.1 Quantitative Summary

| Problem | Algorithm | Mean Best Value | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|---|
| **Rastrigin** | SA | 2.07e+01 | 1.19e+01 | 0.0121 | 501.0 |
| | PSO | **9.95e-02** | 2.98e-01 | 0.4704 | 15,030.0 |
| **Rosenbrock** | SA | 2.33e-02 | 2.20e-02 | 0.0181 | 501.0 |
| | PSO | **0.00e+00** | 0.00e+00 | 0.5078 | 15,030.0 |
| **Constrained** | SA | 1.95e-02 | 2.68e-02 | 0.0184 | 501.0 |
| | PSO | **0.00e+00** | 0.00e+00 | 0.6305 | 15,030.0 |

**Note**: PSO requires significantly more function evaluations per iteration (equal to population size, $N=30$) compared to SA (1 per iteration). This explains the vast difference in run time and solution quality.

### 3.2 Convergence Analysis
*   **Rastrigin**: SA continuously got stuck in local minima. PSO's population-based approach allowed it to explore the landscape more effectively, bringing the mean best value down to near zero.
*   **Rosenbrock**: Both algorithms performed well, but PSO was superior in precision.
*   **Constrained**: PSO effectively stayed within the feasible region.

### 3.3 Penalty Function Sensitivity
We analyzed the effect of the penalty factor $\lambda$ on the *Constrained Rosenbrock* problem using PSO.

| Penalty Factor | Mean Best Value | Mean Constraint Violation |
|---|---|---|
| 1 | 9.86e-33 | 1.69e-15 |
| 10 | 0.00 | 8.88e-17 |
| 100 | 0.00 | 0.00 |
| 1000 | 9.86e-33 | 6.22e-16 |
| 10000 | 0.00 | 8.88e-16 |

**Observation**: For this specific problem, the constraint boundary is naturally respected or the optima lies safely within/on it such that even low penalty factors are sufficient. Increasing the penalty factor did not destabilize the search, indicating robustness in the PSO implementation.

## 4. Conclusion
Particle Swarm Optimization (PSO) demonstrated superior performance across all three tested problems. However, it is computationally more expensive, performing 30x more evaluations than Simulated Annealing for the same number of iterations. 

SA is extremely lightweight but failed to converge to the global minimum in the highly multimodal Rastrigin function. For constrained problems, the penalty function method proved effective for both algorithms, with PSO showing robustness across a wide range of penalty magnitudes.

## 5. Appendices
### 5.1 Convergence Plots
*   [Rastrigin Convergence](results/rastrigin_convergence.png)
*   [Rosenbrock Convergence](results/rosenbrock_convergence.png)
*   [Constrained Rosenbrock Convergence](results/constrained_rosenbrock_convergence.png)
*   [Penalty Sensitivity Analysis](results/penalty_sensitivity.png)
