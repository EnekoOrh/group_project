# Task 4 Comparison Report: Deterministic vs Stochastic Optimization

**Module:** Applications of Computational Engineering Design Optimisation
**Themes:** BFGS, Quasi-Newton, Code-to-Maths, Log-Log Comparison
**Date**: 2026-01-13

---

## 1. Introduction
The objective of Task 4 is to investigate a **Deterministic, Gradient-Based** optimization technique and compare its performance against the **Stochastic** methods (Simulated Annealing and Particle Swarm Optimization) analyzed in Task 3. 

We selected the **BFGS (Broyden-Fletcher-Goldfarb-Shanno)** algorithm, a Quasi-Newton method. Unlike simple Gradient Descent, BFGS approximates the Hessian matrix (second-order curvature) to optimize step direction and size, offering super-linear convergence on smooth problems.

---

## 2. Mathematical Formulation & Implementation
A key requirement is to demonstrate the link between the mathematical definition of BFGS and our implementation.

### 2.1 The BFGS Update Formula
The core of the algorithm is the update of the **Inverse Hessian Approximation**, denoted as $H_k$. This allows the algorithm to learn the local curvature without the expensive operation of inverting a matrix.

Let:
*   $s_k = x_{k+1} - x_k$ (Displacement vector)
*   $y_k = \nabla f_{k+1} - \nabla f_k$ (Change in gradient)
*   $\rho_k = \frac{1}{y_k^T s_k}$ (Scalar scaling factor)

The update formula for $H_{k+1}$ is:
$$ H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T $$

### 2.2 Code Implementation Check
Our Python implementation in `src/algorithms/deterministic.py` manually calculates this update, avoiding "black box" library calls for the core logic.

```python
# Code snippet from src/algorithms/deterministic.py (Class BFGS)

# 1. Calculate change vectors
s_k = x_next - x_k
y_k = g_next - g_k

# 2. Calculate rho scalar
rho_k = 1.0 / (np.dot(y_k, s_k) + 1e-10)

# 3. BFGS Update Formula (Matches Equation Above)
# A1 = (I - rho * s * y^T)
A1 = I - rho_k * np.outer(s_k, y_k)
# A2 = (I - rho * y * s^T)
A2 = I - rho_k * np.outer(y_k, s_k)

# H_new = A1 * H * A2 + rho * s * s^T
H_k = np.dot(A1, np.dot(H_k, A2)) + rho_k * np.outer(s_k, s_k)
```

### 2.3 Termination Criteria (Crucial Difference)
Standard comparisons must account for different stopping mechanics:
*   **Stochastic (Task 3)**: Fixed Budget ($N=10,000$). We run until the budget is exhausted because global optima are hard to confirm.
*   **Deterministic (BFGS)**: Gradient Norm $\|\nabla f\| < \epsilon$. The algorithm **stops early** if it detects a flat region (minima).
    *   *Implication*: A shorter run time for BFGS often means "Success/Efficiency", not failure.

---

## 3. Experimental Setup
We evaluated the algorithms on two distinct landscapes:
1.  **Rosenbrock Function** (Unimodal Valley): A smooth, curved valley where gradient info is highly valuable.
2.  **Rastrigin Function** (Multimodal): A chaotic surface with many local minima, designed to trap gradient methods.

**Visualization Method**: We used **Log-Log plots** (Log evaluations vs Log Best Value) to fairly compare the rapid early convergence of BFGS against the long-tail refinement of PSO.

---

## 4. Results & Analysis

### 4.1 Case 1: Rosenbrock (The "Happy Path")
*   **BFGS Behavior**: Demonstrated **Super-Linear Convergence**. It typically finished in <200 evaluations with an error of $\approx 10^{-16}$ (Machine Precision).
*   **Comparison**:
    *   BFGS: $\approx 197$ evals, Error $10^{-16}$.
    *   PSO: $10,000$ evals, Error $0.0$ (eventually).
*   **Visual Analysis**: In `rosenbrock_log_log_comparison.png`, the BFGS line (Red) drops vertically like a cliff. It solves the problem before PSO/SA (Orange/Blue) have even completed their initial exploration phase.

### 4.2 Case 2: Rastrigin (The "Trap")
*   **BFGS Behavior**: **Premature Convergence**. It followed the gradient to the nearest basin and stopped when the gradient became zero.
    *   Result: Converged to values like $1.98$, $3.98$, or $18.9$ (Local Minima).
    *   Global Minimum is $0.0$.
*   **Comparison**:
    *   BFGS: Failed (Cost > 1.0).
    *   PSO: Succeeded (Cost $\approx 0.0$ or $0.2$).
*   **Visual Analysis**: In `rastrigin_log_log_comparison.png`, the BFGS line flattens out (horizontal line) at a high error value (top of the graph), while the PSO line continues a steady downward trend across the logarithmic x-axis.

### 4.3 Advanced Visualization: Search Trajectories
To further demonstrate the behavioral differences, we generated **2D Trajectory Contours** (`*_trajectory_contour.png`).
*   **BFGS (Red Solid)**: Appears as a short, direct line. This visualizes its "Gradient-Based Efficiency"—it takes few, large, calculated steps directly towards the minimum.
*   **PSO (Orange Dashed)**: Appears as disjointed jumps. This visualizes "Global Search"—the track represents the swarm's *best known position*, which effectively "teleports" across the landscape as different particles discover better regions.
*   **SA (Blue Dotted)**: Shows a wandering path, visualizing the "Stochastic Exploration" that gradually cools down.

---

## 5. Conclusion
The investigation confirms the "No Free Lunch" theorem in optimization:

1.  **Efficiency**: BFGS is orders of magnitude more efficient on smooth, unimodal problems. It is the tool of choice for local refinement.
2.  **Robustness**: BFGS is fragile on multimodal landscapes. It requires a Global Searcher (like PSO) to find the correct valley first.

**Recommendation**: A **Hybrid Strategy** is optimal. Use PSO to identify the promising region (Global Search), then initialize BFGS from that point to rapidly converge to the exact minimum (Local Refinement).
