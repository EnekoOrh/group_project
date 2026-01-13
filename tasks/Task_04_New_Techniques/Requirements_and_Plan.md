# Task 4: Gradient-Based Optimization - Requirements & Output Plan

## 1. Objective
Compare the performance of **Gradient-Based Optimization** methods against the **Stochastic Methods** (SA & PSO) used in Task 3.
**Deadline**: 23/01/2026

## 2. Required Outputs (Deliverables)

### A. New Algorithms Implementation (`src/algorithms/`)
-   [ ] **BFGS (Quasi-Newton)**: STRICTLY REQUIRED.
    -   *Implementation Strategy*: use `scipy.optimize.minimize(method='BFGS')` for robustness, or implement a basic version if "white-box" is preferred. (Plan: Implement from scratch for maximum academic credit/understanding, fall back to scipy if needed).
-   [ ] **Nelder-Mead (Optional)**: Mentioned as another example of non-stochastic methods.
-   *Note*: These must support the same `solve()` interface as SA/PSO for easy comparison.

### B. Modified Benchmarks (`src/benchmarks/`)
-   [ ] **Gradient/Hessian Definitions**: The current benchmarks (`Rastrigin`, `Rosenbrock`) need to return derivatives for the gradient methods to work.
    -   *Action*: Update `functions.py` to optionally return gradient vectors.

### C. Visualizations (Updated)
-   [ ] **Early Convergence Plots**:
    -   *Feedback*: "Most interesting behavior is early on."
    -   *Action*: Plots must show the first ~100-500 iterations clearly (log-x axis or zoomed inset).
-   [ ] **Trajectory Plots**:
    -   Overlay the path taken by Gradient Descent vs PSO on the contour map.

### D. The Report (`tasks/Task_04_New_Techniques/Report.md`)
Structure the report to address specific feedback:

#### Section 1: Methodology (BFGS)
-   **Mathematical Formulation**: detailed explanation of the BFGS update (Hessian approximation).
-   **Code Mapping (CRITICAL)**: "Explain how the code implements the mathematical formulation."
    -   *Action*: Use code snippets in the report and annotate them with the corresponding equation (e.g., $B_{k+1} = B_k + \dots$).

#### Section 2: Results & Comparison
-   **Efficiency**: BFGS should converge *much* faster (fewer evals) on unimodal/smooth functions (Rosenbrock).
-   **Robustness**: BFGS should fail (get stuck in local minima) on multimodal functions (Rastrigin).
-   *Table*: Compare `Mean Best Value`, `evals_to_converge`, and `wall_time`.

#### Section 3: Discussion (Addressing Feedback)
-   **No "Silver Bullet"**: Conclude that Gradient methods are superior for local refinement (exploitation) but poor for global search (exploration), whereas PSO/SA are opposite.
-   **Hybrid Idea**: Briefly mention that a hybrid approach (Global Search -> Local Refinement) would likely be the robust "best" strategy.

## 3. Workflow
1.  **Refactor Benchmarks**: Add gradient analytics to `Rastrigin` and `Rosenbrock`.
2.  **Implement GD**: Create `GradientDescent` class in `src/algorithms/deterministic.py`.
3.  **Run Experiments**:
    -   Run BFGS on Rastrigin (expect failure/local min).
    -   Run BFGS on Rosenbrock (expect fast, super-linear convergence).
4.  **Generate Plots**: Create the "Zoomed-in" convergence comparison.
5.  **Write Report**: Synthesize findings.
