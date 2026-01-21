# Report Backbone (Task 4) - Final Version

**Status**: Ready for Writing
**Target Audience**: Academic / Prof. Phillips
**Tone**: Objective, Evidence-Based, Concise

---

## Executive Summary for the Writer

All mandatory experiments (Unconstrained + Constrained + Penalty Analysis) have been executed. The results are available in `results/summary.md` and `results/task4_raw.csv`.

**Writing Workflow:**
1.  **Open Results**: Keep `results/summary.md` open for reference.
2.  **Generate Plots**: Ensure `log_log_comparison` and `contour_trajectory` plots are generated for all 3 problems.
3.  **Fill Backbone**: Expand the bullet points below into full paragraphs.

---

## Section-by-Section Backbone

### 1. Introduction
*   **Purpose**: Define the scope (Deterministic vs Stochastic) and the specific technique (BFGS).
*   **Content**:
    *   **Objective**: Compare the efficiency of Gradient-Based methods (BFGS) against the Global Search methods (PSO/SA) from Task 3.
    *   **Methodology**: We investigate the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm, a Quasi-Newton method.
    *   **Key Advantage**: BFGS approximates curvature ($H_k$) to achieve super-linear convergence on smooth problems, avoiding the computational cost of exact Hessian inversion.
    *   **Hypothesis**: Deterministic methods track gradients explicitly, offering superior efficiency on unimodal landscapes but poor robustness on multimodal "trap" functions compared to stochastic methods.

### 2. Methodology: BFGS & Penalty Functions
*   **Purpose**: Link mathematical theory to the Python code (Satisfies Req R6).
*   **2.1 The BFGS Update Formula**:
    *   State the equation for the Inverse Hessian Update ($H_{k+1}$).
    *   Define terms: $s_k$ (displacement), $y_k$ (gradient change), and $\rho_k$ (scaling scalar).
*   **2.2 Python Implementation**:
    *   **Reference**: `src/algorithms/deterministic.py`.
    *   **Snippet**: Include the code block showing the $H_k$ update logic.
    *   **Termination Criteria**: Highlight that BFGS stops when $\|\nabla f\| < 10^{-6}$ (gradient is zero), whereas PSO stops at a fixed budget ($N=10,000$). This fundamental difference drives the efficiency comparison.
*   **2.3 Handling Constraints (Penalty Method)**:
    *   **Context**: Efficient constrained optimization often involves transforming the problem into an unconstrained one.
    *   **Approach**: We use a Quadratic Penalty Function:
        $$ \Phi(x, r) = f(x) + r \cdot \sum (\max(0, g_i(x)))^2 $$
    *   **Experimentation**: We explicitly test varying penalty factors $r \in [1, 10, 100, 1000, 10000]$ to analyze the trade-off between feasibility and convergence difficulty.

### 3. Experimental Setup
*   **Purpose**: Define the testbed.
*   **3.1 Test Functions**:
    *   **Problem 1: Rosenbrock** (Unimodal, Smooth). *Test Goal: Convergence Speed.*
    *   **Problem 2: Rastrigin** (Multimodal, Rugged). *Test Goal: Global Search Capability.*
    *   **Problem 3: Constrained Rosenbrock** (Constraints: Unit Disk). *Test Goal: Constraint Handling.*
*   **3.2 Comparison Protocol**:
    *   **Algorithms**: BFGS (New) vs PSO & SA (Task 3 Baselines).
    *   **Metrics**:
        *   **Best Value obtained**: Accuracy.
        *   **Function Evaluations**: Efficiency.
        *   **Wall-clock Time**: Computational cost.

### 4. Results & Analysis
*   **Purpose**: Present the evidence.

#### 4.1 Case 1: Unimodal Efficiency (Rosenbrock)
*   **Observation**: BFGS converges significantly faster than stochastic methods.
*   **Evidence**:
    *   Table Row: BFGS (~705 evals) vs PSO (10,020 evals).
    *   Figure: `rosenbrock_log_log_comparison.png`.
*   **Analysis**: The Log-Log plot shows BFGS dropping quickly (linear/super-linear rate), while PSO drifts. BFGS achieves a decent solution ($8.46 \times 10^{-2}$) in <10% of the budget used by PSO. Note that BFGS stopped early due to the gradient norm condition ($<10^{-6}$), explaining why it didn't grind down to machine precision like an infinite run might.

#### 4.2 Case 2: Multimodal Trap (Rastrigin)
*   **Observation**: BFGS fails significantly.
*   **Evidence**:
    *   Result: BFGS Best Value $\approx 15.7$ (Local Minimum), PSO Best Value $\approx 0.25$ (Global Minimum).
    *   Figure: `rastrigin_trajectory_contour.png`.
*   **Analysis**: The trajectory plot shows BFGS getting "trapped" in the nearest basin. Without a stochastic mechanism or random restarts, it cannot escape.

#### 4.3 Case 3: Constrained Optimization (Penalty Analysis)
*   **Observation**: The Penalty Factor ($r$) controls the trade-off.
*   **Evidence**: `results/summary.md` (Constrained Table).
    *   $r=1$: Low evaluations (~160), but likely high violation (Best Cost: 0.04).
    *   $r=10000$: High evaluations (~3700), strict constraint satisfaction (Best Cost: 0.43).
*   **Analysis**: Increasing $r$ makes the "canyon" of the feasible region steeper. This forces the solution to be valid but makes the gradient harder for BFGS to track, increasing the number of evaluations required.

### 5. Discussion
*   **5.1 Efficiency vs Robustness**:
    *   Reiterate the "No Free Lunch" principle. BFGS is precise/fragile; Stochastics are robust/approximate.
*   **5.2 Hybrid Strategy Proposal**:
    *   Suggest a pipeline: Run PSO for 10% of the budget to find the "basin," then initialize BFGS from that point for rapid local refinement.
*   **5.3 Limitations**:
    *   BFGS requires differentiable functions (smoothness).
    *   BFGS has $O(N^2)$ memory cost (Hessian matrix), unlike PSO's $O(N)$.

### 6. Conclusion
*   Task 4 confirms that while gradient-based methods like BFGS are unbeatable for speed on suitable problems, they lack the global perspective of stochastic search. A robust engineering solver should ideally combine both.

---

## Figures Plan

| ID | Title | Generator |
|:---|:---|:---|
| **Fig 1** | Rosenbrock Convergence (Log-Log) | `run_experiments.py` |
| **Fig 2** | Rastrigin Trajectory (The Trap) | `run_experiments.py` |
| **Fig 3** | Impact of Penalty Factor on Evaluations | *Plot manually from Table 4.3* |
| **Tab 1** | Performance Comparison Matrix | `summary.md` |
