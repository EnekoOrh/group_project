# Task 4 Report Backbone: Deterministic vs Stochastic Optimization

**Target Audience:** Professor Chris Phillips  
**Tone:** Academic, Objective, Evidence-Based  
**Deadline:** 23/01/2026  
**Word Limit:** Not specified (assume standard academic conciseness)

---

## 🚨 Critical Action Items (The "Gap List")

Before writing the full report, **you must complete the following experiments** to satisfy the mandatory assignment requirements (R1-R5). The current implementation is incomplete.

| ID | Gap Description | Requirement Source | Fix Action |
|:---|:---|:---|:---|
| **G1** | **Missing 3rd Problem** | "solve three optimisation problems" | Select a 3rd problem (e.g., *Ackley* or *Sphere*, or a constrained engineering toy problem). Add to `functions.py` and `run_experiments.py`. |
| **G2** | **Missing Constrained Optimization** | "covering both constrained and unconstrained" | Enable `constrained_rosenbrock` in `run_experiments.py`. Ensure BFGS uses the `penalty_function` wrapper. |
| **G3** | **Missing Penalty Experimentation** | "experiment with various multiplying constants" | Run BFGS on the constrained problem with different penalty factors (e.g., $r=[10, 100, 1000]$) and report impact on convergence/validity. |
| **G4** | **Run Count Consistency** | "compare performance" | Ensure BFGS run stats (mean/std) are compared to Task 3's stochastic stats over equivalent runs (currently 20 runs in script). |

---

## 📌 Requirements Coverage Matrix

| Req ID | Requirement Description | Report Section | Evidence Source | Status |
|:---|:---|:---|:---|:---|
| R1 | Investigate Deterministic Technique (BFGS) | Sec 3 (Methodology) | `src/algorithms/deterministic.py` | ✅ Ready |
| R2 | Compare vs Stochastic (Task 3) | Sec 5 (Results) | `tasks/Task_04.../results/` | ✅ Ready |
| R3 | Solve 3 Problems (Constrained + Unconstrained) | Sec 4 (Setup) | `run_experiments.py` | ❌ **Partially Missing** (Only 2 unconstrained) |
| R4 | Experiment with Penalty Constants | Sec 5.3 (Constrained) | `run_experiments.py` | ❌ **Missing** |
| R5 | Graphical Representations | Sec 5 (Figures) | `..._log_log_comparison.png` | ✅ Ready |
| R6 | Link Math to Code Implementation | Sec 3.2 | `deterministic.py` (BFGS Update) | ✅ Ready |

---

## 📝 Report Outline & Writing Guidance

### Abstract
*   **Drafting Note:** Write this last.
*   **Content:**
    *   **Context:** Task 4 focuses on deterministic gradient-based optimization (BFGS).
    *   **Methods:** Implemented BFGS (Quasi-Newton) from scratch and compared against PSO/SA (from Task 3).
    *   **Experiments:** Tested on Unimodal (Rosenbrock), Multimodal (Rastrigin), and [Insert 3rd Constrained Problem].
    *   **Key Result:** BFGS offers superior efficiency (super-linear convergence) on smooth unimodal landscapes but fails prematurely on multimodal functions without global restart mechanisms.
    *   **Constraint Result:** [Insert finding about penalty factors].

### 1. Introduction
*   **Purpose:** Define the scope of Task 4.
*   **Key Points:**
    *   Contrast Task 3 (Stochastic, Global Search) with Task 4 (Deterministic, Local Search).
    *   Introduce BFGS as the specific Quasi-Newton method selected.
    *   State the objective: Evaluate efficiency (evaluations to convergence) vs robustness (ability to avoid local optima).

### 2. Methodology: BFGS Implementation
*   **Purpose:** Satisfy **R6** (Link Math to Code).
*   **Use content from:** Current Draft `Report.md` Section 2.
*   **Subsections:**
    *   **2.1 Mathematical Formulation:** State the BFGS update formula for $H_{k+1}$.
    *   **2.2 Implementation Details:**
        *   **Crucial:** Cite `src/algorithms/deterministic.py`.
        *   Show the Python snippet that mirrors the math (already in draft).
        *   Explain the **Termination Criteria**: Gradient Norm $\|\nabla f\| < \epsilon$ vs Task 3's fixed budget. Explain *why* this matters for efficiency comparisons.

### 3. Experimental Setup
*   **Purpose:** Define the testbed.
*   **Subsections:**
    *   **3.1 Optimization Problems:**
        *   **Problem 1: Rosenbrock** Valleys/Unimodal. (Cite: `functions.py`).
        *   **Problem 2: Rastrigin** Multimodal/Trap. (Cite: `functions.py`).
        *   **Problem 3: [INSERT NAME]** (Constrained). Define the function and the constraints.
    *   **3.2 Comparison Framework:**
        *   **Algorithms:** BFGS (New), PSO (Baseline), SA (Baseline).
        *   **Metrics:** 
            *   Evaluations to Convergence (Primary efficiency metric).
            *   Success Rate (Did it find global optimum?).
            *   Wall-clock time (Secondary).
    *   **3.3 Tuning & Parameters:**
        *   Evaluation Budget: 10,000 (for stochastic).
        *   BFGS Tolerance: $1e-6$.
        *   Penalty Factors evaluated: $r \in \{...\}$ (for Requirement R4).

### 4. Results & Analysis: Unconstrained
*   **Purpose:** Compare performance on smooth vs multimodal terrain.
*   **Evidence:** `results/rosenbrock_log_log_comparison.png` and `rastrigin_log_log_comparison.png`.
*   **4.1 Rosenbrock (The "Happy Path"):**
    *   **Observation:** BFGS graph should show a vertical drop (extremely fast convergence).
    *   **Quantify:** "BFGS solved in ~200 evals vs PSO's 10,000."
    *   **Conclusion:** Gradient information provides "super-linear" speedup on smooth valleys.
*   **4.2 Rastrigin (The "Trap"):**
    *   **Observation:** BFGS converges to a non-zero value (local minimum) and stops.
    *   **Evidence:** `results/rastrigin_trajectory_contour.png`. Describe how BFGS path (Red) goes straight to the nearest basin.
    *   **Conclusion:** Determinstic methods are greedy; they lack the "energy" or "swarm" mechanics to escape local basins.

### 5. Results & Analysis: Constrained (THE GAP)
*   **Purpose:** Satisfy **R3 & R4**.
*   **Content to Generate:**
    *   **Scenario:** Solving Problem 3 (e.g., Constrained Rosenbrock).
    *   **Penalty Analysis:** Compare BFGS performance with low penalty ($r=1$) vs high penalty ($r=1000$).
    *   **Discussion:** Does a high penalty make the gradient too steep/unstable? Does a low penalty allow violation?
    *   **Table:** Report `(Function Value, Violation, Evals)` for different $r$.

### 6. Discussion
*   **Interpretation:**
    *   **Efficiency vs Robustness Trade-off:** BFGS is a "Sniper" (precise, needs sight), PSO is a "Shotgun" (broad, messy).
    *   **Hybrid Proposal:** Suggest using PSO for exploration (first 1000 evals) then BFGS for refinement (as suggested in draft).
*   **Limitations:**
    *   BFGS requires differentiable functions (unlike SA/PSO).
    *   Memory cost of $H_k$ matrix ( $O(N^2)$ ) vs PSO ( $O(N)$ ).

### 7. Conclusion
*   Summarize the main findings relative to the module goals.

---

## 📊 Figures and Tables Plan

| Type | Caption | Source Generator | Status |
|:---|:---|:---|:---|
| **Table 1** | Performance Comparison (Mean Best Val, Evals, Time) | `run_experiments.py` (CSV output) | ✅ Ready |
| **Fig 1** | Rosenbrock Log-Log Convergence (BFGS vs PSO vs SA) | `plot_log_log_comparison` | ✅ Ready |
| **Fig 2** | Rastrigin Trajectory Contour (Visualizing Local Optima Trap) | `plot_contour_trajectory` | ✅ Ready |
| **Fig 3** | [NEW] Constrained Convergence under varying Penalty Factors | **[Action G3]** | ❌ Missing |
| **Table 2** | [NEW] Penalty Analysis Results | **[Action G3]** | ❌ Missing |

