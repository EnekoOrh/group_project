# Project Journey

## 2026-01-13: Task 4 Kickoff
- **Objective**: Implement BFGS ("Investigate non-stochastic methods") and compare with SA/PSO.
- **Feedback Integration**: 
    - Focusing on "early convergence" visualization.
    - Explicitly mapping code to math in the report.
    - Implementing analytic gradients for robustness.
- **Status**: 
    - [x] Restructuring complete.
    - [x] Gradients added to benchmarks.
    - [x] BFGS implemented from scratch (Class `BFGS` in `src/algorithms/deterministic.py`).
    - [x] Experiment script created (`Task 4`).
    - [x] Experiments run & Report written (`tasks/Task_04_New_Techniques/Report.md`).
- **Outcome**: BFGS demonstrated massive speedup on Rosenbrock (197 evals vs 2000) but failed on Rastrigin, as predicted. Code-to-math mapping included in report.
- **Key Insight (Visualization)**: The Log-Log plots revealed a distinct "early stopping" behavior for BFGS.
    -   *Efficiency*: BFGS stops automatically when gradient $\approx 0$ (~200 evals), whereas pure stochastic methods exhaust the full budget (10k evals).
    -   *Interpretation*: The "short line" for BFGS in plots represents efficiency (solved quickly), not premature failure (unless it stopped at a high cost value like in Rastrigin).
- **"Over The Top" Feature**: Implemented `plot_contour_trajectory.py` to visualize the spatial search path on filled contours. Successfully demonstrated BFGS "sprinting" vs PSO "teleporting".

## 2026-01-16: Unified Execution
- **Objective**: Ensure reproducibility and ease of use for the professor/user.
- **Action**: Created `run_project.py` to unify execution of Task 3 and Task 4.
    -   *Features*: Task selection (`--task 3|4|all`) and automated cleanup of `results/` folders.
    -   *Benefit*: A single command (`python run_project.py`) now regenerates all data and plots from scratch, guaranteeing "fresh" results.

## 2026-01-16: Refinement & Validation
- **Visualization Standardization**: Unifying plotting styles across Task 3 and 4.
    -   *Colors*: BFGS (Red), SA (Magenta/Pink), PSO (Orange).
    -   *Metrics*: Added final/mean objective values directly to plot legends.
- **Robustness Check**: Implemented `--random` seeding in `run_project.py` to verify data freshness.
- **Explicit Seeding**: Added `--seed <int>` argument to `run_project.py` for fully deterministic reproducibility.
- **BFGS Bounds**: Confirmed that BFGS exceeding search bounds is expected behavior for unconstrained optimization on valley-like functions (Rosenbrock).
