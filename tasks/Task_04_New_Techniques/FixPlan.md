# Plan to Fix Task 4 Requirements

## Objective
Update the codebase to satisfy the two missing mandatory requirements from the assignment PDF:
1.  **Solve 3 Problems**: Add a constrained optimization problem.
2.  **Penalty Factors**: Experiment with various multiplying constants.

## Code Changes

### 1. Modify `tasks/Task_04_New_Techniques/run_experiments.py`

**A. Enable Constrained Problem**
*   Add `"constrained_rosenbrock"` to the `probs` list (currently only runs Rastrigin/Rosenbrock).
*   Add logic to detect if a problem is constrained.

**B. Implement Penalty Loop**
*   If the problem is constrained, do NOT run the standard loop once.
*   Instead, iterate through `penalty_factors = [1, 10, 100, 1000, 10000]`.
*   For each factor:
    *   Create a partial function or wrapper for the objective that includes the penalty.
    *   Run BFGS.
    *   Store: `Best Value` (Total Cost), `Constraint Violation` (if accessible), `Evals`.

**C. Output Logic**
*   Save a specific `penalty_analysis.csv` to document the effect of `r`.
*   Generate a plot: `penalty_convergence.png` showing Best Value vs Penalty Factor (or Evals).

## Documentation Updates

### 2. Update `ReportBackbone_Final.md`
*   Remove "Critical Gap" alerts.
*   Remove all emojis.
*   Update **Section 4.3** and **Section 5** to explicitly reference the new `penalty_analysis.csv` and figures.
