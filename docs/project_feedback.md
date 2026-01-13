# Project Feedback Log

## Task 3 Feedback (Received 2026-01-13)
**Source**: Professor / Markers

### Key Takeaways
1.  **No "Silver Bullet"**: Avoid generalizing that one algorithm is "best". Different algorithms suit different problem classes.
2.  **Visualization**:
    -   *Critique*: Convergence plots show too much "flat line" where algorithms successfully settled.
    -   *Action*: Zoom in on the early iterations (e.g., first 10-20% of evals) to show the interesting convergence behavior.
3.  **Termination**:
    -   *Observation*: Heuristics lack clear gradient-based termination criteria (norm of gradient = 0).
    -   *Action*: Discuss this challenge in future reports when comparing with gradient methods.
4.  **Task 4 Preview**:
    -   Task 4 will involve comparison with gradient-based methods.
    -   It will be interesting to see the comparison.

### Action Items for Task 4
-   [ ] **Modify Plotting**: Update `src/visualization/plotting.py` to support "zoomed-in" views or log-x axes for convergence plots.
-   [ ] **Algorithm Selection**: When writing the Task 4 report, frame the conclusion around "suitability for problem type" rather than "superiority".
-   [ ] **Discussion**: Add a section on termination criteria differences between Stochastic and Gradient-based methods.
