# Task 6 Report: Hyperboloid Cooling Tower Optimization

Generated: 2026-02-25 18:54

## 1. Objective
Minimize cooling-tower lateral shell area using SA, PSO, and BFGS while enforcing fixed target volume and scenario-specific constructability constraints.
The study compares stochastic and deterministic optimization behavior across eight structurally different problem setups while keeping common engineering requirements (capacity and realistic shape) explicit in the objective.

## 2. Geometry Model
The tower is modeled as a stack of frustums.

- Frustum area: `A_i = pi * (r_{i-1} + r_i) * sqrt((r_i - r_{i-1})^2 + h_i^2)`
- Frustum volume: `V_i = (pi * h_i / 3) * (r_{i-1}^2 + r_{i-1}*r_i + r_i^2)`
- Tower totals: `A = sum(A_i)`, `V = sum(V_i)`

Lateral shell area is optimized (top and bottom caps excluded), which directly corresponds to shell-construction material for fixed end radii.
Analytic gradients were used for both `A` and `V`, and for all penalty terms, so BFGS receives exact first-order information.

### 2.1 Design Variables by Scenario Type
- `radii` mode: optimize interior radii `r1..r_{m-1}` while ring heights are fixed from `z` levels.
- `heights` mode: optimize segment heights `h1..hm` while all ring radii are fixed.
- `joint` mode: optimize both interior radii and all heights simultaneously.

## 3. Optimization Methods

### 3.1 Simulated Annealing (SA)
SA performs single-solution stochastic search. At each evaluation, a local perturbation is proposed and accepted if it improves objective, or probabilistically accepted otherwise using the temperature schedule.
This supports escape from local minima early in search and gradual exploitation as temperature cools.

### 3.2 Particle Swarm Optimization (PSO)
PSO evolves a population of particles with velocity updates using inertia, cognitive pull to personal best, and social pull to global best.
It is typically strong on global exploration and multimodal landscapes, but can require many evaluations to satisfy strict constraints.

### 3.3 BFGS (Deterministic Quasi-Newton)
BFGS updates an inverse-Hessian approximation and computes descent directions with line search, yielding rapid local convergence when gradients are informative.
It is efficient in evaluation count but can be sensitive to initialization and non-convex penalties.

## 4. Penalized Objective and Geometrical Constraints
The optimized scalar objective is `J(x) = A(x) + P_volume + P_height + P_shape + P_smooth + P_bounds + P_ratio` (terms enabled per scenario).

### 4.1 Constraint Terms and Their Function
- `P_volume`: enforces target cooling capacity by penalizing relative deviation from target volume.
- `P_height`: enforces total tower height in scenarios where heights vary, preserving comparable overall structure.
- `P_shape`: enforces hyperboloid-like monotonic contraction to neck and expansion above neck.
- `P_smooth`: penalizes second differences in radii to avoid oscillatory/sawtooth shell profiles.
- `P_bounds`: hinge penalty for design-variable bounds to keep variables in practical/constructable ranges.
- `P_ratio` (S8): limits adjacent-height ratios to promote constructability and avoid abrupt segment transitions.

### 4.2 Feasibility Rule
- Relative volume error must be `<= 1e-3`.
- If heights vary, relative total-height error must be `<= 1e-3`.
- If radii vary, monotonic hyperboloid violation must be `<= 1e-6`.

## 5. Optimization Protocol
- Runs per algorithm per scenario: 1
- Seed offset: 0
- SA: `temp_init=120`, `cooling_rate=0.997`, scenario-dependent `step_size`
- PSO: `w=0.65`, `c1=1.6`, `c2=1.7`, particles = 40 or 50 by dimension
- BFGS: `tol=1e-7`, `max_iter=1200`
- Penalties: volume, height sum (when applicable), shape monotonicity, smoothness, bounds, ratio (S8)

Stochastic budgets are 10k evaluations for lower-dimensional setups and 16k for joint/high-dimensional setups. BFGS uses smaller budgets due to faster local convergence.

## 6. Scenario Definitions
| ID | Mode | m | Target Volume | r0 | rm | Description |
|---|---|---|---|---|---|---|
| S1 | radii | 10 | 70320 | 39.3 | 27.4 | Required case with bounded radii and required ring heights. |
| S2 | radii | 10 | 70320 | 39.3 | 27.4 | Required case with wide radii range (unconstrained-like). |
| S3 | radii | 10 | 70320 | 39.3 | 27.4 | Uniform-height radii design with bounded radii. |
| S4 | radii | 12 | 70320 | 39.3 | 27.4 | Finer discretization (m=12) with bounded radii and smoothness. |
| S5 | heights | 10 | 70320 | 39.3 | 27.4 | Fixed radii, optimize bounded heights. |
| S6 | heights | 10 | 70320 | 39.3 | 27.4 | Fixed radii, optimize wide-range heights (unconstrained-like). |
| S7 | joint | 10 | 70320 | 39.3 | 27.4 | Joint bounded optimization of radii and heights. |
| S8 | joint | 10 | 90000 | 39.3 | 27.4 | Joint constructability-aware design with larger target volume. |

## 7. Cross-Algorithm Summary
| Algorithm | Total Runs | Mean Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| SA | 8 | 7781.46 | 6.159e-04 | 11500.0 | 1.0160 | 62.5% |
| PSO | 8 | 8230.94 | 1.237e-02 | 11500.0 | 1.0459 | 75.0% |
| BFGS | 8 | 7807.58 | 3.919e-04 | 4197.5 | 0.3592 | 75.0% |

![Cross-scenario area comparison](../figures/cross_scenario_area_bar.png)

## 8. Scenario Results

### S1
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7782.01 | 0.00 | 1.537e-03 | 5001.0 | 0.4115 | 0.0% |
| PSO | 7819.93 | 0.00 | 3.504e-04 | 10000.0 | 0.9225 | 100.0% |
| SA | 7779.78 | 0.00 | 1.897e-03 | 10000.0 | 0.9014 | 0.0% |

![S1 convergence](../figures/S1_convergence.png)

![S1 profile](../figures/S1_profile_overlay.png)

![S1 3D towers](../figures/S1_tower_3d.png)


### S2
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7741.67 | 0.00 | 1.899e-06 | 3606.0 | 0.2617 | 100.0% |
| PSO | 9075.03 | 0.00 | 2.663e-04 | 10000.0 | 0.8287 | 100.0% |
| SA | 7751.12 | 0.00 | 2.854e-05 | 10000.0 | 0.8041 | 100.0% |

![S2 convergence](../figures/S2_convergence.png)

![S2 profile](../figures/S2_profile_overlay.png)

![S2 3D towers](../figures/S2_tower_3d.png)


### S3
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7741.69 | 0.00 | 1.900e-06 | 2142.0 | 0.1632 | 100.0% |
| PSO | 8014.30 | 0.00 | 1.301e-05 | 10000.0 | 0.8033 | 100.0% |
| SA | 7744.82 | 0.00 | 2.512e-05 | 10000.0 | 0.8046 | 100.0% |

![S3 convergence](../figures/S3_convergence.png)

![S3 profile](../figures/S3_profile_overlay.png)

![S3 3D towers](../figures/S3_tower_3d.png)


### S4
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7779.85 | 0.00 | 8.915e-04 | 5001.0 | 0.4409 | 100.0% |
| PSO | 8365.55 | 0.00 | 9.705e-02 | 10000.0 | 1.0105 | 0.0% |
| SA | 7766.67 | 0.00 | 1.979e-03 | 10000.0 | 0.9973 | 0.0% |

![S4 convergence](../figures/S4_convergence.png)

![S4 profile](../figures/S4_profile_overlay.png)

![S4 3D towers](../figures/S4_tower_3d.png)


### S5
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7742.56 | 0.00 | 2.331e-06 | 2427.0 | 0.1761 | 100.0% |
| PSO | 7819.97 | 0.00 | 1.489e-04 | 10000.0 | 0.7615 | 100.0% |
| SA | 7754.88 | 0.00 | 6.674e-05 | 10000.0 | 0.7261 | 100.0% |

![S5 convergence](../figures/S5_convergence.png)

![S5 profile](../figures/S5_profile_overlay.png)

![S5 3D towers](../figures/S5_tower_3d.png)


### S6
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7741.72 | 0.00 | 2.071e-06 | 1401.0 | 0.0951 | 100.0% |
| PSO | 7914.47 | 0.00 | 3.403e-06 | 10000.0 | 0.7871 | 100.0% |
| SA | 7783.24 | 0.00 | 1.478e-05 | 10000.0 | 0.7521 | 100.0% |

![S6 convergence](../figures/S6_convergence.png)

![S6 profile](../figures/S6_profile_overlay.png)

![S6 3D towers](../figures/S6_tower_3d.png)


### S7
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 8117.92 | 0.00 | 2.799e-04 | 7001.0 | 0.5477 | 100.0% |
| PSO | 8596.66 | 0.00 | 1.888e-05 | 16000.0 | 1.3884 | 100.0% |
| SA | 7952.90 | 0.00 | 7.725e-05 | 16000.0 | 1.3875 | 100.0% |

![S7 convergence](../figures/S7_convergence.png)

![S7 profile](../figures/S7_profile_overlay.png)

![S7 3D towers](../figures/S7_tower_3d.png)


### S8
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7813.24 | 0.00 | 4.192e-04 | 7001.0 | 0.7771 | 0.0% |
| PSO | 8241.63 | 0.00 | 1.074e-03 | 16000.0 | 1.8651 | 0.0% |
| SA | 7718.27 | 0.00 | 8.381e-04 | 16000.0 | 1.7551 | 0.0% |

![S8 convergence](../figures/S8_convergence.png)

![S8 profile](../figures/S8_profile_overlay.png)

![S8 3D towers](../figures/S8_tower_3d.png)

## 9. Key Findings

- S1: best feasibility-area tradeoff = PSO (feasibility 100.0%, mean area 7819.93 m^2).
- S1: lowest mean evaluation count = BFGS (5001.0 evals).
- S2: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7741.67 m^2).
- S2: lowest mean evaluation count = BFGS (3606.0 evals).
- S3: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7741.69 m^2).
- S3: lowest mean evaluation count = BFGS (2142.0 evals).
- S4: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7779.85 m^2).
- S4: lowest mean evaluation count = BFGS (5001.0 evals).
- S5: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7742.56 m^2).
- S5: lowest mean evaluation count = BFGS (2427.0 evals).
- S6: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7741.72 m^2).
- S6: lowest mean evaluation count = BFGS (1401.0 evals).
- S7: best feasibility-area tradeoff = SA (feasibility 100.0%, mean area 7952.90 m^2).
- S7: lowest mean evaluation count = BFGS (7001.0 evals).
- S8: best feasibility-area tradeoff = SA (feasibility 0.0%, mean area 7718.27 m^2).
- S8: lowest mean evaluation count = BFGS (7001.0 evals).

- Feasibility bottleneck scenarios: S8 (all tested methods yielded 0% feasible runs under current constraints).

## 10. Discussion and Conclusions

Across all scenarios, **PSO** achieved the highest overall feasibility (75.0%).
**BFGS** required the fewest evaluations on average (4197.5), confirming its efficiency for local convergence.
By raw mean area across all runs, **SA** produced the lowest average shell area (7781.46 m^2), but this must be interpreted jointly with feasibility rates.

Result patterns are consistent with expected method behavior: BFGS is computationally efficient and strong when gradients and local curvature align with feasible basins; SA is often robust under hard nonlinear penalties because probabilistic acceptance helps transition between constrained basins; PSO explores broadly but can underperform feasibility when penalty landscapes are steep and high-dimensional.

For practical engineering usage in this problem family, the recommended workflow is a hybrid strategy: use SA/PSO for global search and feasibility discovery, then warm-start BFGS for rapid refinement of the best feasible candidate.

Scenario S8 remains the principal bottleneck under the current penalty/budget setup; this indicates that constructability and enlarged-volume requirements are jointly tight. Future refinement should focus on adaptive penalty scheduling, better initialization heuristics for joint variables, and potentially increased stochastic budget for S8.

## 11. Deliverables
- `results/data/raw_runs.csv`
- `results/data/scenario_summary.csv`
- `results/data/algorithm_summary.csv`
- `results/data/scenario_definitions.json`
- `results/figures/*.png`