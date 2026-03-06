# Task 6 Report: Hyperboloid Cooling Tower Optimization

Generated: 2026-03-06 18:06

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
- Seed offset: 50000
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
| S8 | joint | 10 | 90000 | 39.3 | 27.4 | Joint constructability-aware design with larger target volume. |

## 7. Cross-Algorithm Summary
| Algorithm | Total Runs | Mean Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| SA | 3 | 7728.62 | 6.849e-04 | 12000.0 | 0.6087 | 33.3% |
| PSO | 3 | 8689.56 | 2.365e-01 | 12000.0 | 0.6475 | 0.0% |
| BFGS | 3 | 7778.97 | 6.525e-04 | 4839.0 | 0.2583 | 33.3% |

![Cross-scenario area comparison](../figures/cross_scenario_area_bar.png)

## 8. Scenario Results

### S1
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7782.01 | 0.00 | 1.537e-03 | 5001.0 | 0.2367 | 0.0% |
| PSO | 7827.12 | 0.00 | 7.380e-03 | 10000.0 | 0.5257 | 0.0% |
| SA | 7773.40 | 0.00 | 1.802e-03 | 10000.0 | 0.4912 | 0.0% |

![S1 convergence](../figures/S1_convergence.png)

![S1 profile](../figures/S1_profile_overlay.png)


### S2
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7741.67 | 0.00 | 1.899e-06 | 2515.0 | 0.1016 | 100.0% |
| PSO | 8590.62 | 0.00 | 7.007e-01 | 10000.0 | 0.4301 | 0.0% |
| SA | 7743.17 | 0.00 | 7.252e-06 | 10000.0 | 0.4042 | 100.0% |

![S2 convergence](../figures/S2_convergence.png)

![S2 profile](../figures/S2_profile_overlay.png)


### S8
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7813.24 | 0.00 | 4.192e-04 | 7001.0 | 0.4365 | 0.0% |
| PSO | 9650.93 | 0.00 | 1.389e-03 | 16000.0 | 0.9869 | 0.0% |
| SA | 7669.30 | 0.00 | 2.455e-04 | 16000.0 | 0.9308 | 0.0% |

![S8 convergence](../figures/S8_convergence.png)

![S8 profile](../figures/S8_profile_overlay.png)

## 9. Key Findings

- S1: best feasibility-area tradeoff = SA (feasibility 0.0%, mean area 7773.40 m^2).
- S1: lowest mean evaluation count = BFGS (5001.0 evals).
- S2: best feasibility-area tradeoff = BFGS (feasibility 100.0%, mean area 7741.67 m^2).
- S2: lowest mean evaluation count = BFGS (2515.0 evals).
- S8: best feasibility-area tradeoff = SA (feasibility 0.0%, mean area 7669.30 m^2).
- S8: lowest mean evaluation count = BFGS (7001.0 evals).

- Feasibility bottleneck scenarios: S1, S8 (all tested methods yielded 0% feasible runs under current constraints).

## 10. Discussion and Conclusions

Across all scenarios, **SA** achieved the highest overall feasibility (33.3%).
**BFGS** required the fewest evaluations on average (4839.0), confirming its efficiency for local convergence.
By raw mean area across all runs, **SA** produced the lowest average shell area (7728.62 m^2), but this must be interpreted jointly with feasibility rates.

Result patterns are consistent with expected method behavior: BFGS is computationally efficient and strong when gradients and local curvature align with feasible basins; SA is often robust under hard nonlinear penalties because probabilistic acceptance helps transition between constrained basins; PSO explores broadly but can underperform feasibility when penalty landscapes are steep and high-dimensional.

For practical engineering usage in this problem family, the recommended workflow is a hybrid strategy: use SA/PSO for global search and feasibility discovery, then warm-start BFGS for rapid refinement of the best feasible candidate.

The remaining limitation is that optimization does not by itself confirm structural suitability. S2 shows that mathematically feasible shapes can still look impractical, while S8 shows that higher-capacity, constructability-aware designs remain difficult under the current formulation.

### 10.1 Planned Structural Validation in Abaqus

The next step is to transfer the best candidate geometries into Abaqus shell models with realistic material properties and shell-thickness assumptions for reinforced-concrete cooling towers. These models will be checked under self-weight and representative environmental loading to evaluate displacement, stress distribution, and buckling sensitivity.

Multiple candidate towers will be compared, not only the minimum-area design, so selection can be based on structural feasibility as well as geometric efficiency. The Abaqus results will then be used to refine the optimization model by tightening feasibility rules or adding constraints linked to smoothness, curvature, and stability.