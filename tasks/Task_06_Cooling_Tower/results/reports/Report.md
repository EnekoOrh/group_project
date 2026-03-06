# Task 6 Report: Hyperboloid Cooling Tower Optimization

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
- Runs per algorithm per scenario: 10
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
| S3 | radii | 10 | 70320 | 39.3 | 27.4 | Uniform-height radii design with bounded radii. |
| S4 | radii | 12 | 70320 | 39.3 | 27.4 | Finer discretization (m=12) with bounded radii and smoothness. |
| S5 | heights | 10 | 70320 | 39.3 | 27.4 | Fixed radii, optimize bounded heights. |
| S6 | heights | 10 | 70320 | 39.3 | 27.4 | Fixed radii, optimize wide-range heights (unconstrained-like). |
| S7 | joint | 10 | 70320 | 39.3 | 27.4 | Joint bounded optimization of radii and heights. |
| S8 | joint | 10 | 90000 | 39.3 | 27.4 | Joint constructability-aware design with larger target volume. |

## 7. Scenario Results

### S1
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7782.01 | 0.00 | 1.537e-03 | 5001.0 | 0.2401 | 0.0% |
| PSO | 7993.19 | 251.35 | 4.408e-03 | 10000.0 | 0.5261 | 30.0% |
| SA | 7783.36 | 5.13 | 1.051e-03 | 10000.0 | 0.5017 | 40.0% |

![S1 convergence](results/figures/S1_convergence.png)

![S1 profile](results/figures/S1_profile_overlay.png)

![S1 3D towers](results/figures/S1_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (3.739e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- PSO: **FEASIBLE** - passes volume (5.843e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- BFGS: **INFEASIBLE** - fails volume (1.537e-03 > 1.000e-03). visualized run is lowest penalized objective among infeasible runs.

S1 remains a difficult required case even though the plotted profiles are visually very similar. **SA** gives the best feasibility-area tradeoff with 40.0% feasibility and mean area 7783.36 m^2, while **PSO** improves feasibility only modestly and still pays a clear area penalty. **BFGS** is the fastest method, but here speed does not translate into reliability because the same low-area local outcome remains volume-infeasible.

The main point in S1 is that the search space contains a narrow feasible basin around the required geometry. The algorithms do not differ much in the shapes they ultimately prefer; they differ in how often they can satisfy the strict volume tolerance while keeping the surface area low.


### S2
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 8574.90 | 709.96 | 2.548e-04 | 4215.4 | 0.1657 | 90.0% |
| PSO | 9008.37 | 629.49 | 1.375e-01 | 10000.0 | 0.4415 | 40.0% |
| SA | 7766.94 | 29.81 | 4.278e-05 | 10000.0 | 0.4106 | 100.0% |

![S2 convergence](results/figures/S2_convergence.png)

![S2 profile](results/figures/S2_profile_overlay.png)

![S2 3D towers](results/figures/S2_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (7.252e-06 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- PSO: **FEASIBLE** - passes volume (6.404e-05 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- BFGS: **FEASIBLE** - passes volume (1.899e-06 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).

S2 again shows why mathematical feasibility is not the same as practical plausibility. **SA** is the strongest method here with 100.0% feasibility and mean area 7766.94 m^2, while **BFGS** remains mostly feasible but much less robust in mean area and **PSO** remains weak on both area and consistency.

Because S2 deliberately removes smoothness control and allows a wide radii range, feasible profiles can still look structurally awkward. That is visible in the PSO profile, which has a sharper throat and more abrupt expansion than the smoother SA and BFGS solutions. This is the clearest scenario where the report must distinguish formal feasibility from practical shape quality.


### S3
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7862.58 | 362.68 | 5.966e-05 | 2410.2 | 0.1001 | 100.0% |
| PSO | 8551.22 | 598.39 | 4.491e-05 | 10000.0 | 0.4379 | 70.0% |
| SA | 7750.71 | 12.99 | 2.640e-05 | 10000.0 | 0.4050 | 100.0% |

![S3 convergence](results/figures/S3_convergence.png)

![S3 profile](results/figures/S3_profile_overlay.png)

![S3 3D towers](results/figures/S3_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (6.054e-05 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- PSO: **FEASIBLE** - passes volume (5.240e-06 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- BFGS: **FEASIBLE** - passes volume (1.900e-06 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).

S3 is more stable than S2 because the uniform-height setting simplifies the search. **SA** and **BFGS** both remain fully feasible, with SA giving the lower mean area at 7750.71 m^2 and BFGS remaining the fastest method at 2410.2 evaluations on average. **PSO** can still find a good design, but its lower feasibility and higher variance show that it reaches that quality less consistently.

The plotted profiles support this interpretation: SA and BFGS are almost indistinguishable visually, suggesting that the bounded uniform-height radii problem has a clear preferred geometry. The main remaining uncertainty is therefore algorithmic consistency rather than disagreement about the tower shape itself.


### S4
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7779.85 | 0.00 | 8.915e-04 | 4391.3 | 0.2261 | 100.0% |
| PSO | 8183.83 | 485.21 | 2.006e-02 | 10000.0 | 0.5589 | 20.0% |
| SA | 7776.90 | 12.83 | 1.712e-03 | 10000.0 | 0.5228 | 20.0% |

![S4 convergence](results/figures/S4_convergence.png)

![S4 profile](results/figures/S4_profile_overlay.png)

![S4 3D towers](results/figures/S4_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (3.849e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- PSO: **FEASIBLE** - passes volume (5.113e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- BFGS: **FEASIBLE** - passes volume (8.915e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).

S4 combines finer discretization with smoothness control, so it is a stronger test of repeated convergence than the simpler radii-only cases. **BFGS** remains dominant with 100.0% feasibility and mean area 7779.85 m^2, whereas **SA** and **PSO** find feasible solutions only occasionally even though their selected plotted towers are competitive in shape and area.

The correct interpretation is not that the stochastic methods produce bad geometries when they succeed. Rather, the extra discretization and smoothness penalty make this scenario harder to solve consistently, and BFGS handles that additional structure much more reliably than SA or PSO.


### S5
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7742.56 | 0.00 | 2.331e-06 | 3371.2 | 0.1285 | 100.0% |
| PSO | 7802.45 | 23.76 | 4.302e-04 | 10000.0 | 0.4200 | 70.0% |
| SA | 7758.49 | 6.64 | 4.108e-05 | 10000.0 | 0.3903 | 100.0% |

![S5 convergence](results/figures/S5_convergence.png)

![S5 profile](results/figures/S5_profile_overlay.png)

![S5 3D towers](results/figures/S5_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (5.319e-05 <= 1.000e-03), height (1.809e-04 <= 1.000e-03).
- PSO: **FEASIBLE** - passes volume (5.000e-04 <= 1.000e-03), height (3.988e-04 <= 1.000e-03).
- BFGS: **FEASIBLE** - passes volume (2.331e-06 <= 1.000e-03), height (8.747e-05 <= 1.000e-03).

S5 is a comparatively well-behaved bounded heights-only problem. **BFGS** is strongest on both accuracy and efficiency, reaching 100.0% feasibility with mean area 7742.56 m^2 and the lowest evaluation count. **SA** is also fully feasible but slightly less efficient, while **PSO** is the least reliable method in this setting.

The three plotted profiles are close, which means the scenario is not about discovering radically different shapes. Instead, it is about how precisely each method can fine-tune the segment heights while keeping both volume and total height within tolerance.


### S6
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7741.72 | 0.00 | 2.071e-06 | 1403.3 | 0.0554 | 100.0% |
| PSO | 7971.10 | 107.04 | 2.812e-06 | 10000.0 | 0.4161 | 100.0% |
| SA | 7847.58 | 27.68 | 5.217e-05 | 10000.0 | 0.3912 | 100.0% |

![S6 convergence](results/figures/S6_convergence.png)

![S6 profile](results/figures/S6_profile_overlay.png)

![S6 3D towers](results/figures/S6_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (7.270e-06 <= 1.000e-03), height (3.561e-04 <= 1.000e-03).
- PSO: **FEASIBLE** - passes volume (5.197e-06 <= 1.000e-03), height (9.426e-05 <= 1.000e-03).
- BFGS: **FEASIBLE** - passes volume (2.071e-06 <= 1.000e-03), height (8.632e-05 <= 1.000e-03).

S6 becomes easier once the height bounds are widened. All three methods are fully feasible in this run, but **BFGS** still gives the lowest mean area at 7741.72 m^2 and is by far the fastest method at 1403.3 evaluations. **PSO** becomes competitive on feasibility, but not on area, and **SA** remains the largest of the three mean designs.

This scenario shows that relaxing the height bounds can remove much of the feasibility difficulty without changing the overall algorithm ranking. Greater design freedom helps every method, but it does not overturn the advantage of BFGS on efficiency or final area.


### S7
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7851.72 | 305.19 | 5.959e-05 | 7000.9 | 0.3282 | 100.0% |
| PSO | 8475.39 | 478.15 | 1.649e-03 | 16000.0 | 0.7784 | 60.0% |
| SA | 8213.61 | 258.51 | 1.270e-04 | 16000.0 | 0.7232 | 80.0% |

![S7 convergence](results/figures/S7_convergence.png)

![S7 profile](results/figures/S7_profile_overlay.png)

![S7 3D towers](results/figures/S7_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **FEASIBLE** - passes volume (6.863e-05 <= 1.000e-03), height (1.425e-04 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- PSO: **FEASIBLE** - passes volume (3.294e-05 <= 1.000e-03), height (4.207e-05 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).
- BFGS: **FEASIBLE** - passes volume (1.890e-06 <= 1.000e-03), height (8.481e-05 <= 1.000e-03), shape (0.000e+00 <= 1.000e-06).

S7 is the clearest joint optimization case in the study. **BFGS** is the strongest method here with 100.0% feasibility, mean area 7851.72 m^2, and the lowest evaluation count in the scenario. **SA** and **PSO** can still produce feasible towers, but they do so less reliably and with materially larger mean areas.

The profile overlay helps explain the ranking. BFGS maintains the cleanest transition through the neck region while staying close to the lowest-area profile, whereas PSO shows a boxier middle section and SA settles on a smoother but noticeably larger tower. This is a good example of a coupled radii-height problem where strong local refinement is especially valuable.


### S8
| Algorithm | Mean Area | Std Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| BFGS | 7813.00 | 2.02 | 4.222e-04 | 7000.9 | 0.4426 | 0.0% |
| PSO | 8363.77 | 635.91 | 2.748e-02 | 16000.0 | 1.0040 | 0.0% |
| SA | 7660.78 | 38.06 | 6.456e-04 | 16000.0 | 0.9494 | 0.0% |

![S8 convergence](results/figures/S8_convergence.png)

![S8 profile](results/figures/S8_profile_overlay.png)

![S8 3D towers](results/figures/S8_tower_3d.png)

Feasibility of the shown 3D towers (same selected runs as profile overlay):
- SA: **INFEASIBLE** - fails height (4.596e-03 > 1.000e-03). visualized run is lowest penalized objective among infeasible runs.
- PSO: **INFEASIBLE** - fails volume (1.539e-03 > 1.000e-03), height (5.174e-03 > 1.000e-03), shape (6.325e-04 > 1.000e-06). visualized run is lowest penalized objective among infeasible runs.
- BFGS: **INFEASIBLE** - fails height (2.684e-03 > 1.000e-03), shape (2.649e-03 > 1.000e-06). visualized run is lowest penalized objective among infeasible runs.

S8 remains the strongest bottleneck in the whole study. All three methods have 0.0% feasibility under the current constraints, even though **SA** still reaches the lowest mean area at 7660.78 m^2 and the lowest penalized objective among the infeasible methods.

The convergence and profile plots show that this is not simply a random failure case. The enlarged target volume, smoothness requirement, and adjacent-height ratio control together create a genuinely hard joint problem. BFGS remains more controlled than PSO, but neither method can satisfy the full feasibility rule, so S8 still marks the limit of the current formulation.

## 8. Cross-Algorithm Summary
| Algorithm | Total Runs | Mean Area | Mean Rel Vol Err | Mean Evals | Mean Time (s) | Feasibility Rate |
|---|---|---|---|---|---|---|
| SA | 80 | 7819.80 | 4.623e-04 | 11500.0 | 0.5368 | 67.5% |
| PSO | 80 | 8293.67 | 2.395e-02 | 11500.0 | 0.5729 | 48.8% |
| BFGS | 80 | 7893.54 | 4.036e-04 | 4349.3 | 0.2108 | 73.8% |

![Cross-scenario area comparison](results/figures/cross_scenario_area_bar.png)

## 9. Discussion and Conclusions

Across all scenarios, **BFGS** achieved the highest overall feasibility (73.8%).
**BFGS** required the fewest evaluations on average (4349.3), confirming its efficiency for local convergence.
By raw mean area across all runs, **SA** produced the lowest average shell area (7819.80 m^2), but this must be interpreted jointly with feasibility rates.

Result patterns are consistent with expected method behavior: BFGS is computationally efficient and strong when gradients and local curvature align with feasible basins; SA is often robust under hard nonlinear penalties because probabilistic acceptance helps transition between constrained basins; PSO explores broadly but can underperform feasibility when penalty landscapes are steep and high-dimensional.

For practical engineering usage in this problem family, the recommended workflow is a hybrid strategy: use SA/PSO for global search and feasibility discovery, then warm-start BFGS for rapid refinement of the best feasible candidate.

The remaining limitation is that optimization does not by itself confirm structural suitability. S2 shows that mathematically feasible shapes can still look impractical, while S8 shows that higher-capacity, constructability-aware designs remain difficult under the current formulation.

Under the current penalty and budget settings, S8 remains the clearest feasibility bottleneck, indicating that the present formulation still struggles when constructability and higher-capacity requirements are combined.

### 9.1 Planned Structural Validation in Abaqus

The next step is to transfer the best candidate geometries into Abaqus shell models with realistic material properties and shell-thickness assumptions for reinforced-concrete cooling towers. These models will be checked under self-weight and representative environmental loading to evaluate displacement, stress distribution, and buckling sensitivity.

Multiple candidate towers will be compared, not only the minimum-area design, so selection can be based on structural feasibility as well as geometric efficiency. The Abaqus results will then be used to refine the optimization model by tightening feasibility rules or adding constraints linked to smoothness, curvature, and stability.