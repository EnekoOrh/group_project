# Task 9 Final Presentation Plan — After Slide 34

This document outlines the slides to add/modify in `Task5_Montrésor_Presentation.pptx` for the final presentation, starting after the "No Free Lunch Theorem" comparison (current slide 34).

Per Task 9 requirements: **15–20 slides, final slide = poster**, 20 min presentation + 10 min Q&A. Content must describe both **product** (outcomes from Tasks 2–4, 6–7) and **process** (how the group worked).

---

## Slide 35 — Section Divider

**Title:** "Case Study: Cooling Tower Optimization"

Visual: Large section divider image. Subtitle: "Applying SA, PSO, and BFGS to a Real Engineering Problem"

---

## Slide 36 — Problem Definition

**Title:** "What Are We Optimizing?"

**Content:**
- **Real engineering problem:** Minimize the lateral shell area of a hyperboloid cooling tower while maintaining fixed cooling capacity (volume = 7.032×10⁴ m³).
- **Geometry model:** Tower built as a stack of 10 conical frustums:
  - A_i = π(r_{i-1} + r_i) × sqrt((r_i − r_{i-1})² + h_i²)
  - V_i = (πh_i)/3 × (r_{i-1}² + r_{i-1}r_i + r_i²)
- **Fixed constraints:** Base radius r₀ = 39.3 m, top radius r_m = 27.4 m, total height target = 36.5 m.
- **Design variables:** Interior radii (radii mode), segment heights (heights mode), or both (joint mode).
- **Key insight:** This transforms abstract mathematics into a concrete constructability challenge.

**Visual:** Diagram of frustum stack with annotated radii and heights.

---

## Slide 37 — From Math to Engineering: 8 Scenarios

**Title:** "8 Scenarios of Increasing Complexity"

**Content (table):**

| ID | Mode | m | Target Vol | Key Challenge |
|---|---|---|---|---|
| S1 | radii | 10 | 70,320 | Baseline constrained case with smoothness |
| S2 | radii | 10 | 70,320 | Wide radii range, relaxed constraints |
| S3 | radii | 10 | 70,320 | Uniform height discretization |
| S4 | radii | 12 | 70,320 | Higher resolution (12 segments) + smoothness |
| S5 | heights | 10 | 70,320 | Heights-only with fixed radii template |
| S6 | heights | 10 | 70,320 | Wide-range heights (unconstrained-like) |
| S7 | joint | 10 | 70,320 | Joint radii + heights optimization |
| S8 | joint | 10 | 90,000 | **Industrial stress test:** larger volume + strict constructability |

Each scenario: 10 runs per algorithm, deterministic seed control.

---

## Slide 38 — Penalty Functions: Engineering Realism

**Title:** "How We Enforced Engineering Feasibility"

**Content:**
- The optimizers minimize a penalized objective:
  - **J(x) = Area + P_volume + P_height + P_shape + P_smooth + P_bounds + P_ratio** (terms enabled per scenario)
- **P_volume:** Penalizes deviation from target volume (λ_vol = 2×10⁸)
- **P_shape:** Enforces hyperboloid monotonicity — radii must contract to neck then expand (λ_shape = 5×10⁶)
- **P_smooth:** Penalizes second differences in radii to avoid oscillatory profiles
- **Two levels of success:**
  - **Mathematical compliance:** Relative volume error ≤ 10⁻³, shape violation ≤ 10⁻⁶
  - **Engineering feasibility:** Also passes practical checks: neck radius ≥ 18 m, smooth transitions, height ratios

**Key message:** A design can be mathematically compliant but ENGINEERING INFEAISIBLE! This distinction is critical for real-world applications.

---

## Slide 39 — Cross-Algorithm Aggregate Results

**Title:** "All 8 Scenarios: 240 Total Runs"

**Content (table):**

| Algorithm | Math Compliance | Engineering Feasibility | Mean Evals | Feasible Mean Area (m²) |
|---|---|---|---|---|
| **BFGS** | **73.8%** | **61.3%** | 4,408 | 7,871 |
| SA | 71.2% | 50.0% | 11,500 | 7,842 |
| PSO | 50.0% | 7.5% | 11,500 | 8,213 |

**Bullet points:**
- **BFGS:** Highest mathematical compliance AND engineering feasibility. Most evaluation-efficient.
- **SA:** Lowest raw mean areas (7,815 m²), but lower feasibility rates.
- **PSO:** Struggled in high-dimensional penalty landscapes (7.5% engineering feasibility).
- Budget asymmetry: SA/PSO used ~2.6× more evaluations than BFGS.

**Visual:** Cross-scenario area bar chart from `figures/cross_scenario_area_bar.png`.

---

## Slide 40 — Scenario-by-Scenario Winners

**Title:** "Who Won Where?"

**Content (table):**

| Scenario | Best Math Compliance | Best Engineering Feasibility | Most Efficient |
|---|---|---|---|
| S1 | SA (50%) | SA (50%) | BFGS (3,973 evals) |
| S2 | SA (100%) | SA (100%) | BFGS (3,550 evals) |
| S3 | BFGS (100%) | SA (100%) | BFGS (2,959 evals) |
| S4 | BFGS (100%) | BFGS (100%) | BFGS (5,001 evals) |
| S5 | BFGS (100%) | BFGS (100%) | BFGS (3,657 evals) |
| S6 | BFGS (100%) | BFGS (100%) | BFGS (2,121 evals) |
| S7 | BFGS (100%) | BFGS (60%) | BFGS (7,001 evals) |
| S8 | ALL 0% | ALL 0% | BFGS (7,001 evals) |

**Key observations:**
- BFGS dominates efficiency and handles simpler scenarios perfectly.
- SA is remarkably robust on S2 and S3 (100% feasibility).
- S8 bottleneck: combined high demands + constructability constraints → no feasible design.

---

## Slide 41 — Visual: Tower Profiles (S7)

**Title:** "S7: Joint Optimization — Radii + Heights"

**Content:**
- Side-by-side 3D tower comparison for S7 (SA / PSO / BFGS).
- All three achieved 100% mathematical compliance.
- Only BFGS produced engineering-feasible designs (60%).
- SA and PSO designs had engineering issues: neck_radius failures, radius_second_diff violations.

**Visual:** 3D tower figures from `S7_tower_3d.png`.

---

## Slide 42 — From Optimization to Structural Reality

**Title:** "Abaqus FEA: Validating the Optimized Designs"

**Content:**
- **Selection rule:** 24 towers (8 scenarios × 3 algorithms), picking lowest-area engineering-feasible Task 6 run per pair.
  - Fallback: mathematically compliant if no engineering-feasible exists.
  - Warning: best penalized run if neither exists.
  - Result: 16 engineering-feasible, 3 mathematical fallbacks, 5 warning-only.
- **Material model:** Equivalent reinforced concrete
  - E = 33 GPa, ν = 0.20, ρ = 2,500 kg/m³, shell t = 0.20 m
- **Loading:** Self-weight (gravity) + one-directional wind
  - Reference wind speed: 30 m/s → dynamic pressure q = 562.5 Pa
  - Circumferential pressure law: C_p(θ) = clip(0.8 cos θ, −0.5, 0.8)
- **Mesh:** Refined: 40 circumferential divisions, 2 axial subdivisions per segment.

**Visual:** Abaqus model diagram or mesh screenshot.

---

## Slide 43 — Abaqus: Weighted Ranking

**Title:** "Global Structural Ranking — All 24 Towers"

**Content:**
- **Weighted score formula:** S_i = 0.45 × rank(buckling) + 0.25 × rank(displacement) + 0.20 × rank(stress) + 0.10 × rank(area)
- **Penalty:** J_i = S_i + P_i where P_i ∈ {0, 10, 20} (engineering-feasible / math fallback / warning)
- Lower = better.

| Rank | Case | Status | Score | Buckling | Disp (mm) | Stress (MPa) | Area (m²) |
|---|---|---|---|---|---|---|---|
| **1** | **S7 / BFGS** | Eng. Feasible | **6.75** | 24.95 | 10.1 | 4.13 | 7,741 |
| 2 | S3 / SA | Eng. Feasible | 7.05 | 23.83 | 9.0 | 3.53 | 7,743 |
| 3 | S2 / SA | Eng. Feasible | 7.50 | 24.55 | 10.0 | 3.82 | 7,744 |
| 4 | S3 / PSO | Eng. Feasible | 9.65 | 23.50 | 9.9 | 3.79 | 7,743 |
| 5 | S2 / BFGS | Eng. Feasible | 9.85 | 23.56 | 10.2 | 3.87 | 7,742 |

**Important caveat:** Convergence evidence is 0/24 all-pass (coarse vs refined mesh). Rankings are **screening quality**, not final qualification.

---

## Slide 44 — The Global Winner: S7 / BFGS

**Title:** "S7 / BFGS: Best Compromise Across Stability, Stiffness, and Stress"

**Content:**
- Show stress, displacement, and buckling mode field plots for S7/BFGS.
- **Structural interpretation:**
  - Buckling factor 24.95: well within safety margin.
  - Displacement 10.1 mm: acceptable for cooling tower structure.
  - Stress 4.13 MPa: far below concrete capacity.
- **Load decomposition:** Gravity-dominated response.
  - Median wind/gravity ratio: 4.8% for displacement, 4.3% for stress.
  - This dataset is self-weight-driven; wind controls only local behavior in specific geometries.

**Visual:** Abaqus field figures: `task7_s7_bfgs_stress.png`, `task7_s7_bfgs_displacement.png`, `task7_s7_bfgs_buckling_mode1.png`.

---

## Slide 45 — Bottleneck: Why S8 Matters

**Title:** "S8: The Industrial Stress Test — And What It Taught Us"

**Content:**
- **Setup:** Joint optimization, target volume = 90,000 m³, strict constructability (adjacent height ratio ≤ 1.8).
- **Result:** No algorithm achieved mathematical OR engineering feasibility (all 0%).
- **Why this is valuable:**
  - It reveals the limits of current penalty-based approach.
  - Conflicting constraints (larger volume + tight geometric limits) create unsolvable penalty landscapes.
- **Recommended improvements:**
  - Adaptive penalty weights that adjust during optimization.
  - Better initialization from known feasible regions.
  - Stronger regularization or problem reformulation.

---

## Slide 46 — Recommend Hybrid Workflow

**Title:** "No Free Lunch in Practice: Our Recommended Workflow"

**Content:**
Across Tasks 3, 4, 6, and 7:

| Method | Strengths | Weaknesses |
|---|---|---|
| **SA** | Robust on hard constraints, probabilistic escape from local minima | Slow convergence, high evaluation cost |
| **PSO** | Strong global exploration, population diversity | Underperforms in high-dimensional constrained penalty landscapes |
| **BFGS** | Ultra-efficient local convergence (~4,400 evals avg) | Fragile on non-smooth landscapes, sensitive to initialization |

**Recommended hybrid strategy:**
1. Use **SA** for broad exploration and feasible-region discovery.
2. Warm-start **BFGS** on the best compliant candidates for rapid refinement.
3. Validate final designs with **Abaqus** structural analysis.

---

## Slide 47 — Group Process: How We Worked

**Title:** "Behind the Results: Our Team & Process"

**Content:**
- **Team Montrésor:** 4 members with complementary roles.
  - **Achille** (Group Lead) — project management, task assignment, reports, coordination.
  - **Paul** (Technical Strategist) — project structure, environment setup, coding standards, module integration.
  - **Eneko** (Technical Designer & Quality Analyst) — code review, statistical analysis, graphs & tables.
  - **Aiert** (Technical Lead & System Architect) — math-to-code translation, solver implementation, optimization integration.

**Methodology:**
- Git version control throughout the project.
- Modular architecture: `src/` core library (algorithms, benchmarks, visualization) + isolated `tasks/` folders.
- Universal entry point: `python run_project.py --task all` for full reproducibility.
- Deterministic seeding for fair cross-algorithm comparison.
- Automated LaTeX report generation pipeline for Task 6 and Task 7.

**Tools:** Python 3, NumPy, Matplotlib, Plotly, Abaqus, LaTeX, Git.

---

## Slide 48 — Project Timeline

**Title:** "Our Journey: October 2025 — May 2026"

**Content:**

| Period | Tasks | What We Did |
|---|---|---|
| Oct–Nov 2025 | 1–2 | Team formation, role allocation, project setup |
| Dec 2025 | 3 | SA vs PSO implementation & benchmark comparison |
| Jan 2026 | 4–5 | BFGS implementation + interim in-person presentation |
| Feb–Mar 2026 | 6 | Cooling tower optimization: 8 scenarios, 240 runs |
| Mar 2026 | 7 | Abaqus wind load analysis: 24 towers, 96 solved jobs |
| Mar–Apr 2026 | 8 | Individual reflective self-assessments |
| May 2026 | 9 | **Final presentation (today)** |

**Visual:** Horizontal timeline graphic.

---

## Slide 49 — Conclusions

**Title:** "What We Learned"

**Content:**
- Successfully implemented and compared **stochastic** (SA, PSO) and **deterministic** (BFGS) optimization from scratch in Python.
- Applied the algorithms to both **mathematical benchmarks** and a **real engineering problem** (cooling tower).
- **Cooling tower results:** BFGS achieved 73.8% math compliance and 61.3% engineering feasibility across 8 scenarios.
- **Abaqus validation:** S7/BFGS emerged as global leader (weighted score 6.75), but convergence work continues.
- **Distinction between mathematical compliance and engineering feasibility is critical.** Optimization without real-world constraints produces non-buildable designs.
- **No single algorithm wins everywhere.** The "No Free Lunch" theorem holds in practice.
- **Hybrid SA → BFGS workflow** recommended for industrial use.
- All code, data, and reports are fully reproducible.

---

## Slide 50 — Thank You / Questions

**Title:** "Thank You — Questions?"

---

## Slide 51 (FINAL) — POSTER SLIDE

This must use the provided poster template `CSTE_Group_Project_Poster_Template.pptx` as the basis.

### Poster Concept: "Two Towers, One Lesson"

**Title:**  
**"Optimization Quality Depends on Boundary Conditions — Not Just Objectives"**

**Layout:**

```
┌────────────────────────────────────────────────────────────────┐
│  MSc Computational Software Techniques in Engineering           │
│  ESTIA Group Project 2025/26  |  Team Montrésor                │
├──────────────────┬─────────────────────┬───────────────────────┤
│                  │                     │                       │
│   FEASIBLE       │    KEY MESSAGE      │   RELAXED (UNCONSTR.)│
│   Tower 3D       │                     │   Tower 3D            │
│   (blue)         │  "Without explicit  │   (red)               │
│                  │   feasibility       │                       │
│   ✓ Volume OK    │   constraints,      │   ✗ Lower area BUT   │
│   ✓ Shape OK     │   optimizers find   │   ✗ Volume violation │
│   ✓ Smooth       │   minimal but       │   ✗ Shape violation  │
│                  │   unbuildable       │   ✗ Not realistic    │
│  "BUILDABLE"     │   geometries."      │  "NON-BUILDABLE"     │
│                  │                     │                       │
├──────────────────┴─────────────────────┴───────────────────────┤
│                                                                 │
│  RESULTS (240 Cooling Tower Runs, 3 Algorithms, 8 Scenarios)    │
│  ┌───────────────┬──────────────────┬─────────────────────────┐ │
│  │ BFGS: 73.8%   │ SA: 71.2% math  │ PSO: 50.0% math        │ │
│  │ math compliance│ compliance       │ compliance              │ │
│  │ 61.3% eng feas │ 50.0% eng feas  │ 7.5% eng feas          │ │
│  │ 4,408 evals    │ 11,500 evals    │ 11,500 evals           │ │
│  └───────────────┴──────────────────┴─────────────────────────┘ │
│                                                                 │
│  ABAQUS WINNER: S7/BFGS — Weighted score 6.75                  │
│  RECOMMENDATION: SA exploration → BFGS refinement → Abaqus     │
│                                                                 │
│  Achille | Paul | Eneko | Aiert            GitHub: [repo]       │
└────────────────────────────────────────────────────────────────┘
```

**Visual style:**
- Clean, professional, blue/gray academic color scheme.
- Centerpiece: side-by-side 3D tower renders (feasible constrained vs relaxed over-optimized).
- Minimal text — let the imagery and numbers tell the story.
- Technical but accessible to a non-specialist audience.

**Source figures:** Run the poster exploration script to generate the 3D contrast figures:
```bash
python tasks/Task_09_Presentation/task6_poster_exploration/run_poster_exploration.py
```
Outputs: `profile_contrast.png`, `tower_contrast_3d.png` in `results/figures/`.
