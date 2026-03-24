# Task 7 Report: Abaqus Wind Loading on Cooling Towers

## 1. Objective

This Task 7 study expands the previous three-tower screening into a full structural comparison across the eight Task 6 scenarios and the three optimization algorithms SA, PSO, and BFGS.
The engineering objective is to identify which Task 6 concepts remain convincing once the towers are checked under self-weight, one-direction wind loading, and linear buckling in Abaqus.

## 2. Selection Rule and Scenario Matrix

The Task 7 matrix contains **24 refined presentation models**. For scenarios S1 to S7, the preferred choice is the lowest-area engineering-feasible Task 6 run for each scenario/algorithm pair. If no engineering-feasible run exists, the workflow falls back to the lowest-area mathematically compliant run. If neither exists, the best penalized run is kept only as a warning case. Scenario S8 is intentionally carried as warning-only because it has no compliant Task 6 solution in the current study.

Across the 24 selected cases, there are **16 engineering-feasible selections**, **3 mathematical fallbacks**, and **5 warning-only cases**.

| Case | Status | Selection basis | Task 6 run | Decision mode | Task 6 area (m^2) |
|---|---|---|---|---|---|
| S1 / BFGS | Warning case | best_penalized_warning | 6 | radii | 7782.01 |
| S1 / PSO | Engineering-feasible | lowest_area_engineering_feasible | 7 | radii | 7804.56 |
| S1 / SA | Engineering-feasible | lowest_area_engineering_feasible | 5 | radii | 7780.73 |
| S2 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 2 | radii | 7741.67 |
| S2 / PSO | Math fallback | lowest_area_mathematical_fallback | 0 | radii | 9075.03 |
| S2 / SA | Engineering-feasible | lowest_area_engineering_feasible | 5 | radii | 7744.01 |
| S3 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 6 | radii | 7741.69 |
| S3 / PSO | Engineering-feasible | lowest_area_engineering_feasible | 6 | radii | 7743.34 |
| S3 / SA | Engineering-feasible | lowest_area_engineering_feasible | 6 | radii | 7742.77 |
| S4 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 1 | radii | 7779.85 |
| S4 / PSO | Warning case | best_penalized_warning | 1 | radii | 7931.66 |
| S4 / SA | Engineering-feasible | lowest_area_engineering_feasible | 7 | radii | 7769.88 |
| S5 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 0 | heights | 7742.56 |
| S5 / PSO | Engineering-feasible | lowest_area_engineering_feasible | 6 | heights | 7747.93 |
| S5 / SA | Engineering-feasible | lowest_area_engineering_feasible | 6 | heights | 7749.52 |
| S6 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 1 | heights | 7741.72 |
| S6 / PSO | Engineering-feasible | lowest_area_engineering_feasible | 3 | heights | 7748.25 |
| S6 / SA | Engineering-feasible | lowest_area_engineering_feasible | 0 | heights | 7783.24 |
| S7 / BFGS | Engineering-feasible | lowest_area_engineering_feasible | 1 | joint | 7740.66 |
| S7 / PSO | Math fallback | lowest_area_mathematical_fallback | 9 | joint | 8324.25 |
| S7 / SA | Math fallback | lowest_area_mathematical_fallback | 9 | joint | 7890.85 |
| S8 / BFGS | Warning case | best_penalized_warning | 5 | joint | 7813.24 |
| S8 / PSO | Warning case | best_penalized_warning | 8 | joint | 7899.07 |
| S8 / SA | Warning case | best_penalized_warning | 0 | joint | 7718.27 |

![Task 7 candidate profiles](results/figures/candidate_profiles.png)

## 3. Material, Loading, and CAE Setup

The structural baseline keeps the same assumptions for every model: equivalent reinforced concrete with `E = 33 GPa`, `nu = 0.20`, and `rho = 2500 kg/m^3`, together with a constant shell thickness of `0.20 m`.
The reference wind speed is `30.0 m/s`, which gives a dynamic pressure of `562.5 Pa` through `q = 0.5 rho V^2`.
Wind loading is interpreted as one-direction external action: windward sectors receive positive external pressure and leeward sectors receive suction. This is not internal cabin-style pressurization.
The directional pressure model is:
$$
C_p(\theta) = \mathrm{clip}(0.8\cos\theta, -0.5, 0.8), \quad p(\theta) = q\,C_p(\theta)
$$
Task 7 uses a strict `Y`-up coordinate convention: `Y` is vertical, the horizontal wind plane is `X/Z`, and self-weight acts along `-Y`.
The clean user-facing Abaqus/CAE database contains the **24 refined models only**, with one gravity load and one wind load per model. The automated solver decks still use the same circumferential pressure law in sectorized form for coarse-versus-refined verification.
The refined study uses one uniform high mesh policy for all 24 towers (`40` circumferential divisions, `2` axial subdivisions per segment). Refined node counts range from `840` to `1000` nodes, which keeps every case within the Abaqus Learning Edition limit while using the highest common density.
Solver-deck mesh compliance (`mesh_summary.csv`) is tracked separately from CAE integrity: refined solver decks span `840` to `1000` nodes and remain within the Abaqus LE cap.
CAE-native integrity (`cae_integrity_audit.json`) reports a model-node range of `840` to `1075`. This audit validates model-tree integrity and CAE consistency, not LE solve-cap compliance.
A deterministic input-deck audit runs before solving and confirms that all 48 decks use identical material constants, gravity definition, and wind-pressure sector logic.
The towers look squat in Abaqus because the geometry is genuinely squat: the Task 6 cooling towers are about `36.5 m` tall for a base diameter of about `78.6 m`, so the aspect ratio is below one. This is source geometry, not an Abaqus distortion.

## 4. Geometry Integrity Check

The Task 7 generator preserves the exact Task 6 meridian coordinates. A geometry verification pass is written before solving and checks the mesh rings against the selected Task 6 source profile for every coarse and refined job.

| Mesh | Max total-height diff (m) | Max ring-radius diff (m) | Max ring-z diff (m) |
|---|---|---|---|
| coarse | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| refined | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

## 5. Refined Structural Comparison Across All 24 Cases

| Case | Status | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|
| S1 / BFGS | Warning case | 7782.01 | 4.264 | 2.262 | 20.907 |
| S1 / PSO | Engineering-feasible | 7804.56 | 4.126 | 2.244 | 21.300 |
| S1 / SA | Engineering-feasible | 7780.73 | 4.627 | 2.368 | 21.481 |
| S2 / BFGS | Engineering-feasible | 7741.67 | 10.202 | 3.867 | 23.563 |
| S2 / PSO | Math fallback | 9075.03 | 94.040 | 10.268 | -3.229 |
| S2 / SA | Engineering-feasible | 7744.01 | 9.968 | 3.818 | 24.549 |
| S3 / BFGS | Engineering-feasible | 7741.69 | 10.153 | 3.834 | 23.529 |
| S3 / PSO | Engineering-feasible | 7743.34 | 9.850 | 3.794 | 23.496 |
| S3 / SA | Engineering-feasible | 7742.77 | 9.036 | 3.526 | 23.834 |
| S4 / BFGS | Engineering-feasible | 7779.85 | 3.831 | 2.282 | 20.267 |
| S4 / PSO | Warning case | 7931.66 | 12.073 | 4.541 | 22.491 |
| S4 / SA | Engineering-feasible | 7769.88 | 4.287 | 2.296 | 19.753 |
| S5 / BFGS | Engineering-feasible | 7742.56 | 8.695 | 4.148 | 22.725 |
| S5 / PSO | Engineering-feasible | 7747.93 | 11.481 | 4.941 | 19.566 |
| S5 / SA | Engineering-feasible | 7749.52 | 10.229 | 4.642 | 23.232 |
| S6 / BFGS | Engineering-feasible | 7741.72 | 10.522 | 5.312 | 24.772 |
| S6 / PSO | Engineering-feasible | 7748.25 | 11.907 | 6.705 | 21.611 |
| S6 / SA | Engineering-feasible | 7783.24 | 14.270 | 7.690 | 19.128 |
| S7 / BFGS | Engineering-feasible | 7740.66 | 10.105 | 4.134 | 24.948 |
| S7 / PSO | Math fallback | 8324.25 | 28.467 | 9.258 | 23.712 |
| S7 / SA | Math fallback | 7890.85 | 12.541 | 6.711 | 26.587 |
| S8 / BFGS | Warning case | 7813.24 | 10.319 | 5.446 | 19.273 |
| S8 / PSO | Warning case | 7899.07 | 9.235 | 4.493 | 27.335 |
| S8 / SA | Warning case | 7718.27 | 8.163 | 4.955 | 23.837 |

![Task 7 refined comparison metrics](results/figures/comparison_metrics.png)

## 6. Methodological Clarity and Global Ranking

### 6.1 Methodological Clarity

The global ranking is a rank-based weighted score on the refined results, designed to stay robust when one or two cases have extreme values.
For each metric, rank `1` is best and rank `24` is worst, with deterministic tie-breaks by scenario/algorithm identifiers.

$$
R_{k,i} = \\text{rank of case } i \\text{ on metric } k
$$

$$
S_i = 0.45 R_{\\mathrm{buckling},i} + 0.25 R_{\\mathrm{disp},i} + 0.20 R_{\\mathrm{stress},i} + 0.10 R_{\\mathrm{area},i}
$$

$$
J_i = S_i + P_i, \\quad P_i \\in \\{0,10,20\\}
$$

Penalty points are `0` for engineering-feasible, `10` for mathematical fallback, and `20` for warning cases. The best model is the one with the lowest `J_i`.

### 6.2 Global Weighted Top-3

| Rank | Case | Status | Weighted score | Penalty | Buckling factor | Max disp. (mm) | Max stress (MPa) | Area (m^2) |
|---|---|---|---|---|---|---|---|---|
| 1 | S7 / BFGS | Engineering-feasible | 6.750 | 0.0 | 24.948 | 10.105 | 4.134 | 7740.66 |
| 2 | S3 / SA | Engineering-feasible | 7.050 | 0.0 | 23.834 | 9.036 | 3.526 | 7742.77 |
| 3 | S2 / SA | Engineering-feasible | 7.500 | 0.0 | 24.549 | 9.968 | 3.818 | 7744.01 |

![Task 7 weighted ranking](results/figures/task7_weighted_ranking.png)

### 6.3 Top-3 by Criterion

#### 6.3.1 Raw Top-3 (all statuses)

This raw criterion table keeps all statuses visible, so warning and fallback cases can appear if they are numerically extreme on a single indicator.

| Criterion | Rank | Case | Status | Value |
|---|---|---|---|---|
| Buckling factor | 1 | S8 / PSO | Warning case | 27.335 |
| Buckling factor | 2 | S7 / SA | Math fallback | 26.587 |
| Buckling factor | 3 | S7 / BFGS | Engineering-feasible | 24.948 |
| Max displacement | 1 | S4 / BFGS | Engineering-feasible | 3.831 mm |
| Max displacement | 2 | S1 / PSO | Engineering-feasible | 4.126 mm |
| Max displacement | 3 | S1 / BFGS | Warning case | 4.264 mm |
| Max stress | 1 | S1 / PSO | Engineering-feasible | 2.244 MPa |
| Max stress | 2 | S1 / BFGS | Warning case | 2.262 MPa |
| Max stress | 3 | S4 / BFGS | Engineering-feasible | 2.282 MPa |
| Task 6 area | 1 | S8 / SA | Warning case | 7718.27 m^2 |
| Task 6 area | 2 | S7 / BFGS | Engineering-feasible | 7740.66 m^2 |
| Task 6 area | 3 | S2 / BFGS | Engineering-feasible | 7741.67 m^2 |

![Task 7 criterion leaders](results/figures/task7_criterion_top3.png)

#### 6.3.2 Engineering-eligible Top-3

This decision-grade table filters to engineering-feasible entries only.

| Criterion | Rank | Case | Status | Value |
|---|---|---|---|---|
| Buckling factor | 1 | S7 / BFGS | Engineering-feasible | 24.948 |
| Buckling factor | 2 | S6 / BFGS | Engineering-feasible | 24.772 |
| Buckling factor | 3 | S2 / SA | Engineering-feasible | 24.549 |
| Max displacement | 1 | S4 / BFGS | Engineering-feasible | 3.831 mm |
| Max displacement | 2 | S1 / PSO | Engineering-feasible | 4.126 mm |
| Max displacement | 3 | S4 / SA | Engineering-feasible | 4.287 mm |
| Max stress | 1 | S1 / PSO | Engineering-feasible | 2.244 MPa |
| Max stress | 2 | S4 / BFGS | Engineering-feasible | 2.282 MPa |
| Max stress | 3 | S4 / SA | Engineering-feasible | 2.296 MPa |
| Task 6 area | 1 | S7 / BFGS | Engineering-feasible | 7740.66 m^2 |
| Task 6 area | 2 | S2 / BFGS | Engineering-feasible | 7741.67 m^2 |
| Task 6 area | 3 | S3 / BFGS | Engineering-feasible | 7741.69 m^2 |

![Task 7 engineering criterion leaders](results/figures/task7_criterion_top3_engineering.png)

## 7. Scenario-by-Scenario Comparison

### S1

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Warning case | 30.850 | 7782.01 | 4.264 | 2.262 | 20.907 |
| PSO | Engineering-feasible | 10.150 | 7804.56 | 4.126 | 2.244 | 21.300 |
| SA | Engineering-feasible | 10.950 | 7780.73 | 4.627 | 2.368 | 21.481 |

**Winner:** S1 / PSO with weighted score `10.150`.
The score gap to second place (S1 / SA) is `0.800` points.
**Spread/Risk:** displacement spans `4.126` to `4.627` mm, stress spans `2.244` to `2.368` MPa, and buckling spans `20.907` to `21.481`.
**Status caveat:** warning-only entries are present for BFGS and are not decision-grade designs.
**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs.

### S2

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 9.850 | 7741.67 | 10.202 | 3.867 | 23.563 |
| PSO | Math fallback | 34.000 | 9075.03 | 94.040 | 10.268 | -3.229 |
| SA | Engineering-feasible | 7.500 | 7744.01 | 9.968 | 3.818 | 24.549 |

**Winner:** S2 / SA with weighted score `7.500`.
The score gap to second place (S2 / BFGS) is `2.350` points.
**Spread/Risk:** displacement spans `9.968` to `94.040` mm, stress spans `3.818` to `10.268` MPa, and buckling spans `-3.229` to `24.549`.
**Status caveat:** mathematical fallback entries are present for PSO.
**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs.

### S3

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 9.950 | 7741.69 | 10.153 | 3.834 | 23.529 |
| PSO | Engineering-feasible | 9.650 | 7743.34 | 9.850 | 3.794 | 23.496 |
| SA | Engineering-feasible | 7.050 | 7742.77 | 9.036 | 3.526 | 23.834 |

**Winner:** S3 / SA with weighted score `7.050`.
The score gap to second place (S3 / PSO) is `2.600` points.
**Spread/Risk:** displacement spans `9.036` to `10.153` mm, stress spans `3.526` to `3.834` MPa, and buckling spans `23.496` to `23.834`.
**Status caveat:** all three entries are engineering-feasible.
**Critical note:** the winner remains robust inside this scenario under the current weighted scoring contract.

### S4

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 10.800 | 7779.85 | 3.831 | 2.282 | 20.267 |
| PSO | Warning case | 36.300 | 7931.66 | 12.073 | 4.541 | 22.491 |
| SA | Engineering-feasible | 12.100 | 7769.88 | 4.287 | 2.296 | 19.753 |

**Winner:** S4 / BFGS with weighted score `10.800`.
The score gap to second place (S4 / SA) is `1.300` points.
**Spread/Risk:** displacement spans `3.831` to `12.073` mm, stress spans `2.282` to `4.541` MPa, and buckling spans `19.753` to `22.491`.
**Status caveat:** warning-only entries are present for PSO and are not decision-grade designs.
**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs.

### S5

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 10.600 | 7742.56 | 8.695 | 4.148 | 22.725 |
| PSO | Engineering-feasible | 18.150 | 7747.93 | 11.481 | 4.941 | 19.566 |
| SA | Engineering-feasible | 13.350 | 7749.52 | 10.229 | 4.642 | 23.232 |

**Winner:** S5 / BFGS with weighted score `10.600`.
The score gap to second place (S5 / SA) is `2.750` points.
**Spread/Risk:** displacement spans `8.695` to `11.481` mm, stress spans `4.148` to `4.941` MPa, and buckling spans `19.566` to `23.232`.
**Status caveat:** all three entries are engineering-feasible.
**Critical note:** the winner remains robust inside this scenario under the current weighted scoring contract.

### S6

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 10.150 | 7741.72 | 10.522 | 5.312 | 24.772 |
| PSO | Engineering-feasible | 16.600 | 7748.25 | 11.907 | 6.705 | 21.611 |
| SA | Engineering-feasible | 21.950 | 7783.24 | 14.270 | 7.690 | 19.128 |

**Winner:** S6 / BFGS with weighted score `10.150`.
The score gap to second place (S6 / PSO) is `6.450` points.
**Spread/Risk:** displacement spans `10.522` to `14.270` mm, stress spans `5.312` to `7.690` MPa, and buckling spans `19.128` to `24.772`.
**Status caveat:** all three entries are engineering-feasible.
**Critical note:** the winner remains robust inside this scenario under the current weighted scoring contract.

### S7

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Engineering-feasible | 6.750 | 7740.66 | 10.105 | 4.134 | 24.948 |
| PSO | Math fallback | 26.250 | 8324.25 | 28.467 | 9.258 | 23.712 |
| SA | Math fallback | 22.350 | 7890.85 | 12.541 | 6.711 | 26.587 |

**Winner:** S7 / BFGS with weighted score `6.750`.
The score gap to second place (S7 / SA) is `15.600` points.
**Spread/Risk:** displacement spans `10.105` to `28.467` mm, stress spans `4.134` to `9.258` MPa, and buckling spans `23.712` to `26.587`.
**Status caveat:** mathematical fallback entries are present for PSO, SA.
**Critical note:** scenario fairness is reduced by mixed compliance statuses; ranking penalties keep the comparison visible but do not make warning/fallback cases equivalent to engineering-feasible designs.

### S8

| Algorithm | Status | Weighted score | Task 6 area (m^2) | Max disp. (mm) | Max stress (MPa) | Buckling factor |
|---|---|---|---|---|---|---|
| BFGS | Warning case | 39.600 | 7813.24 | 10.319 | 5.446 | 19.273 |
| PSO | Warning case | 27.400 | 7899.07 | 9.235 | 4.493 | 27.335 |
| SA | Warning case | 27.700 | 7718.27 | 8.163 | 4.955 | 23.837 |

**Winner:** S8 / PSO with weighted score `27.400`.
The score gap to second place (S8 / SA) is `0.300` points.
**Spread/Risk:** displacement spans `8.163` to `10.319` mm, stress spans `4.493` to `5.446` MPa, and buckling spans `19.273` to `27.335`.
**Status caveat:** warning-only entries are present for BFGS, PSO, SA and are not decision-grade designs.
**Critical note:** this scenario is tightly clustered, so small modeling shifts can reorder first and second place.

## 8. Convergence Summary

The verification workflow solves both coarse and refined meshes for every case. The acceptance criteria are: displacement change below `5.0%`, stress change below `5.0%`, and first buckling-factor change below `10.0%`.

At the current stage, **0 of 24 comparison pairs** satisfy all three convergence criteria.
Because the study is currently at `0/24` all-pass convergence, the ranking should be interpreted as comparative screening quality, not final structural qualification.

| Case | Status | Delta disp. (%) | Pass | Delta stress (%) | Pass | Delta buckling (%) | Pass | All pass |
|---|---|---|---|---|---|---|---|---|
| S1 / BFGS | Warning case | 8.66 | no | 3.95 | yes | -51.38 | no | no |
| S1 / PSO | Engineering-feasible | 17.03 | no | 8.32 | no | -51.39 | no | no |
| S1 / SA | Engineering-feasible | 9.59 | no | 2.24 | yes | -49.67 | no | no |
| S2 / BFGS | Engineering-feasible | 27.17 | no | 0.60 | yes | -48.05 | no | no |
| S2 / PSO | Math fallback | -11.47 | no | -60.11 | no | -5.64 | yes | no |
| S2 / SA | Engineering-feasible | 25.49 | no | -1.19 | yes | -46.75 | no | no |
| S3 / BFGS | Engineering-feasible | 26.57 | no | -0.73 | yes | -48.06 | no | no |
| S3 / PSO | Engineering-feasible | 27.31 | no | 1.44 | yes | -50.01 | no | no |
| S3 / SA | Engineering-feasible | 21.16 | no | -3.57 | yes | -46.37 | no | no |
| S4 / BFGS | Engineering-feasible | -5.76 | no | 7.01 | no | -51.16 | no | no |
| S4 / PSO | Warning case | 16.53 | no | -3.15 | yes | -158.30 | no | no |
| S4 / SA | Engineering-feasible | -7.69 | no | 1.43 | yes | -49.02 | no | no |
| S5 / BFGS | Engineering-feasible | -5.89 | no | 10.45 | no | -46.31 | no | no |
| S5 / PSO | Engineering-feasible | 1.17 | yes | 23.43 | no | -49.25 | no | no |
| S5 / SA | Engineering-feasible | -2.76 | yes | 5.00 | no | -45.85 | no | no |
| S6 / BFGS | Engineering-feasible | 2.08 | yes | 20.49 | no | -41.81 | no | no |
| S6 / PSO | Engineering-feasible | 17.42 | no | 39.74 | no | -54.85 | no | no |
| S6 / SA | Engineering-feasible | 15.36 | no | 38.65 | no | -55.49 | no | no |
| S7 / BFGS | Engineering-feasible | 17.31 | no | 0.69 | yes | -43.58 | no | no |
| S7 / PSO | Math fallback | 11.87 | no | -19.73 | no | -240.70 | no | no |
| S7 / SA | Math fallback | -6.17 | no | -19.68 | no | -63.19 | no | no |
| S8 / BFGS | Warning case | 2.71 | yes | 10.91 | no | -53.42 | no | no |
| S8 / PSO | Warning case | 7.91 | no | -0.58 | yes | -49.86 | no | no |
| S8 / SA | Warning case | 5.04 | no | 15.88 | no | -53.02 | no | no |

![Winner mesh comparison](results/figures/task7_s7_bfgs_mesh_comparison.png)

The global winner is therefore reported as **provisional**: the ranking is valid for comparative screening, but convergence evidence is not yet strong enough for a final structural sign-off.

## 9. Warning Cases from Scenario S8

Scenario S8 is kept intentionally as a warning family. None of the Task 6 S8 runs are mathematically compliant or engineering-feasible, so these Abaqus models are useful only as structural cautionary examples and not as candidate towers for recommendation.

![Scenario S8 warning metrics](results/figures/s8_warning_metrics.png)

## 10. Field Visualizations of the Global Top-3

All detailed field figures are rendered from actual Abaqus ODB data with Python, not from Abaqus screenshots.
Task 7 figures use a true-scale equal-axis policy (no axis stretching). This differs from some legacy Task 6 renderings that used a different plotting style.

### S7 / BFGS

![S7 / BFGS stress](results/figures/task7_s7_bfgs_stress.png)

![S7 / BFGS displacement](results/figures/task7_s7_bfgs_displacement.png)

![S7 / BFGS buckling](results/figures/task7_s7_bfgs_buckling_mode1.png)

### S3 / SA

![S3 / SA stress](results/figures/task7_s3_sa_stress.png)

![S3 / SA displacement](results/figures/task7_s3_sa_displacement.png)

![S3 / SA buckling](results/figures/task7_s3_sa_buckling_mode1.png)

### S2 / SA

![S2 / SA stress](results/figures/task7_s2_sa_stress.png)

![S2 / SA displacement](results/figures/task7_s2_sa_displacement.png)

![S2 / SA buckling](results/figures/task7_s2_sa_buckling_mode1.png)

## 11. Recommendation for Later Tasks

The current Task 7 recommendation is **S7 / BFGS** with a `provisional` status. Its weighted score is `6.750` and its refined response gives a first buckling factor of `24.948`, a maximum displacement of `10.105 mm`, and a maximum stress of `4.134 MPa`.

The main practical outcome for the project is that Task 6 geometric alternatives can now be compared with one consistent structural score, explicit warning handling, and scenario-level interpretation suitable for Task 9 communication.
