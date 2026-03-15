# Task 7 Report: Abaqus Wind Loading on Cooling Towers

## 1. Objective

This Task 7 study transfers three selected Task 6 cooling-tower geometries into Abaqus shell models in order to compare their structural response under self-weight and one-direction wind loading.
The engineering question is not which geometry had the lowest optimized shell area alone, but which geometry remains the most convincing once displacement, stress, and shell-buckling sensitivity are checked under a common structural load case.

## 2. Towers Carried Forward from Task 6

| Case ID | Task 6 source | Why it was kept |
|---|---|---|
| Radius baseline (S2 / BFGS) | S2 / BFGS | Radius-only optimized tower used as the clean Task 6 baseline candidate. |
| Height variant (S6 / BFGS) | S6 / BFGS | Height-only optimized tower used to test structural sensitivity to vertical segmentation. |
| Joint reference (S7 / BFGS) | S7 / BFGS | Joint radii-height optimized tower used as the main realistic reference for Task 7. |

![Task 7 candidate profiles](../figures/candidate_profiles.png)

## 3. Material and Loading Assumptions

- Equivalent reinforced concrete: `E = 33 GPa`, `nu = 0.20`, `rho = 2500 kg/m^3`.
- Constant shell thickness: `0.20 m`.
- Reference wind speed: `30.0 m/s`, giving `q = 562.5 Pa` from `q = 0.5 rho V^2`.
- Circumferential wind coefficient law: `clip(0.8*cos(theta), -0.5, 0.8)` with wind acting from the `+X` direction.
- Base boundary condition: fully fixed support ring at the tower base.
- Analysis sequence: static response under self-weight plus wind, followed by linear buckling on the preloaded state.

## 4. Abaqus Model Setup

Each tower is meshed as a full 3D shell surface generated from the Task 6 meridian profile. The study uses the same material, shell thickness, and wind basis for all three towers so the structural comparison remains controlled.
The comparison mesh uses `16` circumferential sectors and `1` axial subdivision per Task 6 frustum segment.

## 5. Results Summary

All figures below are rendered from actual Abaqus ODB field data exported with `abaqus python`; they are not screenshots from Abaqus/CAE.

| Tower | Task 6 area (m^2) | Max disp. (mm) | Max von Mises (MPa) | Base reaction resultant (kN) | First buckling factor |
|---|---:|---:|---:|---:|---:|
| Radius baseline (S2 / BFGS) | 7741.67 | 8.101 | 3.896 | 37503.03 | 45.080 |
| Height variant (S6 / BFGS) | 7741.72 | 10.407 | 4.444 | 37505.70 | 42.313 |
| Joint reference (S7 / BFGS) | 7740.66 | 8.690 | 4.138 | 37498.95 | 43.947 |

![Task 7 comparison metrics](../figures/comparison_metrics.png)

## 6. Field Visualizations

### Radius baseline (S2 / BFGS)

![Radius baseline (S2 / BFGS) stress](../figures/task7_radius_baseline_stress.png)

![Radius baseline (S2 / BFGS) displacement](../figures/task7_radius_baseline_displacement.png)

![Radius baseline (S2 / BFGS) buckling mode 1](../figures/task7_radius_baseline_buckling_mode1.png)

### Height variant (S6 / BFGS)

![Height variant (S6 / BFGS) stress](../figures/task7_height_variant_stress.png)

![Height variant (S6 / BFGS) displacement](../figures/task7_height_variant_displacement.png)

![Height variant (S6 / BFGS) buckling mode 1](../figures/task7_height_variant_buckling_mode1.png)

### Joint reference (S7 / BFGS)

![Joint reference (S7 / BFGS) stress](../figures/task7_joint_reference_stress.png)

![Joint reference (S7 / BFGS) displacement](../figures/task7_joint_reference_displacement.png)

![Joint reference (S7 / BFGS) buckling mode 1](../figures/task7_joint_reference_buckling_mode1.png)

## 7. Mesh Sensitivity

The reference mesh-sensitivity check was run on the joint-reference tower because it is the most realistic candidate carried over from Task 6.

| Metric | Baseline | Refined | Relative change (%) |
|---|---:|---:|---:|
| max_displacement_m | 0.00868998 | 0.0100617 | 15.79 |
| max_mises_pa | 4.13791e+06 | 4.11571e+06 | -0.54 |
| buckling_factor_1 | 43.947 | 28.071 | -36.13 |

The joint-reference mesh check showed a large change in first buckling factor when the mesh was refined. That means the current buckling ranking is informative for screening, but it is not yet converged strongly enough to be treated as a final design decision without at least one more refinement pass.

## 8. Recommendation

The recommended tower is **Radius baseline (S2 / BFGS)**. The selection rule is deliberately transparent: prioritize the highest first buckling factor, then the lower maximum displacement, then the lower maximum von Mises stress, and finally the lower Task 6 shell area.

Under that rule, `Radius baseline (S2 / BFGS)` ranked first with a first buckling factor of `45.080`, a maximum displacement of `8.101 mm`, and a maximum von Mises stress of `3.896 MPa`.

This recommendation should therefore be read as a **baseline screening result**, not as a mesh-converged final structural verdict.

## 9. Feedback into Task 6

The simpler radius-only tower outperformed the richer Task 6 variants structurally, which suggests Task 6 should keep a strong bias toward smooth radius control before adding extra height freedom.

The weakest buckling response came from the height-only wide-bound case, so future feasibility checks should continue to penalize abrupt height redistribution even when the geometry stays mathematically compliant.

## 10. Limitations and Next Steps

This is a comparative baseline study. The shell is linear elastic, the thickness is constant, and the wind field is represented by an explicit sector-based circumferential pressure law rather than a full code-based site model.
If the project is extended, the next upgrades should be: refined wind action based on a chosen design standard, shell-thickness sensitivity, and possibly geometric or material nonlinearity for the recommended tower only.
