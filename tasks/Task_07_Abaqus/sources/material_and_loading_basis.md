# Task 7 Material and Loading Basis

This note fixes the baseline assumptions used by the Task 7 Abaqus workflow. The goal is a consistent comparative study of three Task 6 towers, not a certification-grade structural design.

## 1. Equivalent material model

The baseline shell material is modeled as linear-elastic equivalent reinforced concrete with:

- Young's modulus: `E = 33 GPa`
- Poisson ratio: `nu = 0.20`
- Density: `rho = 2500 kg/m^3`

Reasoning:

- `E = 33 GPa` is consistent with typical `E_cm` values used for normal-weight structural concrete around class `C30/37`.
- `nu = 0.20` is a standard elastic-analysis value for concrete.
- `rho = 2500 kg/m^3` is a common design density for reinforced concrete.

These values are appropriate for a first-pass shell comparison where the aim is to rank candidate geometries under common assumptions.

## 2. Shell thickness

The baseline shell thickness is taken as:

- `t = 0.20 m`

Reasoning:

- Hyperbolic reinforced-concrete cooling towers are thin-shell structures, and published references commonly report shell thicknesses in the order of a few tenths of a meter.
- The Task 6 tower is an academic geometry with total height around `36.5 m`, much smaller than full industrial natural-draft towers. A constant `0.20 m` shell is therefore treated as a representative equivalent thickness for comparative analysis, not as a final construction detail.

## 3. Wind load idealization

The Task 7 assignment specifies wind from one direction only. The baseline load case is therefore:

- one representative reference wind speed: `V = 30 m/s`
- air density: `rho_air = 1.25 kg/m^3`
- reference dynamic pressure:

`q = 0.5 * rho_air * V^2 = 562.5 Pa`

To keep the study implementable and comparable in Abaqus, the external pressure is distributed around the shell circumference using the simplified coefficient law:

`Cp(theta) = clip(0.8 cos(theta), -0.5, 0.8)`

and sector pressure:

`p(theta) = q * Cp(theta)`

This gives a directional windward-to-leeward pressure field with suction on the rear sectors. It is a comparative engineering load model, not a site-specific wind-code envelope.

## 4. Why this is acceptable for Task 7

The assignment wording is short and does not prescribe:

- a site location,
- a specific wind code,
- nonlinear concrete behavior,
- or detailed reinforcement modeling.

The chosen basis is therefore intentionally restrained:

- enough realism to compare the three towers structurally,
- simple enough to run robustly in Abaqus Learning Edition,
- and explicit enough that it can later be upgraded if site-specific wind data or a more detailed shell model is required.

## 5. Public reference trail

These values are aligned with standard structural-analysis practice and public references such as:

1. Eurocode-based concrete property summaries for normal-weight structural concrete.
2. Public cooling-tower shell references describing reinforced-concrete hyperbolic shells with thicknesses in the decimeter range.
3. Classical dynamic-pressure relation `q = 0.5 rho V^2` for wind loading.

The report should cite the specific sources used in the final write-up, but the workflow itself fixes the above values so the Abaqus comparisons remain reproducible.
