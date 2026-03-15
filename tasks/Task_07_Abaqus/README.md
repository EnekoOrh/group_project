# Task 7: Abaqus Wind Loading on Cooling Towers

This workspace turns selected Task 6 cooling-tower geometries into Abaqus shell models for one-direction wind analysis.

## Scope

Task 7 is treated as the structural follow-on to Task 6:

- input geometries come from Task 6 results,
- three candidate towers are compared under the same assumptions,
- the baseline analysis is `static response + linear buckling`,
- the output is a compact comparison report that also feeds Task 9 visuals.

Selected towers:

- `radius_baseline`: `S2 / BFGS`
- `height_variant`: `S6 / BFGS`
- `joint_reference`: `S7 / BFGS`

## Structure

```text
Task_07_Abaqus/
  config/
    study_config.json
  sources/
    material_and_loading_basis.md
  candidates/
    selected_cases.json
    candidate_summary.csv
    profiles/
  scripts/
    export_task6_candidates.py
    build_abaqus_inputs.py
    postprocess_odb.py
    export_odb_fields.py
    plot_abaqus_fields.py
    generate_report.py
    run_task7_pipeline.ps1
  results/
    data/
    figures/
    inputs/
    jobs/
    report/
  Task7_Report.md
```

## Baseline assumptions

The default study assumptions are fixed in:

- `config/study_config.json`
- `sources/material_and_loading_basis.md`

Baseline values:

- equivalent reinforced concrete: `E = 33 GPa`, `nu = 0.20`, `rho = 2500 kg/m^3`
- shell thickness: `0.20 m`
- one representative wind case: `V = 30 m/s`
- dynamic pressure: `562.5 Pa`
- circumferential pressure model: `Cp(theta) = clip(0.8 cos(theta), -0.5, 0.8)`

## Workflow

Run the full Task 7 pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1
```

Useful modes:

```powershell
# Export Task 6 cases and build Abaqus input files only
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1 -BuildOnly

# Skip Abaqus solves and regenerate report/plots from existing extracted data
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1 -SkipAbaqus
```

## Expected outputs

Key outputs after a successful run:

- `tasks/Task_07_Abaqus/candidates/selected_cases.json`
- `tasks/Task_07_Abaqus/results/inputs/*.inp`
- `tasks/Task_07_Abaqus/results/data/abaqus_summary.csv`
- `tasks/Task_07_Abaqus/results/data/mesh_sensitivity.csv`
- `tasks/Task_07_Abaqus/results/data/fields/*`
- `tasks/Task_07_Abaqus/results/figures/candidate_profiles.png`
- `tasks/Task_07_Abaqus/results/figures/comparison_metrics.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_stress.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_displacement.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_buckling_mode1.png`
- `tasks/Task_07_Abaqus/results/report/Task7_Report.md`
- `tasks/Task_07_Abaqus/Task7_Report.md`

## Notes

- `results/jobs/` and solver scratch files are ignored because Abaqus creates large transient artifacts.
- Static, displacement, and buckling figures are rendered from exported Abaqus ODB fields in Python, not from Abaqus GUI screenshots.
- The workflow is comparative and academic. If site-specific wind actions or detailed reinforced-concrete behavior are later required, the load model and material model should be upgraded rather than patched ad hoc.
