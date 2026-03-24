# Task 7: Abaqus Wind Loading on Cooling Towers

This workspace turns the Task 6 cooling-tower optimization results into a full Task 7 structural comparison in Abaqus.

## Scope

The Task 7 study now compares the full `8 scenarios x 3 optimizers` matrix:

- scenarios: `S1` to `S8`
- optimizers: `SA`, `PSO`, `BFGS`
- total refined presentation models in the clean CAE: `24`
- total verification jobs in the automated solver path: `48` (`coarse + refined` for each case)

Selection rules:

- `S1` to `S7`: lowest-area engineering-feasible Task 6 run per scenario/optimizer pair
- fallback: lowest-area mathematically compliant run if no engineering-feasible run exists
- warning case: best penalized run if neither engineering-feasible nor mathematically compliant exists
- `S8`: warning-only structural comparison by design in this workflow

## Structure

```text
Task_07_Abaqus/
  config/
    study_config.json
  candidates/
    selected_cases.json
    candidate_summary.csv
    profiles/
  scripts/
    task7_common.py
    export_task6_candidates.py
    build_abaqus_inputs.py
    validate_input_contracts.py
    build_cae_study.py
    validate_cae_integrity.py
    postprocess_odb.py
    export_odb_fields.py
    plot_abaqus_fields.py
    generate_report.py
    run_task7_pipeline.ps1
  report_latex/
    main.tex
    latexmkrc
    refs.bib
    scripts/
    sections/generated/
  results/
    cae/
    data/
    figures/
    inputs/
    jobs/
    report/
  Task7_Report.md
  Task7_Montrésor_CoolingTowerWind.pdf
```

## Baseline assumptions

Defaults are fixed in [study_config.json](/C:/Users/achille.larregle/OneDrive%20-%20ESTIA/Bureau/CRANFIELD/CRANFIELD%20COURSE/GroupProject2026/group_project/tasks/Task_07_Abaqus/config/study_config.json):

- equivalent reinforced concrete: `E = 33 GPa`, `nu = 0.20`, `rho = 2500 kg/m^3`
- shell thickness: `0.20 m`
- reference wind speed: `30 m/s`
- dynamic pressure: `562.5 Pa`
- global axis convention: `Y` is vertical, `X/Z` is horizontal
- gravity direction: `[0, -1, 0]` (self-weight downward in `-Y`)
- solver pressure law: `Cp(theta) = clip(0.8 cos(theta), -0.5, 0.8)`
- clean CAE pressure display: continuous field form of the same law
- coarse mesh: `16` circumferential divisions, `1` axial subdivision per segment
- refined mesh: `40` circumferential divisions, `2` axial subdivisions per segment (uniform high mesh near the Abaqus LE node cap)

## Workflow

Full run:

```powershell
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1
```

The pipeline now validates every generated `.inp` immediately after deck creation and fails fast if material, gravity, or wind contracts drift from the config.
It also enforces CAE integrity with a candidate-to-canonical promotion flow:
- preserve canonical CAE to `Task7_Montresor_WindStudy_last_good*.cae`,
- quarantine stale `*_staged.cae` and `ABQcae*.exception` artifacts,
- build `Task7_Montresor_WindStudy_candidate.cae`,
- validate candidate contracts and mesh density headlessly,
- promote candidate to canonical only after successful validation.

Useful modes:

```powershell
# Export Task 6 selections, build 48 input decks, and build the clean CAE only
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1 -BuildOnly -SkipAbaqus -SkipReport -SkipPlots -SkipLatex

# Reuse existing ODB results and rebuild figures/report/PDF only
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1 -SkipAbaqus -SkipCae

# Reuse existing ODB results but skip the final PDF compile
powershell -ExecutionPolicy Bypass -File tasks/Task_07_Abaqus/scripts/run_task7_pipeline.ps1 -SkipAbaqus -SkipCae -SkipLatex
```

## Expected outputs

Key outputs after a successful run:

- `tasks/Task_07_Abaqus/candidates/selected_cases.json`
- `tasks/Task_07_Abaqus/results/data/job_manifest.json`
- `tasks/Task_07_Abaqus/results/data/geometry_verification.csv`
- `tasks/Task_07_Abaqus/results/data/input_contract_audit.csv`
- `tasks/Task_07_Abaqus/results/data/cae_integrity_audit.csv`
- `tasks/Task_07_Abaqus/results/data/cae_integrity_audit.json`
- `tasks/Task_07_Abaqus/results/data/abaqus_summary.csv`
- `tasks/Task_07_Abaqus/results/data/mesh_sensitivity.csv`
- `tasks/Task_07_Abaqus/results/data/weighted_ranking.csv`
- `tasks/Task_07_Abaqus/results/data/criterion_top3.csv`
- `tasks/Task_07_Abaqus/results/data/criterion_top3_engineering.csv`
- `tasks/Task_07_Abaqus/results/data/report_consistency_audit.csv`
- `tasks/Task_07_Abaqus/results/data/report_consistency_audit.json`
- `tasks/Task_07_Abaqus/results/data/fields/*`
- `tasks/Task_07_Abaqus/results/cae/Task7_Montresor_WindStudy.cae`
- `tasks/Task_07_Abaqus/results/cae/Task7_Montresor_WindStudy_last_good.cae` (latest backup of canonical before regeneration)
- `tasks/Task_07_Abaqus/results/cae/quarantine/<timestamp>/...` (staged/corrupt artifacts moved aside)
- `tasks/Task_07_Abaqus/results/inputs/task7_*_coarse.inp`
- `tasks/Task_07_Abaqus/results/inputs/task7_*_refined.inp`
- `tasks/Task_07_Abaqus/results/figures/candidate_profiles.png`
- `tasks/Task_07_Abaqus/results/figures/comparison_metrics.png`
- `tasks/Task_07_Abaqus/results/figures/s8_warning_metrics.png`
- `tasks/Task_07_Abaqus/results/figures/task7_weighted_ranking.png`
- `tasks/Task_07_Abaqus/results/figures/task7_criterion_top3.png`
- `tasks/Task_07_Abaqus/results/figures/task7_criterion_top3_engineering.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_stress.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_displacement.png`
- `tasks/Task_07_Abaqus/results/figures/task7_*_buckling_mode1.png`
- `tasks/Task_07_Abaqus/results/report/Task7_Report.md`
- [Task7_Report.md](/C:/Users/achille.larregle/OneDrive%20-%20ESTIA/Bureau/CRANFIELD/CRANFIELD%20COURSE/GroupProject2026/group_project/tasks/Task_07_Abaqus/Task7_Report.md)
- [Task7_Montrésor_CoolingTowerWind.pdf](/C:/Users/achille.larregle/OneDrive%20-%20ESTIA/Bureau/CRANFIELD/CRANFIELD%20COURSE/GroupProject2026/group_project/tasks/Task_07_Abaqus/Task7_Montr%C3%A9sor_CoolingTowerWind.pdf)

## Open directly in Abaqus

Primary file to open in Abaqus/CAE:

```powershell
abaqus cae database="tasks\Task_07_Abaqus\results\cae\Task7_Montresor_WindStudy.cae"
```

If the canonical file fails to open after an interrupted run, use:
- `tasks\Task_07_Abaqus\results\cae\Task7_Montresor_WindStudy_last_good.cae`
- then rerun the pipeline with Abaqus fully closed to regenerate and re-promote.

The clean CAE contains the `24 refined models` only, named as:

- `S1_SA`, `S1_PSO`, `S1_BFGS`
- `S2_SA`, `S2_PSO`, `S2_BFGS`
- ...
- `S8_SA`, `S8_PSO`, `S8_BFGS`

Refined job names follow the same mapping, for example:

- `task7_s1_sa_refined`
- `task7_s4_pso_refined`
- `task7_s8_bfgs_refined`

Use the `.odb` files in `tasks/Task_07_Abaqus/results/jobs/` for solved results in the Visualization module.

Example direct viewer launch:

```powershell
abaqus viewer database="tasks\Task_07_Abaqus\results\jobs\task7_s7_bfgs_refined.odb"
```

Recommended review workflow:

1. Open the clean `.cae` first.
2. Inspect the chosen model tree for section, steps, BCs, and the single `SelfWeight` + single `Wind` load objects.
3. Switch to the Visualization module and open the corresponding solved `.odb`.
4. For the static result, inspect `STATIC_WIND`, last frame.
5. Use `Plot -> Contours on Deformed Shape`.
6. For stress, set `Result -> Field Output` to `S` and choose invariant `Mises`.
7. For displacement, set `Result -> Field Output` to `U` and choose magnitude.
8. For buckling, switch to `BUCKLING`, `Mode 1` and display `U` magnitude.
9. Compare coarse and refined `.odb` files for the same case when checking convergence.

## Notes

- The clean CAE is a presentation database with refined models only.
- The automated verification path still runs coarse and refined solver decks for all `24` cases.
- Task 7 uses a strict `Y-up` convention in both CAE models and solver decks.
- Geometry integrity is checked before solving in `results/data/geometry_verification.csv`.
- Material and load consistency is checked before solving in `results/data/input_contract_audit.csv`.
- Report ranking is generated from refined results with explicit weighted-score settings in `config/study_config.json` under `ranking`.
- Report generation enforces a consistency audit in `results/data/report_consistency_audit.{csv,json}` and fails if ranking or mesh narratives drift from source data.
- The towers are visually squat because the Task 6 source geometry is squat; there is no aspect-ratio normalization in the structural model.
- All stress, displacement, and buckling figures are rendered from exported Abaqus ODB field data in Python, not from Abaqus screenshots, with a true-scale equal-axis visual policy.
- Warning-only cases, especially all of `S8`, are included for structural awareness and comparison, not as recommended designs.
