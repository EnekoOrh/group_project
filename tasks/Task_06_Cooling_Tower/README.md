# Task 6: Cooling Tower

This folder is organized so generated artifacts are separated from source code.

## Structure

```text
Task_06_Cooling_Tower/
  run_experiments.py            # Main experiment runner
  Report.md                     # Live markdown report used for the LaTeX build
  scripts/                      # Helper scripts for report-only rebuilds
  tests/                        # Unit tests for geometry and gradients
  report_latex/                 # LaTeX manuscript source
    scripts/generate_tables.py  # CSV -> .tex table generator
    sections/                   # Report sections
    tables/                     # Generated .tex table fragments
  results/                      # Generated outputs
    data/                       # CSV/JSON metrics
    figures/                    # PNG plots
    latex/                      # CSV inputs for LaTeX tables
    reports/                    # Generated markdown report(s)
```

## Recommended Workflow

1. Run experiments:
```bash
python tasks/Task_06_Cooling_Tower/run_experiments.py
```

2. If you only changed report text/layout and want to reuse the existing results, rebuild the markdown report without rerunning the optimization:
```bash
python tasks/Task_06_Cooling_Tower/scripts/rebuild_report_from_results.py
```

3. Generate LaTeX tables:
```bash
python tasks/Task_06_Cooling_Tower/report_latex/scripts/generate_tables.py
```

4. Sync markdown report into LaTeX source:
```bash
python tasks/Task_06_Cooling_Tower/report_latex/scripts/sync_report_md_to_tex.py
```

5. Compile LaTeX report:
```bash
cd tasks/Task_06_Cooling_Tower/report_latex
latexmk -pdf -interaction=nonstopmode main.tex
```

The compiled PDF is written to:

- `tasks/Task_06_Cooling_Tower/Task6_Montrésor_CoolingTower.pdf`
