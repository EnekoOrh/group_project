# Task 6: Cooling Tower

This folder is organized so generated artifacts are separated from source code.

## Structure

```text
Task_06_Cooling_Tower/
  run_experiments.py            # Main experiment runner
  Report.md                     # Legacy snapshot report (kept for reference)
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

2. Generate LaTeX tables:
```bash
python tasks/Task_06_Cooling_Tower/report_latex/scripts/generate_tables.py
```

3. Compile LaTeX report:
```bash
cd tasks/Task_06_Cooling_Tower/report_latex
latexmk -pdf -interaction=nonstopmode main.tex
```
