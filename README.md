# Group Project: Stochastic vs Deterministic Optimization

A comparative study of optimization algorithms (Simulated Annealing, Particle Swarm Optimization, and BFGS) applied to standard benchmarks (Rastrigin, Rosenbrock) and engineering problems.

## Quick Start

The project includes a unified runner script `run_project.py` to easily execute experiments and regenerate results.

### Running Tasks

To run the entire project (Task 3 and Task 4):
```bash
python run_project.py
```
or
```bash
python run_project.py --task all
```

To run a specific task:
```bash
python run_project.py --task 3
```
or
```bash
python run_project.py --task 4
```

### Automatic Cleanup
By default, the script **deletes** the old `results` directory for the target task before running to ensure data freshness.

To disable this behavior (e.g., to keep old results):
```bash
python run_project.py --task 4 --no-clean
```

### Global Seeding
To force fresh data generation with a new random seed:
```bash
python run_project.py --random
```

To use a specific seed (for reproducibility):
```bash
python run_project.py --seed 12345
```

## Structure

*   `src/`: Core implementation of algorithms and benchmarks.
*   `tasks/`: Task-specific scripts and reports.
    *   `Task_03_Comparison/`: Initial stochastic methods comparison.
    *   `Task_04_New_Techniques/`: Implementation of BFGS and advanced visualizations.
*   `run_project.py`: Main entry point.

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib
