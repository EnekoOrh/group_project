# Task 4 Experiment Summary
**Seed Offset**: 0

## Problem: rastrigin
| Algorithm | Mean Best | Mean Evals | Mean Time |
|---|---|---|---|
| BFGS | 1.57e+01 | 1600.5 | 0.0116 |
| PSO | 2.49e-01 | 10020.0 | 0.2372 |
| SA | 1.92e+01 | 10000.0 | 0.3362 |

## Problem: rosenbrock
| Algorithm | Mean Best | Mean Evals | Mean Time |
|---|---|---|---|
| BFGS | 8.46e-02 | 705.2 | 0.0173 |
| PSO | 0.00e+00 | 10020.0 | 0.3846 |
| SA | 1.15e-05 | 10000.0 | 0.2977 |

## Problem: constrained_rosenbrock (Constrained BFGS Analysis)
| Penalty Factor | Mean Total Cost | Mean Evals |
|---|---|---|
| 1 | 4.2387e-02 | 159.6 |
| 10 | 1.9825e-01 | 2195.8 |
| 100 | 2.0326e-01 | 2234.6 |
| 1000 | 1.6379e-01 | 2756.8 |
| 10000 | 4.3377e-01 | 3748.1 |