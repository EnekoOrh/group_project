# Optimization Experiment Report
**Experiment Budget**: 10000 Evaluations
**Method**: Comparison of SA and PSO on equal evaluation cost basis.

**Seed Offset**: 42

## Problem: rastrigin
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 2.016462e+01 | 1.222324e+01 | 0.1864 | 10000.0 |
| PSO | 3.316530e-01 | 5.347744e-01 | 0.2518 | 10020.0 |


## Problem: rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 8.940713e-06 | 8.321278e-06 | 0.2099 | 10000.0 |
| PSO | 0.000000e+00 | 0.000000e+00 | 0.2456 | 10020.0 |


## Problem: constrained_rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 4.608746e-02 | 3.276498e-04 | 0.2785 | 10000.0 |
| PSO | 4.567112e-02 | 5.665583e-18 | 0.3333 | 10020.0 |

