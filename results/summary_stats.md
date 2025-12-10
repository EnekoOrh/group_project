# Optimization Experiment Report
**Experiment Budget**: 10000 Evaluations
**Method**: Comparison of SA and PSO on equal evaluation cost basis.

## Problem: rastrigin
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 1.783376e+01 | 1.045146e+01 | 0.2929 | 10000.0 |
| PSO | 2.321571e-01 | 4.208205e-01 | 0.4394 | 10020.0 |


## Problem: rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 1.197197e-05 | 8.664794e-06 | 0.3964 | 10000.0 |
| PSO | 0.000000e+00 | 0.000000e+00 | 0.3906 | 10020.0 |


## Problem: constrained_rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals |
|---|---|---|---|---|
| SA | 4.621426e-02 | 3.614006e-04 | 0.5450 | 10000.0 |
| PSO | 4.567112e-02 | 5.942114e-18 | 0.4938 | 10020.0 |


## Penalty Sensitivity (Constrained Rosenbrock - PSO)
| Penalty Factor | Mean Best Val | Mean Violation |
|---|---|---|
| 1 | 4.238716e-02 | 5.422616e-02 |
| 10 | 4.531039e-02 | 5.999067e-03 |
| 100 | 4.563795e-02 | 6.067123e-04 |
| 1000 | 4.567112e-02 | 6.074055e-05 |
| 10000 | 4.567444e-02 | 6.074748e-06 |