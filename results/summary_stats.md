## Problem: rastrigin
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals | Success Rate |
|---|---|---|---|---|---|
| SA | 1.784132e+01 | 1.045155e+01 | 0.0061 | 501.0 | 0.0% |
| PSO | 2.321571e-01 | 4.208205e-01 | 0.2614 | 15030.0 | 76.7% |


## Problem: rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals | Success Rate |
|---|---|---|---|---|---|
| SA | 1.341958e-02 | 1.742835e-02 | 0.0074 | 501.0 | 6.7% |
| PSO | 0.000000e+00 | 0.000000e+00 | 0.2870 | 15030.0 | 100.0% |


## Problem: constrained_rosenbrock
| Algorithm | Mean Best Val | Std Dev | Mean Time (s) | Mean Evals | Success Rate |
|---|---|---|---|---|---|
| SA | 1.504220e-02 | 2.130381e-02 | 0.0103 | 501.0 | 3.3% |
| PSO | 0.000000e+00 | 0.000000e+00 | 0.4018 | 15030.0 | 100.0% |


## Penalty Sensitivity (Constrained Rosenbrock - PSO)
| Penalty Factor | Mean Best Val | Mean Violation |
|---|---|---|
| 1 | 9.860761e-33 | 1.687539e-15 |
| 10 | 0.000000e+00 | 8.881784e-17 |
| 100 | 0.000000e+00 | 0.000000e+00 |
| 1000 | 9.860761e-33 | 6.217249e-16 |
| 10000 | 0.000000e+00 | 8.881784e-16 |