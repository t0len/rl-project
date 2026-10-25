## Table 1. Final comparison (greedy eval, 200 episodes x 10 seeds; mean ± std over seeds)

| Method | Mean return | Success rate | Crash rate |
|---|---|---|---|
| Q-learning (eps=0.1) | 15.80 ± 0.60 | 1.00 ± 0.00 | 0.00 ± 0.00 |
| Random | -40.58 ± 0.16 | 0.00 ± 0.00 | 1.00 ± 0.00 |
| Battery-blind rule | -38.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 |
| Fuel-aware rule | 8.69 ± 1.09 | 0.93 ± 0.02 | 0.07 ± 0.02 |
| Value iteration (model known) | 16.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 |

## Table 2. Epsilon study

| epsilon | Eval return | Success rate | Crash rate | Training return (last 500 ep.) |
|---|---|---|---|---|
| 0.01 | 14.36 ± 2.25 | 0.99 ± 0.02 | 0.01 ± 0.02 | 12.78 ± 2.21 |
| 0.1 | 15.80 ± 0.60 | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.57 ± 1.31 |
| 0.3 | 16.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | -31.00 ± 0.57 |
