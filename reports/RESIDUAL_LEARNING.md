# Residual Learning (M02.1 §4)

Reference = **B3**. Models predict `y − B3`, evaluated on original scale. 11 origins, n=1,825.

| Model | RMSE |
|---|---|
| Ridge (α=10) | 26.51 |
| Shallow HGB (150×15) | 32.39 |
| RF residual (100×depth10) | 31.97 |
| **B3 alone** | **25.57** |

Residual learning **does not beat the baseline** — Ridge closest (−0.94 worse), trees much worse (+6 to +7). Per-origin: residual HGB/RF help only on Oct 5/12 origins (12–20 vs B3 25–30) but collapse on Nov 17 (94–97 vs 64). Measured conclusion: no superiority; do not adopt residual formulation.
