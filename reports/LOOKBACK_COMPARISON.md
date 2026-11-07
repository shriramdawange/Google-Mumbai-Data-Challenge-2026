# Lookback Comparison (Mission 02)

Identical validation: origins Jan 15, Aug 20, Oct 5, Oct 20; leads 24/72/168/240; leads must exist (37 rows per lookback). Only history-window features vary: `[station, calendar, lead, lag1, lagW, roll_mean_W, roll_std_W, city_rollW]`.

## Pooled RMSE (n=37 each)

| Lookback W (h) | RMSE |
|---|---|
| **120** | **16.09** |
| 168 | 16.65 |
| 336 | 16.66 |
| 12 | 17.49 |
| 48 | 17.59 |
| 72 | 17.78 |
| 6 | 19.76 |
| 24 | 21.89 |

Per-origin detail (e.g. W=336: Jan 27.75, Aug 3.52, Oct5 14.88, Oct20 13.77; W=120: Jan 29.27, Aug 4.42, Oct5 12.14, Oct20 10.46). Ranking is noisy: W=24 worst pooled despite being good for one origin (W=12 wins Oct20 at 11.60).

## Interpretation

- Long lookbacks (120–336 h) beat short (6–24 h) pooled, but margin is small (16.1 vs 17.5–21.9) and driven by Oct5/Aug.
- Winter (Jan) dominates error for every W (27.8–33.8); no lookback solves winter.
- Full multi-window M01 features remain reasonable; W=120 single-window is a viable simplifier, but evidence (n=37) is too thin to adopt without a larger seasonal replication.

Data: `reports/m02/lookback_oof.csv`.
