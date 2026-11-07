# Baseline Comparison — Mission 01 (measured, not simulated)

Validation: chronological rolling-origin, 4 folds, 1–72 h horizons (n=1,000 OOF hours). All models share the same 28 causal features. CatBoost library unavailable → **not run**.

## Overall

| Model | RMSE | MAE | Bias |
|---|---|---|---|
| Random Forest | **10.57** | **8.11** | −3.66 |
| 24 h rolling-mean | 10.93 | 8.47 | −3.38 |
| HistGradientBoosting | 11.68 | 8.45 | **−1.67** |
| Persistence (lag1) | 12.60 | 9.66 | −3.94 |

Machine learning barely beats the rolling-mean baseline overall; the win is driven by longer horizons (25–72 h), while persistence wins at horizon 1–6 h.

## Fold-wise (see MISSION_01_RESULTS for per-station and lead breakdowns)

| Cutoff | RF | HGB | Persistence | Roll24 |
|---|---|---|---|---|
| 2026-08-20 | 3.89 | 4.08 | 5.23 | 3.86 |
| 2026-09-20 | 11.37 | 10.30 | 9.68 | 11.73 |
| 2026-10-05 | 13.48 | 12.70 | 19.58 | 14.08 |
| 2026-10-20 | 10.55 | 15.43 | 10.93 | 10.84 |

Reading: RF is the only stable model (HGB collapses to 15.43 in the latest fold). Seasonal regime shift (Aug mean ~13.7 → Oct ~44.9 → Nov 68.5 in clean train) explains rising fold RMSE.

## Extremes (measured)

| Cohort | n | RF RMSE | HGB RMSE |
|---|---|---|---|
| y > 50 | 242 | 12.02 | 15.90 |
| y > 62 | 90 | 13.57 | 15.78 |
| y > 75 | 19 | 20.33 | 19.42 |
| y > 100 | 0 | — | — |
| y > 200 | 0 | — | — |

## Decision

**Ship Random Forest** as `baseline_v0`. HGB (bias-corrected but high-variance) and both naive baselines are retained as reference. Next: persistence-blend RF and winter-extreme validation.
