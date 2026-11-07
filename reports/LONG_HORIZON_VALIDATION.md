# Long-Horizon Validation (Mission 02)

Direct-horizon, rolling-origin. Origins: Dec 15 2025, Jan 15 2026 (winter), Aug 20 / Oct 5 / Oct 20 2026. Leads evaluated: 24, 48, 72, 120, 168, 194, 216, 240, 241. Training uses only data ≤ origin; horizons up to 240 h; step 6 h.

## Coverage (measured)

Only **100 of 180 possible station×lead×fold cells** have observed targets (81 missing due to ~20% missingness). Eval rows: F0 23, F1 16, F2 14, F3 32, F4 15. Small-n caveat per lead (9–16 pooled rows). RF config M02: 150 trees (vs 200 in M01).

## Pooled RMSE by lead

| Lead | n | RF | Persistence | Roll24 |
|---|---|---|---|---|
| 24 | 13 | 13.53 | 14.52 | **10.53** |
| 48 | 9 | **8.82** | 11.09 | 9.22 |
| 72 | 11 | 39.24 | 24.52 | **15.51** |
| 120 | 10 | **15.18** | 27.66 | 17.45 |
| 168 | 15 | 31.39 | 32.30 | **30.09** |
| 194 | 16 | 22.44 | 25.61 | **20.35** |
| 216 | 11 | **17.05** | 24.56 | 20.89 |
| 240 | 11 | 41.52 | 35.33 | **26.56** |
| 241 | 4 | **21.82** | 39.46 | 31.85 |

Overall (n=100): RF 26.31 · persistence 26.62 · **rollmean24 21.15**. Private band (≥194 h, n=42): RF 27.66 · persistence 29.72 · **rollmean24 23.52**.

## Interpretation

- Ranking **reverses vs M01 (1–72 h)**: tree model degrades faster with lead (72 h RF 39.2 vs roll 15.5). No model is stable: per-fold RF ranges 2.2–74.7.
- Causes: direct trees cannot extrapolate regimes; persistence/roll24 are flat but robust. RF wins only at 48, 120, 216, 241 h (isolated cells, n≤11).
- F0 (Dec) is hardest (RF 63.7 at 72 h, 74.7 at 240 h); F2 (Aug low) easiest (all RMSE ≤6).

Data: `reports/m02/longhorizon_oof.csv`, `blend_grid.csv`, `lookback_oof.csv`, `update_sim.csv`.

**Verdict:** For T+194–241, the honest M02 evidence favors a **shrunk/rolling-mean-type forecast over raw RF**. A raw RF chosen on 1–72 h RMSE is not justified for the private horizon.
