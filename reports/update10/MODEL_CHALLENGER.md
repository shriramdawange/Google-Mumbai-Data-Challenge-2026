# MISSION 03.2 — FINAL MODEL CHALLENGER: B2 vs B3 (actual final UPDATE-10 data)

**Date:** 2026-11-10 (Mission 03.2)
**Scope:** Compare B2 vs B3 using **only the last seven untouched validation origins** (folds 5–11; parameter-fitting folds 0–4 excluded).

## Setup
- Training source: actual final `UPDATE-10/Train.csv` (48,400 rows, cutoff 2026-11-10 09:00:00+05:30) → parquet `data/processed/UPDATE-10/training_dataset.parquet`.
- Private forecast leads: T+16 … T+63 (48h / 192 private rows).
- Untouched origins (folds 5–11): 2026-03-26, 2026-05-21, 2026-07-16, 2026-09-10, 2026-10-22, 2026-10-29, 2026-11-05 (n=1120 pooled rows).

## Challenger results (last seven untouched origins)

| model | RMSE | MAE | bias | n |
|---|---:|---:|---:|---:|
| **B2** | **14.2927** | 9.5085 | 0.2203 | 1120 |
| B3 | 16.6181 | 11.8856 | −2.5077 | 1120 |
| B4 | 16.6181 | 11.8856 | −2.5077 | 1120 |
| RF | 14.1979 | 10.0941 | 3.7738 | 1120 |

- B4 ≡ B3 on this comparison (β=0).

## Paired origin-block bootstrap (last seven, 10k resamples, B2 vs B3)

- Observed diff (sqrtMSE(B3) − sqrtMSE(B2)) = **2.3254**.
- 95% bootstrap CI = **[−1.01, 5.57]; P(B2 better) = 0.9298** → B2 favored on untouched recent origins.

## Per-origin (last seven) RMSE — B2 vs B3

| fold | origin | n | B2 | B3 |
|---:|---|---:|---:|---:|
| 5 | 2026-03-26 | 184 | 17.161 | 19.040 |
| 6 | 2026-05-21 | 147 | 16.084 | 18.776 |
| 7 | 2026-07-16 | 166 | 5.319 | 5.348 |
| 8 | 2026-09-10 | 174 | 5.151 | 5.579 |
| 9 | 2026-10-22 | 107 | 9.504 | 21.753 |
| 10 | 2026-10-29 | 172 | 16.186 | 10.745 |
| 11 | 2026-11-05 | 170 | 20.657 | 25.451 |

## Latest origin (2026-11-05) separate check

- B2: RMSE 20.657, MAE 15.118, bias −14.264.
- B3: RMSE 25.451, MAE 20.679, bias −20.545.
- RF: RMSE 15.309, MAE 11.664, bias −3.690.
- On the latest origin, B2 beats B3; RF edges B2 (pooled, not decisive).

## Per-station metrics (last seven, B2 vs B3)

| station | n | B2 RMSE | B3 RMSE | B2 MAE | B3 MAE |
|---|---:|---:|---:|---:|---:|
| 11606 | 284 | 10.532 | 13.509 | 7.305 | 10.559 |
| 11611 | 292 | 13.843 | 18.319 | 9.655 | 13.958 |
| 3409510 | 257 | 19.214 | 19.083 | 11.642 | 11.576 |
| 3409511 | 287 | 12.725 | 15.152 | 9.630 | 11.367 |

## Final B2 vs B3 recommendation

- **Based on untouched recent origins + robustness:** **B2 is recommended** (pooled RMSE 14.29 vs B3 16.62; bootstrap P(B2 better)=0.93; B2 also better on latest origin).
- **Uncertainty (no fabricated certainty):** bootstrap 95% CI for B2 vs B3 spans [−1.01, 5.57]; B4 ≡ B3; per-station and per-origin heterogeneity exist; latest-origin result favors B2 but is single-origin.
- **Challenger conclusion:** the final model challenger selects **B2** as the final candidate based on untouched recent-origin performance, while the previously submitted final candidate **B3** (`submissions/candidate_update10_final.csv`, SHA `e059e4ba…a40`) remains preserved.

## Candidate artifacts

- B2 final challenger candidate: `submissions/candidate_update10_B2.csv` — SHA256 `417cbcee…d1f52` — ALL PASS.
- B3 final candidate (preserved): `submissions/candidate_update10_final.csv` — SHA256 `e059e4ba…a40`.
