# Mission 01 Results — Google Mumbai Data Challenge 2026

**Date executed:** 2026-11-07 (IST) · **Deadline:** 2026-11-10 · **Mode:** Competitive execution  
**Pipeline:** `src/mission01.py` (validation) + `src/finalize.py` (final train + submission). Raw data in `data/` preserved untouched.

## 1. Environment status

| Item | Measured value |
|---|---|
| Project | `C:\Users\shrir\OneDrive\Desktop\CODE\Mumbai-Forecasting-Hackathon` (git repo) |
| Python | 3.13.15 (MSC v.1944 AMD64) |
| Platform | Windows 11 (build 26300) |
| CPU / RAM | 16 CPUs · 16.8 GB RAM · GPU NVIDIA RTX 3050 6 GB Laptop + Intel UHD (unused; sklearn CPU job) |
| pandas / numpy / sklearn | 3.0.5 / 2.5.3 / 1.9.0 |
| catboost / lightgbm | **not installed** (CatBoost therefore skipped; no install per instructions) |
| xgboost | installed but **not used** (kept to required model set) |
| Docker | not installed, not used |
| Prior artifacts | none (clean `data/` + `src/` only); no model artifacts existed |

## 2. Dataset findings (audited, see `reports/DATA_STATUS.md` for detail)

- `Train.csv`: **47,849 rows**, 2025-02-19 03:00 IST → **2026-11-02 23:00 IST** (all four stations end at the same hour). 10 columns; target `pm25` (µg/m³, end-of-hour).
- `Test.csv`: **2,592 rows** = 4 stations × 648 hours, 2026-10-22 → 2026-11-21. Blocks: **public 768 / private 192 / unscored 1,632**. No observed targets supplied.
- `SampleSubmission.csv`: 2,592 IDs in Test order; target column blank in this snapshot.
- Missing observations: Train hourly coverage **77–82%** (≈18–23% missing per station over 14,925 hourly slots). Missingness is not imputed in raw files.
- Target distribution (raw): mean 43.99, median 21.84, max 241,588. **19 rows >1,000** (all `flag_high_1000=True`; 57 rows >500). After excluding those 19 anomalies: **mean 30.07, std 32.10, median 21.83, p90 62.22, p95 75.87, p99 112.93, max 928.83** (still heavy tail).
- Forecast cutoff: **2026-11-02 23:00 IST** for all four stations. Lead to current public window (Nov 5–6) ≈ 24–97 h; lead to private window (Nov 11–12) ≈ **194–241 h**. Reporting gap is **not fixed/72 h**. Scheduled releases Nov 9 and Nov 10 09:00 IST are in the future relative to the snapshot.
- Previous-work caution: `AdditionalHistory/` (2018–2026 auxiliary archives) was **not merged** — source IDs/sensor definitions are incomparable per `HistoryNotes.txt`, and cross-source mapping is not supplied.

## 3. Protocol and leakage control (checked before accepting results)

- **Design:** direct-horizon forecasting. For each cutoff `C`, features use **only observations ≤ C**; a row predicts PM2.5 at `C+h` with features [station, calendar at `C+h`, lead `h`, lags/rolling city stats at `C`]. No lookahead.
- **Causal features** (28): `station_code`, calendar at target (hour, dow, month + sin/cos, weekend), `lead`, `log_lead`, `lag1/2/3/6/12/24/48` at cutoff, rolling means 6/24/72/168 h and stds 24/72 h at cutoff, trend_24, city-mean lag/24h/72h at cutoff. Lags/rollings are `shift`-based on hourly grids.
- **Outlier rule:** `flag_high_1000` or `pm25>1000` → `NaN` for features and targets (19 rows excluded from training/labels; retained in raw). Both conventions reported; raw RMSE would be astronomically inflated by a single 241,588 value.
- **No future target info:** validation predicts horizons 1–72 h strictly after each fold cutoff; training rows satisfy `target ≤ fold_cutoff − coverage` (training cutoffs stop at `cutoff − 240 h`, horizons up to 240 h). Test inference uses **one** cutoff (Nov 2 23:00).
- **Validation:** chronological rolling-origin, **4 folds**, no random split: cutoffs Aug 20, Sep 20, Oct 5, Oct 20 2026, each forecasting 1–72 h (1,000 scored OOF hours total). Train sizes 73k–81k sampled rows per fold.

### Models actually trained

1. **Station-specific historical baselines:** persistence (`lag1` at cutoff) and 24 h rolling-mean.
2. **Random Forest:** `RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4)`.
3. **HistGradientBoosting:** `HGB(max_iter=400, lr=0.06, max_leaf_nodes=63, min_samples_leaf=20, l2=0.1)`.
4. **CatBoost:** **not available** (library missing) — not trained; recorded as a documented blocker, not fabricated. Final train = **84,894** sampled rows (cutoffs every 6 h, horizons 1–240 h).

## 4. Exact validation RMSE (rolling-origin OOF, 1–72 h)

### Overall (n=1,000)

| Model | RMSE | MAE | Bias (pred − y) |
|---|---|---|---|
| **Random Forest (best)** | **10.57** | 8.11 | −3.66 |
| HistGradientBoosting | 11.68 | 8.45 | −1.67 |
| 24 h rolling-mean baseline | 10.93 | 8.47 | −3.38 |
| Persistence (lag1) baseline | 12.60 | 9.66 | −3.94 |

### Fold-wise RMSE

| Fold (cutoff) | n | RF | HGB | Persistence | Roll-mean24 |
|---|---|---|---|---|---|
| 2026-08-20 | 226 | 3.89 | 4.08 | 5.23 | **3.86** |
| 2026-09-20 | 245 | 11.37 | 10.30 | **9.68** | 11.73 |
| 2026-10-05 | 252 | 13.48 | **12.70** | 19.58 | 14.08 |
| 2026-10-20 | 277 | **10.55** | 15.43 | 10.93 | 10.84 |

### Per-station RMSE (pooled OOF)

| Station | RF | HGB | Persistence | Roll-mean24 |
|---|---|---|---|---|
| 11606 Borivali East | 10.63 | **9.03** | 11.17 | 10.98 |
| 11611 Malad West | 11.12 | 12.96 | 12.97 | **10.54** |
| 3409510 Byculla | 10.87 | 11.08 | 12.12 | **10.08** |
| 3409511 Shivaji Nagar | **9.56** | 13.13 | 13.99 | 12.03 |

### By lead bucket (RF / HGB / persistence)

| Lead | RF | HGB | Persistence |
|---|---|---|---|
| 1–6 h | 10.50 | 15.72 | **7.66** |
| 7–24 h | 8.91 | 9.66 | 10.36 |
| 25–48 h | **9.46** | 9.67 | 11.10 |
| 49–72 h | **12.61** | 13.69 | 16.01 |

### Extreme-value performance

- Validation windows happened to be low-pollution: **0 of 1,000 OOF hours exceeded 100 µg/m³** (max 86.8, mean 33.8). RMSE for `y>100`/`y>200` is therefore **not estimable** from this OOF and is reported as `n=0`, not zero.
- Measured thresholds within OOF: `y>50` (n=242): RF 12.02 vs HGB 15.90; `y>62` (n=90): RF 13.57 vs HGB 15.78; `y>75` (n=19): RF 20.33 vs HGB 19.42. Models under-predict high episodes (overall bias −3.66/−1.67).

## 5. Best model and submission

- **Best:** Random Forest (overall OOF RMSE 10.57). HGB is unstable at short leads and in the most recent fold (15.43) and was **not** used for the submission.
- **Submission:** `submissions/baseline_v0.csv` (RF) — 2,592 rows, header `ID,Target`, **Test ID order preserved**, no duplicate IDs, no missing/finite failures, all ≥0. Mean 50.22, min 35.46, max 69.08, std 7.48. SHA-256 `a5e709dce45eaf4cb199765cd5f5b009e43e2465c4a65724b9678c79a1725360` (124,946 bytes). HGB-variant saved as `baseline_v0_hgb.csv` for reference only.
- **Validation checks run:** exact row count (2,592), ID set/order vs `Test.csv`, `ID.duplicated()==0`, `Target.notna().all()`, `np.isfinite`, `(Target>=0).all()`.

## 6. Remaining weaknesses

1. **Short-lead underperformance:** persistence beats RF at 1–6 h (7.66 vs 10.50). The current model smooths too much; near-term forecasts would improve with a persistence blend.
2. **Under-dispersion / high bias on clean hours:** submission never predicts below 35.5; validation shows systematic under-prediction (−3.7) on low-to-mid episodes and compression of variance (pred std < y std).
3. **No extreme-episode validation:** OOF contains no >100 events; Diwali-season spikes (Nov mean 68.5 in clean train vs 13.7 in Aug) are untested.
4. **Single-cutoff staleness:** one Nov 2 cutoff drives all 27 days of Test rows, including retrospective rows (lead <0, stale/unscored) — methodologically a backfill, not a forecast; scored leads (public ≤97 h, private 194–241 h) are the meaningful regime.
5. **Tree lead extrapolation:** horizons >240 h (late unscored rows up to +433 h) saturate; private leads (194–241 h) sit at the training edge.
6. **Missingness (~20%)** handled only via lag availability and median imputation; no explicit gap-duration features.

## 7. Recommended next experiment

1. **Persistence-blended RF with lead gating** (e.g., `w(lead)·persistence + (1−w)·RF`, `w` decaying with lead), validated on the same 4 folds with identical split — cheapest fix for weakness (1).
2. **Winter-extreme holdout:** add folds at Dec 2025/Jan 2026 cutoffs to measure >100-hour RMSE before finalizing; tune Huber/quantile or log-target if extremes dominate.
3. **Recency weighting:** exponential decay on training cutoffs (winter-month emphasis) to correct Nov bias; no GNN/dashboards until these are measured.

---

**Files:** `data/` (preserved), `src/mission01.py`, `src/finalize.py`, `submissions/baseline_v0.csv`, `reports/{oof_validation.csv, validation_results.json, overall_metrics.json, submission_diagnostics.csv}`.
