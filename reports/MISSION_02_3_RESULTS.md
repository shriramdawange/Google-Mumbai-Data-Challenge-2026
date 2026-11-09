# Mission 02.3 Results — Update-09 Short-Horizon (T+42–89) Validation

**Date:** 2026-11-09 · Data: `data/processed/UPDATE-09/training_dataset.parquet` (48,133) + `UPDATE-09/Test.csv` (2,592, private 192/48h, leads **42–89 auto-derived**). M01–M02.2 frozen; `src/finalize.py` not run; **no submission created** (inference smoke only: 2,592 finite non-negative, order preserved, **not written** to `submissions/`). Adapter asserts unique keys, 4 stations, chronological order — all pass.

## Models (identical rows, n=1,816 over 12 origins)

- **A. B2** (7d level × station diurnal): RMSE **23.25**, best pooled.
- **B. B3** (w fitted early w=0.45): **24.09**.
- **C. B4** (B3+β·trend, β=0.0 → identical to B3): **24.09**. **B4 does NOT improve on B3** (`p_B4_better_B3=0.0`).
- **D. RF** (short horizons 36–96, seed 42): **41.05** — collapses (Dec-04 fold 119.1). **Not competitive** at short horizons either, but for opposite reason (high variance on winter spikes).

## Stability / honest selection

- **Late folds (untouched, 5 origins): B2 14.97 < RF 15.71 < B3/B4 19.86.** Late winners: B2, then RF. Pooled winner ≠ late winner.
- Per-origin B2 ranges 3.3–51.3; driven by Oct/Dec spikes. Oct-22 origin has **n=3** (missing window, reported).
- Bootstrap (10k origin blocks): `p(B4>RF)=0.69`, B4 CI [13.5, 33.3] (`reports/m023/bootstrap.json`). B4 vs RF is favorable but B4=B3.
- Extremes (y>100, n=4): both ~161 — cleaning policy kept, sensitivity documented (19 >1,000 excluded pre-features, eval targets preserved).

## Recommended model for the final update

**B2 (7-day recent level × station diurnal)** — best pooled (23.25) and best on untouched late folds (14.97), simplest and most stable. **Do not use B4** (no gain) **or raw RF** (variance).

## Artifacts

`src/mission02_3.py`, `src/mission02_3_stage2.py` (UPDATE-10-ready `infer(parquet, test)`), `reports/m023/{config, validation_oof, fold_meta, metrics_by_origin, metrics_by_station, bootstrap, extreme_sensitivity}`, `.gitignore` added (`__pycache__`, `*.pyc`).

**Limitations:** 12 origins (Oct-22 n=3 sparse); RF short grid 36–96 may still miss; winter fold dominates; bootstrap CIs overlap; UPDATE-10 leads not yet validated.
