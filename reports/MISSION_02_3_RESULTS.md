# Mission 02.3 Results — Update-09 Short-Horizon (T+42–89) Validation

**Date:** 2026-11-09 · Data: `data/processed/UPDATE-09/training_dataset.parquet` (48,133) + `UPDATE-09/Test.csv` (2,592, private 192/48h, leads **42–89 auto-derived**). M01–M02.2 frozen; `src/finalize.py` not run; **no submission created** (inference smoke only). Adapter asserts unique keys, 4 stations, chronological order — all pass.

## Models (identical rows, n=1,816 over 12 origins)

- **A. B2** (7d level × station diurnal): RMSE **23.25**, MAE 15.03, bias −2.26 — best pooled.
- **B. B3** (w fitted early w=0.45): RMSE **24.09**, MAE 16.46, bias −3.40.
- **C. B4** (B3+β·trend, β=0.0 → identical to B3): RMSE **24.09**, MAE 16.46, bias −3.40. **B4 does NOT improve on B3** (`p_B4_better_B3=0.0`).
- **D. RF** (short horizons 36–96, seed 42): RMSE **41.05**, MAE 19.11, bias +5.62 — collapses on winter spike.

## Correct reporting audit (M02.3.1)

- `metrics_by_origin.csv` regenerated per-origin: 48 rows (12 origins × 4 models); Oct-22 origin has **n=3** (explicitly sparse/missing-window case), clearly identified.

## Paired bootstrap (M02.3.1 rebuild, 10,000 origin-block resamples, persisted `reports/m023/bootstrap.json`)

| Comparison | Observed diff | 95% CI | p(A better) |
|---|---|---|---|
| B2 vs B3 | 0.842 | [−4.17, 5.71] | 0.638 |
| B2 vs B4 | 0.842 | [−4.17, 5.71] | 0.638 |
| B2 vs RF | 17.79 | [−2.12, 41.66] | 0.805 |
| B3 vs RF | 16.95 | [−6.09, 45.23] | 0.690 |

- Model RMSE bootstrap CIs: B2 [13.26, 32.66], B3/B4 [13.39, 33.42], RF [14.62, 66.84].

## Recommended model for the final update

**B2** — best pooled RMSE (23.25) and best on untouched late folds (14.97 vs RF 15.71 vs B3/B4 19.86). Do not use B4 (no gain) or raw RF (variance).

## M02.3.1 Task outputs

- `reports/m023/metrics_overall.csv` — pooled overall per model (B2/B3/B4/RF).
- `reports/m023/metrics_by_origin.csv` — 48 rows (origin × model with RMSE/MAE/bias/n). Oct-22 n=3 identified.
- `reports/m023/metrics_by_station.csv` — 16 rows (station × model).
- `reports/m023/bootstrap.json` — 10k origin-block paired bootstrap + CIs.
- `reports/m023/inference_diagnostics.json` — B2 inference smoke (2,592 full rows, mean 46.97, min 28.64, max 81.01, leads 42–89).

## M02.3.1 Task 3/4 — B2 inference & UPDATE-10 compatibility

- B2 inference implemented in `src/mission02_3_stage2.py::infer(parquet_path, test_path, selected_model)`. Matches B2 validation definition (7d level × 90d station diurnal norm; missing-handled; pm25>1,000 excluded via clean policy; IST hour via target_end_ist; station mapping by location_id; 2592-row output validated: order, unique, finite, non-negative).
- `infer` fail-louds on: Test row count, Test ID uniqueness, station chronological order, private row count (192), private unique hours (48), contiguous leads, supported lead domain (1–400), selected model support, station history sufficiency.
- Configurable via parameters → supports UPDATE-10 without code changes; tomorrow must re-run with `UPDATE-10/` paths and new leads.

## FINAL SAFETY AUDIT (M02.3.1)

- `submissions/baseline_v0.csv` SHA256 = `a5e709dce45eaf4cb199765cd5f5b009e43e2465c4a65724b9678c79a1725360` — **UNCHANGED**.
- `submissions/baseline_v0_hgb.csv` SHA256 = `eaf4dfc0af17ee0a378637b48aec30381813db492de26348f543ae33a9eecbdc` — **UNCHANGED**.
- `reports/m023/validation_oof.csv` SHA256 = `b96201eba625c79a36e957b13bceb0f225ceabec291c0e2de067363bdd89af87` — **UNCHANGED**.
- Parquet `data/processed/UPDATE-09/training_dataset.parquet` SHA256 = `5a50da1a0065af35674d5fd2025747a3e25f572418f6e0d1931b7eede4b054c5` — **UNCHANGED**.
- M01–M02.2 artifacts untouched (git diff shows only m023/src/.gitignore changes).
- No submission generated; no raw file modified; no commit/push performed for this task.

## FINAL RESPONSE (Mission 02.3.1 deliverables)

1. Exact files modified: `src/mission02_3_stage2.py`, `src/mission02_3_audit.py`, `reports/m023/{bootstrap.json,metrics_by_origin.csv,metrics_by_station.csv,metrics_overall.csv,inference_diagnostics.json}`, `reports/MISSION_02_3_RESULTS.md`, staged `src/__pycache__` deletions, `.gitignore`.
2. Corrected model comparison: B2 23.25 < B3/B4 24.09 < RF 41.05 (pooled; bootstrap).
3. Paired bootstrap results: B2-vs-B3 diff 0.842 [−4.17, 5.71], p=0.638; B2-vs-B4 same; B2-vs-RF 17.79 [−2.12, 41.66], p=0.805; B3-vs-RF 16.95 [−6.09, 45.23], p=0.690; model RMSE CIs [B2 13.26–32.66, B3/B4 13.39–33.42, RF 14.62–66.84].
4. B2 inference verification: PASS — 2,592 rows, mean 46.97, min 28.64, max 81.01, private leads 42–89, train_max 2026-11-09 07:00.
5. UPDATE-10 compatibility status: interface configurable (parquet/test/model args, fail-loud checks) — compatible; pending re-run with UPDATE-10 paths and re-derived leads.
6. Baseline SHA256 verification: PASS (both unchanged).
7. Git status/diff: staged m023/src updates + staged pyc deletions; untracked `reports/m023/inference_diagnostics.json`; working tree otherwise clean; branch master @ ea1a86d + uncommitted M02.3.1 changes.
8. Reproduction: `python src/mission02_3_stage2.py` (inference+dashboard); `python src/mission02_3_audit.py` (metrics+bootstrap); `python src/update09_integrate.py`; `python src/mission02_3.py` (short-horizon backtest).
9. Remaining blockers: (a) UPDATE-10 files not yet present — must re-run integration with new cutoff/horizon; (b) B2's inference uses only level×diurnal (statistical model) — final submission still pending per mission sequence; (c) per-origin variance (Dec spike); (d) bootstrap CIs wide.

## Remaining blockers / human review

- Nov-10 UPDATE-10 release expected; re-validate with auto-derived leads before final training.
- B4 gain over B3 = 0.0 (not recommended).
- Wide bootstrap CIs; single sparse fold (Oct-22 n=3).
- Final inference for all 2592 Test IDs uses B2; no final submission produced.
