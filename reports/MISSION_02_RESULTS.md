# Mission 02 Results — Forecasting Intelligence

**Date:** 2026-11-07 · **Frozen:** M01 artifacts untouched (`submissions/baseline_v0.csv`, `reports/oof_validation.csv`). New code: `src/mission02.py`; data: `reports/m02/`.

## 1. Long-horizon validation (leads 24–241, 5 origins, n=100 observed)

Pooled: **rollmean24 21.15 < RF 26.31 < persistence 26.62**. Private band ≥194 h (n=42): roll24 **23.52** < RF 27.66 < persistence 29.72. Per-lead table + coverage in `LONG_HORIZON_VALIDATION.md`. RF wins isolated leads (48, 120, 216, 241) but collapses at 72 h (39.2) and 240 h (41.5). Ranking **reverses** vs M01's 1–72 h (RF 10.57 best). **Do not select RF blindly.**

## 2. Persistence blending

Early-optimal (honest, F0–F2): w=0.0–0.1 (reject RF). Late folds want w≥0.5 (pure RF 16.90 best). Oracle w per lead flips 0.0↔1.0. Blending **rejected** (`PERSISTENCE_BLEND.md`).

## 3. Lookback (identical validation, leads 24–240, n=37/W)

Pooled: W120 **16.09**, W168 16.65, W336 16.66, W12 17.49, W48 17.59, W72 17.78, W6 19.76, W24 21.89. Long lookbacks slightly better; winter dominates all. Insufficient n to replace multi-window features (`LOOKBACK_COMPARISON.md`).

## 4. Seasonal / extremes

Winter (Dec+Jan, n=39) is hardest; `y>75` RF 33.2 (vs pooled ~26); `y>100` n=1 only — reported as n=1, no >150 cases exist. M01's ≤86.8 max never tested winter. (`SEASONAL_ERROR_ANALYSIS.md`.)

## 5. Nov 9/10 update simulation (fixed 48 h targets, long vs refreshed cutoff)

- Oct 15 target: long 17.62 → refreshed (8 days later) **12.97 (−26%)**.
- Dec 20 target: long 26.91 → refreshed **28.71 (+7%)**.

Lead reduction helps when regime is stable, hurts during winter regime shift. Data release alone does not guarantee improvement.

## 6. Recommendation for the private horizon (T+194–241)

**Best honest choice today: a shrinkage / rolling-mean-leaning forecast, not raw RF.** Specifically, for a final v1 submission:
1. Use winter-calibrated shrinkage toward the 168–336 h rolling mean (or W=120 features) rather than the M01 multi-window RF.
2. Consider clipping RF predictions toward roll24 with a conservative fixed gate (e.g., 0.5) only if re-validated on winter origins — honest blend tuning failed.
3. Retain `baseline_v0.csv` as-is; create a new `baseline_v1_winter_shrunk.csv` after a larger seasonal replication (more origins) and explicit Nov-9/10-release retraining plan.

**Leakage checks:** all experiments use cutoff-only features; training targets ≤ origin−coverage; no `scoring_block` feature; anomalies excluded pre-scoring. M02 RF uses 150 trees (speed); M01 used 200 — noted.

**Files:** `src/mission02.py`, `reports/m02/{longhorizon_oof.csv, blend_grid.csv, lookback_oof.csv, update_sim.csv, fold_info.json}`.
