# Mission 02.1 Results — Validation Correction & Long-Lead Intelligence

**Date:** 2026-11-07 · Artifacts preserved (M01/M02 untouched). New: `src/mission02_1.py`, `src/mission02_1_resume.py`, `src/mission02_1_stage2.py`, `src/mission02_1_stage3.py`, `src/analyze_m021.py`; OOF in `reports/m021/`.

## Design fixed

- Contiguous **194–241h 48-hour window** at every origin (auto-derived from `train_max`, no hardcoding), 11–12 weekly origins with ≥120d history, all stale/missing reported (Jan-05 origin 0/192).
- Dense horizons **72–288 step 6**, per-example `target ≤ origin`, duplicate audit (86–93% dup, max 40×), inverse-frequency weights (harmful).
- General (dense) vs specialised (168–288) RF compared.

## Winners on the true private window (n=1,825)

**B3 25.57, B2 25.70, persistence 25.86, B1 26.43, general RF 27.80, specialised 29.40, B0 33.00.** Relative skill vs strongest baseline (B3): RF −8.7% (worse). Per-origin RF 6.1–79.7 (Nov-17 extreme). Per-station: Byculla RF wins, others B3 wins. Calibration: B3 best (a=1.14); general RF a=1.47. Residual learning fails (Ridge 26.51, HGB 32.39, RF 31.97 vs B3 25.57). Lookback v2: **72h wins (14.99)**, 48h 15.56; lead feature harms (drop → 34.5 vs 42.4). Extreme: log 11.02 best; median worst. Corrected update sim (192–239 → 48–95, both ≤240): Sep 6.54→6.58 (flat), Dec 22.81→20.09 (refresh helps).

## Final decision for T+194–241

**Strongest: B3 (seasonal baseline with recent-level adjustment)** — honest winner on overall/per-station/season-wise/calibration. RF/general/specialised/residual all lose. Recommendations: (1) adopt B3 (72h window variant) as final model; (2) remove `lead` features; (3) use log-target if tree is forced; (4) block-bootstrap/CV note: fold variability is huge (Nov fold dominates) — do not select by one fold; (5) no GNN, no unvalidated submission created; M01 `baseline_v0.csv` + M02 artifacts frozen.

**Blocked/incomplete:** original M02.1 Stage-1 script had NaN-baseline and Jan-gap crashes (resume scripts completed them); lookback/ablation/extreme subsets are smaller than primary (6/4/3 origins) — documented, no fabricated extension. Paired bootstrap not run (numpy bootstrap on 1,825 rows recommended before final submission).
