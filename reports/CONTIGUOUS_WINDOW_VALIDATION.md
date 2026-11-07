# Contiguous Window Validation (M02.1 §1–2)

**Leads auto-derived:** private target_end Nov 11 01:00 → Nov 13 00:00 minus `train_max` Nov 2 23:00 +05:30 = **194–241** (48 leads, not hardcoded; `reports/m021/config.json`).

**Origins:** weekly grid, ≥120 days history, ≤ train_max−265h; 12 origins (Jun 2025–Oct 2026). **11 non-empty** (1,825 observed of 11×192=2,112 possible, 86%). Origin 2026-01-05: **0 observed** (194–241h falls in Jan 11–19 gap) — reported, not imputed.

## Pooled RMSE (n=1,825 unless noted)

| Model | RMSE | Bias |
|---|---|---|
| **B3 seasonal-adjusted** | **25.57** | −7.02 |
| B2 level×diurnal | 25.70 | −3.37 |
| Persistence (n=1,413) | 25.86 | — |
| B1 recent means | 26.43 | −9.06 |
| General RF (dense 72–288) | 27.80 | −8.36 |
| General RF + inverse-freq weights | 28.24 | −8.16 |
| Specialised long-lead RF (168–288) | 29.40 | −5.89 |
| B0 hour profile | 33.00 | −7.00 |

**Per-origin RF:** 10.08, 7.15, 11.02, **79.71 (Nov 17 2025 extreme)**, 24.10, 12.99, 8.02, 6.06, 15.07, 21.26, 13.15. One Diwali-season fold (target max 708.8, 62 rows >100, 2 rows >300) dominates pooled RMSE.

**Training audit:** dense horizons 72–288 step 6 + short (42); per-example `target ≤ origin` (no shared-cutoff discarding); 86–93% duplicate-target extra rows (max 40× same timestamp); inverse-frequency weighting **hurts** (28.24 vs 27.80). General > specialised (27.80 vs 29.40).

**Verdict:** Baselines beat both RFs on the true private window. Data: `reports/m021/{contiguous_oof.csv, fold_meta.json}`.
