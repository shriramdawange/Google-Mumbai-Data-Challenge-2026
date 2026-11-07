# Missingness Analysis (M02.1 §6)

Causal features added (never zero-fill): `hours_since_last_valid`, `valid_cnt_24`, `cov_72`, `cov_168`, `recent_median_24`, `roll_med_24`, `lag1` (last value).

Evidence:
- **Origin 2026-01-05: 0/192 targets observed** — 194–241h window sits in Jan 11–19 outage. 1 of 12 origins unfalsifiable. Reported as missing coverage, not dropped silently in planning.
- Contiguous coverage 86% (1,825/2,112). Persistence baseline has **412 NaN** (lag1 missing, 1,413 scored) — direct cost of missingness.
- Training duplicates: 86–93% extra rows from overlapping cutoffs (max 40× per timestamp); inverse-frequency weights hurt (28.24 vs 27.80) — duplicates are not independent noise.
- Missingness features are included in all M02.1 RFs; ablation of recent-level block worsens pooled (42.50 vs 42.38 full on same origins) — modest value once rolling means present.
