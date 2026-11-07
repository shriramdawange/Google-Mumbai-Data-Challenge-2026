# Mission 02.2 Results — Statistical Forecasting Optimisation

**Date:** 2026-11-07 · M01/M02/M02.1 artifacts frozen (no overwrite, no submission created). Code: `src/mission02_2.py`, `src/mission02_2_stage2.py`; OOF in `reports/m022/`.

## Findings

1. **Timestamp windows fix +2 points:** M02.1 B3 (count windows) 25.57 → M02.2 B3 (calendar) **23.60**; B4 (trend) **23.42** best. B0 33.00.
2. **Shrinkage w=1.0 (pure fast 7d), not 50%.** Station/lead splits ≈ pooled. β(trend)=+0.15.
3. **Single origin dominates:** Nov-17 SE share **65.5%** (was 58.8%). Block-bootstrap CIs all overlap: B3 [11.4,36.2], RF [11.4,44.7].
4. **RF still loses:** 27.80 vs B4 23.42 (−19% relative skill negative). Residual V2 (causal, PASS) 28.69 vs 23.60 — rejected twice.
5. **Diurnal:** multiplicative station-specific retained (amplitudes 5–14); additive B0 fails.
6. **Horizons:** 194–241 auto-validated; 100–170 provisional scored (2,522 rows). Domain assertion enforced.

## Final decision

**B4 (B3 + 0.15·trend, timestamp windows) is the strongest model for T+194–241.** B3-close second. Recommend B4 as final statistical forecaster; next work: reduce Nov-origin dominance (robust/quantile or log-target), then validated submission — not GNN.
