# Residual V2 (M02.2 §6)

Baseline reconstructed **per training example at its own origin** (time-rolling 7/90d levels + 90d diurnal norm, definition identical to validation B3). Causality test: **PASS** — replacing all post-origin obs with 99,999 changed baseline by <1e−9. No future profile info.

Training: 2,492 examples (4 origins, weekly cutoffs, leads 194–241 step 12); in-sample base RMSE 26.85. Ridge(α=10) on [fast,slow,lead,log_lead,sin,cos,station]:

- **Ridge-V2 validation RMSE 28.69 vs B3 23.60** — residual still loses by 5.1.
- No tree residual run in V2 (M02.1 trees already +6–7 worse; Ridge was best of the three).

Conclusion (second independent confirmation): residual learning rejected for long leads.
