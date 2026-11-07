# Validation V3 (M02.2 §3–4)

- **Coverage:** 11 non-empty origins + 2026-01-05 empty (**0/192**, Jan 11–19 outage). 1,825 observed of 2,112 (86%). Full 48h windows, all hours, causal.
- **SE contribution (B3):** Nov 17 2025 origin = **65.5%** of total squared error (vs 58.8% in M02.1 review — single-fold dominance worsened with timestamp windows). Next: Feb-23 9.8%, Oct-05 7.7%.
- **Horizons:** private 194–241 (auto from Train.csv cutoff; fails if outside training lead domain — assertion in `mission02_2.py`); provisional refresh 100–170 (2,522 rows in `refresh_oof.csv`, RMSEs computed in analysis).
- **Seasonal:** Jun–Sep origins RMSE 6–11 (low); Nov–Feb 24–80 (high) — season dominates pooled.
- **Block bootstrap (origins, 1000×):** B3 23.60 CI [11.39, 36.16]; B2-fast 23.70 [12.19, 35.74]; RF 27.80 [11.36, 44.65]. Intervals overlap heavily — single pooled RMSE understates origin risk.

Data: `reports/m022/{baseline_oof.csv, refresh_oof.csv, window_audit.csv}`.
