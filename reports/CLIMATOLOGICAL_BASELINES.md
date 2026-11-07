# Climatological / Statistical Baselines (M02.1 §3)

All computed **only from history ≤ origin** (no future). Same 11 origins / 194–241h window.

- **B0** station×hour profile (all history): RMSE **33.00** — ignores level.
- **B1** mean of 7/14/30-day recent means: **26.43**.
- **B2** last-24h level × normalised diurnal profile: **25.70**, bias −3.37 (most unbiased).
- **B3** 30-day seasonal × diurnal, 50% recent-level adjusted: **25.57** (best).

Baselines beat RF (27.80) and persistence (25.86). B2/B3 win because they track **recent level** while diurnal structure remains. B0's failure shows level dominates diurnal at 200h+.

Reference for residual learning: **B3** (best). Data: `contiguous_oof.csv` `pred_B*`.
