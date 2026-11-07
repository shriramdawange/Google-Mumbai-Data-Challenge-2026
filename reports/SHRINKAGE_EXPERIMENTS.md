# Shrinkage (M02.1 50% was wrong)

w fit on early folds only (5 origins), applied to all:

- **Pooled w = 1.00** (pure fast 7d). Early 30.69, late 14.76.
- Station: 11606 1.00, 11611 1.00, 3409510 0.95, 3409511 1.00.
- Lead buckets: 194–210 0.85, 211–225 1.00, 226–241 1.00.
- **50% is not optimal** — shrinkage toward slow (90d) uniformly hurts; station/lead splits barely differ (B3_station 23.60, B3_leaddep 23.65 vs pooled 23.60).
- B4 trend β = +0.15 (early 30.45, late 14.66): B4 23.42 edges B3 by 0.18.

Configs: `reports/m022/shrinkage.json`.
