# Extreme Sensitivity (M02.1 §8)

Targets 300–1,000 detected in validation (y max 708.8; 62 rows >100, 2 rows >300). Eval targets **preserved** (no silent deletion). 3 origins (Jun 2025, Apr 2026, Oct 2026; n=481).

| Training variant | RMSE |
|---|---|
| **Robust log1p target** | **11.02** |
| Raw (incl. >1,000) | 11.28 |
| >1,000-excluded, mean rolling (base) | 12.27 |
| Median rolling features | 14.16 |

Log/median take: raw slightly beats exclusion here; median rolling **hurts** (+1.9). Log is most robust. Caveat: subset excludes the Nov 17 extreme fold (79 RMSE overall), so raw may look better than it would under spikes. Keep >1,000 exclusion + consider log for final. Data: `extreme_oof.csv`.
