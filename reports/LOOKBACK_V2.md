# Lookback V2 (M02.1 §5, 72h-lag bug fixed)

Fixed lag72 (was missing in M02). Same 6 origins (n=980), leads 194–241, single-window feature set.

| W (h) | Pooled RMSE |
|---|---|
| **72** | **14.99** |
| 48 | 15.56 |
| 120 | 15.92 |
| 168 | 16.27 |
| 336 | 16.52 |
| 6 | 16.68 |
| 12 | 19.73 |
| 24 | 21.40 |

Per-origin: short windows fail on Oct 19 (W12 40.2, W24 43.9 vs W72 18.2). W72 wins pooled; W36 not tested. All single-window RFs beat the multi-window full RF's same-origin subset? No — different origins/count; interpret as: keep 48–72h window, avoid 12–24h alone. Data: `lookback2_oof.csv` (7,840 rows).
