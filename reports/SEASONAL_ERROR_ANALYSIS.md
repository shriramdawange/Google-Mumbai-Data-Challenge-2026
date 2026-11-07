# Seasonal & Extreme Error Analysis (Mission 02)

Winter origins: Dec 15 2025 (n=23) + Jan 15 2026 (n=16). Winter y: mean 51.2, median 44.1, max 121.3 (vs long-horizon pooled mean ~45). M01's 1–72 h OOF (Aug–Oct) had mean 33.8, max 86.8, **0 rows >100** — winter episodes were untested.

## Extreme cohorts (long-horizon OOF, n=100)

| Cohort | n | RF | Persistence | Roll24 |
|---|---|---|---|---|
| y > 75 | 10 | 33.23 | 49.28 | 36.96 |
| y > 100 | 1 | 20.28 | 21.11 | **8.76** |
| y > 150 | 0 | — | — | — |

Sample is tiny (one >100 case); do not over-claim. Directionally, all methods degrade 2–4× on high episodes; persistence degrades most.

## Winter vs summer

- F0 (Dec): RF blows up at 72 h (63.7) and 240 h (74.7). Roll24 stays 16–31.
- F1 (Jan): mostly flat across methods at 194–240 h (RF 6–18 vs persist 7–51).
- F2 (Aug): all methods ≤6 RMSE — summer low episodes are easy.

Implication: RMSE differences are dominated by regime (winter), not lead alone. A pooled RMSE hides this; the private Nov 11–12 window is winter (Nov clean mean 68.5), so **winter folds are the most relevant proxy despite n=39**. Model choice for winter = roll24-leaning; RF spikes risk.

**No fabricated extremes:** >150 RMSE reported as n=0. The 19 train anomalies (>1,000) were excluded from targets/features and never scored.
