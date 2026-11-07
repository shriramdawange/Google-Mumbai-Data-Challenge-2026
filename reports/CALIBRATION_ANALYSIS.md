# Calibration & Ablation (M02.1 §7)

**Calibration (y on pred, n=1,825):**

| Model | slope a | intercept b | Bias |
|---|---|---|---|
| General RF | 1.469 | −5.16 | −8.36 |
| Specialised RF | 0.858 | +10.35 | −5.89 |
| B3 | 1.144 | +2.67 | −7.02 |

General RF is under-dispersed (a=1.47); specialised over-shrinks (a=0.86). B3 closest to identity.

**Per-station bias (general RF):** Borivali −11.63, Malad −6.86, Byculla −1.29, Shivaji Nagar −13.48. Per-station RMSE (gen/B3): Byculla 15.99/18.83 (RF wins), others B3 wins (e.g. Shivaji 42.6 vs 38.0).

**Variation decomposition:** predicted mean by hour 26–31 (1 AM spike 41.6 = extreme-fold artifact); by lead bucket 28.4–29.8 (flat). Models learn almost no lead trend.

**Ablation (n=658, 4 origins; full RF on same origins = 42.38, extreme-fold dominated):**

| Dropped | RMSE |
|---|---|
| **lead** | **34.52 (improves 7.9!)** |
| longterm | 40.59 |
| recent | 42.50 |
| hour/diurnal | 42.65 |
| station | 43.02 |
| month | 43.32 |
| dow | 43.43 |

`lead`/`log_lead` **harm long-lead forecasts** — drop or gate them. All else helps mildly. Data: `ablation_oof.csv`.
