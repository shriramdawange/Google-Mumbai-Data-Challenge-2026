# Baseline V3 — B0–B4 with timestamp windows (M02.2)

Windows are now **calendar-time** (not last-N-obs). Audit (288 station×origin×window cells):

| Window | mean n | missing % |
|---|---|---|
| 3d | 59.7 | 17.1% |
| 7d | 138.3 | 17.7% |
| 14d | 284.2 | 15.4% |
| 30d | 601.3 | 16.5% |
| 60d | 1,166.6 | 19.0% |
| 90d | 1,732.2 | 19.8% |

Missing readings are skipped, never zero-filled. Full detail: `reports/m022/window_audit.csv`.

**Definitions:** B0 station×hour climatology (expanding history); B1 recent mean (calendar); B2 recent level × normalised 90d diurnal; B3 `w·fast+(1−w)·slow` (7d vs 90d levels × same diurnal, w from early folds); B4 B3 + β·trend (β from early folds).

Pooled 194–241h (n=1,825): **B4 23.42 < B3 23.60 < B2-fast 23.70 < B0 33.00**. Timestamp windows improve M02.1 B3 25.57 → **23.60**. Bias: B4 −5.45, B3 −6.34 (B3 bias still worse than B2 in M02.1; narrowing). RF 27.80 loses to all B2–B4.
