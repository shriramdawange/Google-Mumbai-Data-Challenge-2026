# Persistence Blending (Mission 02)

Blends: `w·RF + (1−w)·persistence` and `w·RF + (1−w)·rollmean24`, w∈{0,0.1,…,1}. Honest protocol: **optimize w on early folds (Dec, Jan, Aug) only**, then apply to late folds (Oct 5, Oct 20).

## Early-fold optima (pooled F0–F2, n=53)

- vs persistence: **w=0.1** (RMSE 27.70) — 90% persistence.
- vs rollmean: **w=0.0** (RMSE 21.73) — pure rolling mean.

Both optima say: trust RF ~0% on early data.

## Late-fold check (n=47)

| Blend | Early RMSE | Late RMSE | Pooled |
|---|---|---|---|
| w=0.9·RF + 0.1·persist | 31.48 | 16.65 | 25.60 |
| w=0.75·RF + 0.25·rollmean | 28.83 | 16.72 | 23.92 |
| w=0.5·RF + 0.5·rollmean | 25.70 | 17.31 | 22.16 |
| Pure RF (w=1) | 32.45 | **16.90** | 26.31 |
| Pure rollmean (w=0 rollmean) | 21.73* | ~31* | 21.15 |

*Pure-rollmean pooled is 21.15, but its late-fold RMSE is worse than pure RF here (early and late regimes disagree). The honest early-optimal w does **not** transfer: w=0.1 persist would score far worse on late folds than any late-oracle w≥0.5.

## Per-lead oracle (diagnostic only, all folds)

| Lead | Oracle w (vs persist) |
|---|---|
| 24 | 0.60 |
| 48 | 0.75 |
| 72 | 0.00 |
| 120 | 1.00 |
| 168 | 0.60 |
| 194 | 0.60 |
| 216 | 0.90 |
| 240 | 0.10 |
| 241 | 1.00 |

No consistent w; optimal flips sign across leads (n=9–16 each).

## Conclusion

Blending is **not recommended** as-tuned: honest chronological optimization fails (early says 0–0.1, late wants ≥0.5), and oracle w is lead-unstable. If blending is later used, fix w per lead from a large-n seasonal study, never from pooled early folds.
