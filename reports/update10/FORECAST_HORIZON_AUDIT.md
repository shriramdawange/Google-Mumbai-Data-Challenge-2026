# UPDATE-10 Forecast Horizon Audit (Mission 03, Stage C / 03.1 RESUME)

**Date:** 2026-11-10 (updated 2026-11-10, Mission 03.1 RESUME)
**Sources:** `UPDATE-10/Train.csv` (actual final release, now present), `UPDATE-10/Test.csv`, `UPDATE-10/DataDictionary.txt`
**Training source for horizon derivation:** `data/processed/UPDATE-10/training_dataset.parquet` (regenerated from actual `UPDATE-10/Train.csv`)

## 1. Latest training observation timestamp

- Actual final `UPDATE-10/Train.csv`: cutoff **2026-11-10 09:00:00+05:30** (hour-end convention), 48,400 rows, span 2025-02-19 → 2026-11-10.
- Final-release claim reconciliation: file contains **48,400** observed station-hours ✓ (audit: `reports/update10/TRAIN_AUDIT.json`).

## 2. Scoring allocation in Test.csv (actual file)

| block | rows | unique target_end hours | target_end range (IST) |
|---|---:|---:|---|
| private | 192 | 48 | 2026-11-11 01:00 → 2026-11-13 00:00 |
| public | 768 | 192 | 2026-10-22 01:00 → 2026-11-07 00:00 |
| unscored | 1632 | 408 | 2026-10-29 01:00 → 2026-11-21 00:00 |

## 3. Re-derived private forecast horizon (from ACTUAL new training cutoff)

- Train cutoff T = **2026-11-10 09:00:00+05:30**.
- Private forecast leads (target_end − T, hours): **16 … 63** — contiguous, n = 48 hours, 192 rows / 4 stations.

## 4. Horizon change vs interim derivation

- Interim source (UPDATE-09 parquet, cutoff 2026-11-09 07:00): private leads **42–89**.
- Actual final source (UPDATE-10 Train.csv, cutoff 2026-11-10 09:00): private leads **16–63**.
- **Material shift confirmed** — final inference/validation MUST use the new horizon.

## 5. Conclusion

- Private horizon now derived from actual final training observations: **T+16 … T+63**.
- Re-run of validation and final inference required before producing the final candidate.
