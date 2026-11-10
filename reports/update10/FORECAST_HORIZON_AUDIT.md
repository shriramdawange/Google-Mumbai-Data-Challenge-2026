# UPDATE-10 Forecast Horizon Audit (Mission 03, Stage C)

**Date:** 2026-11-10
**Source:** `UPDATE-10/Test.csv`, `UPDATE-10/SampleSubmission.csv`, `UPDATE-10/Stations (1).csv`, `UPDATE-10/DataDictionary.txt`
**Interim training source (Train.csv absent at audit time):** `data/processed/UPDATE-10/training_dataset.parquet` (copy of UPDATE-09 parquet, 48,133 rows, max timestamp 2026-11-09 07:00:00 IST)

## 1. Latest training observation timestamp

- Interim source max: **2026-11-09 07:00:00 IST** (hour-end convention; 48,133 station-hours).
- Final release claim (DataDictionary.txt, SHA `8786b98f…`): *"10 November 2026 final release — Train contains 48,400 observed station-hours. The 7–8 November measurements are included as historical training observations. The final public window uses 201 verified station-hours from 8–10 November available at the release cutoff. These observations are also included in Train. The final private period is 11–12 November."*
- Because `UPDATE-10/Train.csv` was **absent** from the final release directory at audit time, the interim training cutoff of **2026-11-09 07:00 IST** was used for all horizon derivation. Re-run this stage after the final `Train.csv` lands.

## 2. Scoring allocation in Test.csv (actual file)

| scoring_block | rows | unique target_end hours | target_end range (IST) |
|---|---:|---:|---|
| private | 192 | 48 | 2026-11-11 01:00 → 2026-11-13 00:00 |
| public | 768 | 192 | 2026-10-22 01:00 → 2026-11-07 00:00 |
| unscored | 1632 | 408 | 2026-10-29 01:00 → 2026-11-21 00:00 |

IDs in Test.csv match `SampleSubmission.csv` IDs exactly and in order (2,592 rows each, no duplicates).

## 3. Derived private forecast horizon

Using interim train cutoff **T = 2026-11-09 07:00:00 IST** (hour-end convention):

- Private forecast leads (target_end − T, hours): **42 … 89** — **contiguous**, n = 48 unique hours.
- Private scoring period: 2026-11-11 01:00 → 2026-11-13 00:00 IST (48 hourly targets).

## 4. Horizon change vs UPDATE-09

- UPDATE-09 private horizon was also **T+42 … T+89** (from UPDATE-09 parquet cutoff 2026-11-09 07:00 IST).
- Test.csv SHA256 (`e7b9cc88…`) is byte-identical to UPDATE-09 Test.csv; SampleSubmission and Stations hashes also identical.
- Therefore the **final private forecast horizon is unchanged at T+42 … T+89** for the interim training source.

## 5. Conclusion

- The final private horizon is **contiguous T+42–T+89** (48 hours), 192 rows / 4 stations.
- The horizon derivation is fail-loud validated in `src/mission02_3_stage2.py::_validate_inputs` and in `src/update10_validate.py`.
- Any change in the final `Train.csv` cutoff must re-trigger Stage C and Stage D before final inference.
