# Data Status — Google Mumbai Data Challenge 2026 (Mission 01 audit)

All raw files in `data/` are **preserved unmodified**. Audit run on the Nov-2 snapshot.

## Files

| File | Rows / content |
|---|---|
| `Train.csv` | 47,849 rows × 10 cols; 2025-02-19 03:00 → 2026-11-02 23:00 IST |
| `Test.csv` | 2,592 rows (= 4 stations × 648 h); 2026-10-22 → 2026-11-21 |
| `SampleSubmission.csv` | 2,592 IDs in Test order; target blank |
| `Stations.csv` | 4 stations (see below) |
| `DataDictionary.txt` | Definitions + evaluation schedule (below) |
| `AdditionalHistory/*` | auxiliary only — **not used** (IDs incomparable, no cross-map) |
| `Mumbai_Air_Quality_Starter.ipynb` | present, not used |

## Train schema (`DataDictionary.txt`)

`timestamp_ist` (hour-end, IST) · `pm25` (target, µg/m³) · `mean` (= pm25) · `count` (≥3 quarter-hour readings) · `available_at` (source file time / retrieval time — **not** a verified publication clock) · `location_id`, `sensor_id`, `station_name`, `series_id` · `flag_high_1000`.

## Stations

| location_id | station_name | sensor_id |
|---|---|---|
| 11611 | Malad West, Mumbai – IITM | 12236426 |
| 3409511 | Shivaji Nagar, Mumbai – BMC | 12238576 |
| 11606 | Borivali East, Mumbai – IITM | 12236418 |
| 3409510 | Byculla, Mumbai – BMC | 12238567 |

## Coverage & missingness

Rows per station: Borivali 12,101 · Malad 12,203 · Byculla 11,552 · Shivaji Nagar 11,993 (total 47,849). Hourly-grid coverage (14,925 slots/station): Borivali 81.1% (2,824 missing) · Malad 81.8% (2,722) · Byculla 77.4% (3,373) · Shivaji Nagar 80.4% (2,932). No duplicate (location, timestamp); no nulls in columns. Missing targets are **not imputed** in the raw release.

## Target / extremes

- Raw `pm25`: mean 43.99, std 1,311.77, median 21.84, max **241,587.56**.
- `flag_high_1000=True`: **19 rows** (every row >1,000; 57 rows >500), max 241,588 (Dec 3, 2025, Shivaji Nagar cluster; plus Borivali Aug 2025 clusters, Byculla May 2026). Treated as sensor anomalies: excluded (NaN) from features/labels for modeling.
- Clean (flag=False, n=47,830): mean **30.07**, std **32.10**, median **21.83**, p90 62.22, p95 75.87, p99 112.93, p99.9 466.83, max 928.83.
- Clean monthly means: Jan 49.2 · Feb 48.7 · Mar 39.4 · Apr 24.8 · May 20.5 · Jun 15.7 · Jul 14.3 · Aug **13.7** · Sep 16.8 · Oct **44.9** · Nov **68.5** · Dec 53.4 → strong winter peak; Diwali-season spike regime is unvalidated (see Mission 01).
- No impossible negatives in raw; negative/missing already excluded upstream.

## Latest observation & protocol

- **Last timestamp (all 4 stations): 2026-11-02 23:00 IST.** This is the sole forecast cutoff for the submission.
- **Evaluation schedule (2026, IST):** launch public 22–23 Oct (from Oct 26) → Oct 31 update: 27–28 Oct → Nov 5 update: **3–4 Nov** (revised window) → Nov 9 update: **5–6 Nov** → **private: 11–12 Nov only**. Only the active 2-day window scores per update. Rows outside these periods (including all unscored) do not contribute — but the full 2,592-row file is always required.
- **Test blocks (this snapshot):** public 768 (Oct 22–23, Oct 27–28, Nov 1–2, Nov 5–6) · private 192 (Nov 11–12) · unscored 1,632 (incl. gaps Oct 24–26, which also have no train coverage). Some pre-cutoff Test rows overlap train observations (482 public rows are already observed); the scored future rows (Nov 5–6: lead 24–97 h; Nov 11–12: lead **194–241 h**) do not. Gap is **not** a constant 72 h.
- **Scheduled releases:** Nov 9 and Nov 10 09:00 IST — future vs. this snapshot; do not assume a fixed gap.
- `available_at` is **not** treated as a verified first-publication/delay field; no availability-based leakage adjustment was possible.

## SampleSubmission & Test notes

- `Test.ID` format `MUM_<location>_<YYYYMMDDTHHMM>_IST`; no duplicate IDs. `unit` column present. `scoring_block` is launch allocation, **not** a modeling feature.
