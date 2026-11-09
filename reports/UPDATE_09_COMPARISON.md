# Update 09 — Previous-vs-New Comparison

Key: `(location_id, timestamp_ist)`. Old `data/Train.csv` (47,849) vs `UPDATE-09/Train.csv` (48,133).

- **Added 284 rows, removed 0, common 47,849, revised (`pm25` differs) 0.** Clean append; no blind concatenation needed (new is superset).
- Added by station: Borivali 71, Malad 73, Byculla 68, Shivaji Nagar 72.
- Added range: **2026-11-03 02:00 → 2026-11-09 07:00 IST** (the Nov-9 release: 5–6 Nov public now observed; notes say active public 7–8 Nov, 171/192 valid).
- Coverage: old cutoff Nov-2 23:00 → **new cutoff Nov-9 07:00**. Private (Nov 11–12) leads: **42–89 h** (was 194–241).
- Schema: no additions/removals; same 10 cols, same dtypes.
- PM2.5: mean 43.99→44.03, median 21.84→21.96, max unchanged 241,587.56; >1,000 still 19. `flag_high_1000` unchanged.
- Duplicates (new file): 0. Test/Sample/Stations: byte-identical.

Full numbers: `reports/update09/comparison_report.json`.
