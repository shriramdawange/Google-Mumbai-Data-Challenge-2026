# Update 09 — Data Inventory

`UPDATE-09/` (7 files, read-only). Raw SHA256 in `reports/update09/update09_inventory.csv`.

| File | Size | Rows | Schema |
|---|---|---|---|
| Train.csv | 6,797,848 | 48,133 | 10 cols (same as old) |
| Test.csv | 296,429 | 2,592 | 7 cols — **identical SHA to old** |
| SampleSubmission.csv | 80,363 | 2,592 | 2 cols — **identical SHA** |
| Stations.csv | 369 | 4 | 6 cols — **identical SHA** |
| DataDictionary.txt | 3,499 | — | **changed** (Nov-9 release notes) |
| Mumbai_Air_Quality_Starter.ipynb | 376,317 | — | identical |
| AdditionalHistory.zip | 4,817,970 | — | identical |

No README in either version. DataDictionary delta: Nov-9 release (48,133 rows; active public **7–8 Nov**, 171/192 valid; public+private excluded from Train).
