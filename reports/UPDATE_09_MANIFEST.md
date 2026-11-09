# Update 09 — Dataset Manifest (SHA256)

Sources (read-only):
- `data/Train.csv` (Nov-2 snapshot, 47,849)
- `UPDATE-09/Train.csv` SHA `11d6c4a4…41047ee` (48,133; Nov-3→Nov-9 +284)
- `UPDATE-09/Test.csv` `e7b9cc88…` (identical to old), `SampleSubmission.csv` `ee4682ce…`, `Stations.csv` `383269fa…`

Output: `data/processed/UPDATE-09/training_dataset.parquet` (48,133×11) + `dataset_manifest.json` (lineage + output SHA256), `comparison_report.json`, `data_quality_report.json`, plus `reports/update09/*` (inventory, tests, feature smoke, readiness).

Test.csv / SampleSubmission.csv **unchanged** (2,592 IDs, original order). Baselines preserved.
