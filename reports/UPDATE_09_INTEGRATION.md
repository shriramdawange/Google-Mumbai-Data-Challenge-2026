# Update 09 — Integration Report

- `UPDATE-09/` and `data/` **preserved unchanged** (read-only access). No overwrite of datasets/artifacts/submissions.
- Versioned output: **`data/processed/UPDATE-09/`** (new convention, does not clash with existing `data/` flat layout):
  - `training_dataset.parquet` (48,133 rows × 11 cols) — union of releases
  - `dataset_manifest.json`, `data_quality_report.json`, `comparison_report.json`
- **Union rule:** new wins on key conflict; old kept only if key missing in new. In this release new ⊇ old (0 removed, 0 revised) so output = new + `source_release=UPDATE-09`. `pm25>1000` kept raw; pipeline cleaning unchanged.
- Reproduction: `python src/update09_integrate.py`.

Lineage + SHA256: `data/processed/UPDATE-09/dataset_manifest.json`.
