# UPDATE-10 Submission Candidate Validation (Mission 03 / 03.1 RESUME)

- Candidate: `submissions/candidate_update10_final.csv`
- SHA256: `e059e4ba073e7743d0802330f4ffa1128fc285c5c0b068721c3d11c153ac4a40`

## Independent validation checks

| check | status |
|---|---|
| row_count_ok | PASS |
| columns_ok | PASS |
| id_sequence_ok | PASS |
| id_unique_ok | PASS |
| missing_predictions_ok | PASS |
| nan_inf_ok | PASS |
| nonnegative_ok | PASS |
| private_scored_ids_covered | PASS |
| public_scored_ids_covered | PASS |
| all_test_ids_present | PASS |
| target_range_ok | PASS |

## Candidate summary

- rows: 2592
- columns: ['ID', 'Target']
- Target min: 27.9872
- Target max: 75.0013
- Target mean: 41.7766
- private rows covered: 192 / 192

## Submission diagnostics

{
  "selected_model": "B3",
  "parquet_path": "data/processed/UPDATE-10/training_dataset.parquet",
  "test_path": "UPDATE-10/Test.csv",
  "train_max": "2026-11-10 09:00:00+05:30",
  "private_rows": 192,
  "private_hours": 48,
  "private_leads": [
    16,
    63
  ],
  "inference_target_rows": 2592,
  "inference_full_rows": 2592,
  "inference_mean": 41.776628029052965,
  "inference_min": 27.987213214577622,
  "inference_max": 75.00126426870966,
  "supported_models": [
    "B2",
    "B3",
    "B4"
  ],
  "supported_leads": [
    1,
    400
  ],
  "model_in_supported_set": true
}

