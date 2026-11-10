# UPDATE-10 Submission Candidate Validation (Mission 03, Stage G)

- Candidate: `submissions/candidate_update10.csv`
- SHA256: `823222a2704d7a56b5497c2c52aacb17adfd6092af37ae6ed77416594c3c6148`

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
- Target min: 28.6361
- Target max: 81.0109
- Target mean: 46.9706
- private rows covered: 192 / 192

## Submission diagnostics

{
  "selected_model": "B2",
  "parquet_path": "data/processed/UPDATE-10/training_dataset.parquet",
  "test_path": "UPDATE-10/Test.csv",
  "train_max": "2026-11-09 07:00:00+05:30",
  "private_rows": 192,
  "private_hours": 48,
  "private_leads": [
    42,
    89
  ],
  "inference_target_rows": 2592,
  "inference_full_rows": 2592,
  "inference_mean": 46.970588257336395,
  "inference_min": 28.636096838886818,
  "inference_max": 81.01092687348174,
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

