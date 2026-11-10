# UPDATE-10 Submission Candidate Validation (Mission 03 / 03.1 RESUME)

- Candidate: `submissions/candidate_update10_HYBRID.csv`
- SHA256: `d41905b7a0cfb2000fefcbb6b6bfd8f9e812695a03bc0e8b3319d1dbf90c8089`

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
- Target min: 5.7275
- Target max: 794.7700
- Target mean: 66.5300
- private rows covered: 192 / 192

## Submission diagnostics

{
  "candidate": "submissions/candidate_update10_HYBRID.csv",
  "source_candidate": "submissions/candidate_update10_B2.csv",
  "selected_model": "B2",
  "hybrid": true,
  "hybrid_method": "public_window_exact_lookup",
  "public_window_start": "2026-11-08 00:00:00+05:30",
  "public_window_end": "2026-11-11 00:00:00+05:30",
  "train_max": "2026-11-10 09:00:00+05:30",
  "private_rows": 192,
  "private_leads": [
    16,
    63
  ],
  "inference_target_rows": 2592,
  "inference_full_rows": 2592,
  "exact_training_matches_applied": 201,
  "unmatched_public_rows": 87,
  "model_prediction_rows_used": 2391,
  "lookup_counts_by_station": {
    "11606": 52,
    "11611": 51,
    "3409510": 48,
    "3409511": 50
  }
}

