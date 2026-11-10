# UPDATE-10 Model Selection (Mission 03, Stage D)

## Setup

- Training source: interim UPDATE-10 dataset (`data/processed/UPDATE-10/training_dataset.parquet`), cutoff 2026-11-09 07:00:00+05:30
- Test: `UPDATE-10/Test.csv` (2592 rows, private 192 rows / 48 unique hours)
- Private forecast leads: 42..89 (contiguous, n=48)

## Pooled UPDATE-10 validation metrics (short-horizon re-run)

| model | RMSE | MAE | bias | n |
|---|---:|---:|---:|---:|
| B2 | 23.2525 | 15.0282 | 2.2588 | 1816 |
| B3 | 24.0942 | 16.4588 | 3.4017 | 1816 |
| B4 | 24.0942 | 16.4588 | 3.4017 | 1816 |
| pred_rf | 41.0465 | 19.1097 | -5.6196 | 1816 |

**Best pooled model: B2** (RMSE 23.2525).

## Paired bootstrap (10k origin-block, this run)

{
  "B2_vs_B3": {
    "model_A": "B2",
    "model_B": "B3",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 0.841701378308656,
    "ci": [
      -4.169449072786551,
      5.709515489452677
    ],
    "p_A_better": 0.6379
  },
  "B2_vs_B4": {
    "model_A": "B2",
    "model_B": "B4",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 0.841701378308656,
    "ci": [
      -4.169449072786551,
      5.709515489452677
    ],
    "p_A_better": 0.6379
  },
  "B2_vs_RF": {
    "model_A": "B2",
    "model_B": "RF",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 17.794052191401413,
    "ci": [
      -2.1235985804301722,
      41.663629069792826
    ],
    "p_A_better": 0.8047
  },
  "B3_vs_RF": {
    "model_A": "B3",
    "model_B": "RF",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 16.952350813092757,
    "ci": [
      -6.094016447386409,
      45.23218646765793
    ],
    "p_A_better": 0.6899
  }
}

Model RMSE bootstrap CIs:
{"B2": [13.263153570489367, 32.65867031407384], "B3": [13.39387130242295, 33.42441868971166], "B4": [13.39387130242295, 33.42441868971166], "RF": [14.62240138500467, 66.83591209402864]}


RMSE: {"B2": 23.252490568434332, "B3": 24.09419194674299, "B4": 24.09419194674299, "RF": 41.046542759835745}

## Decision

The selected model for the final UPDATE-10 inference is **B2** based on pooled RMSE with the
short-horizon validation re-run on the actual UPDATE-10 training source. See `reports/update10/FORECAST_HORIZON_AUDIT.md`
and `reports/MISSION_03_FINAL_READINESS.md` for the full chain of evidence.
