# UPDATE-10 Model Selection (Mission 03, Stage D)

## Setup

- Training source: interim UPDATE-10 dataset (`data/processed/UPDATE-10/training_dataset.parquet`), cutoff 2026-11-10 09:00:00+05:30
- Test: `UPDATE-10/Test.csv` (2592 rows, private 192 rows / 48 unique hours)
- Private forecast leads: 16..63 (contiguous, n=48)

## Pooled UPDATE-10 validation metrics (short-horizon re-run)

| model | RMSE | MAE | bias | n |
|---|---:|---:|---:|---:|
| B2 | 20.6272 | 12.9879 | 1.6703 | 1896 |
| B3 | 19.7565 | 14.0992 | 4.1156 | 1896 |
| B4 | 19.7565 | 14.0992 | 4.1156 | 1896 |
| pred_rf | 51.8160 | 19.0047 | -7.5831 | 1896 |

**Best pooled model: B3** (RMSE 19.7565).

## Paired bootstrap (10k origin-block, this run)

{
  "B2_vs_B3": {
    "model_A": "B2",
    "model_B": "B3",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": -0.8706736542032232,
    "ci": [
      -5.527369392643567,
      3.138368397484731
    ],
    "p_A_better": 0.3787
  },
  "B2_vs_B4": {
    "model_A": "B2",
    "model_B": "B4",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": -0.8706736542032232,
    "ci": [
      -5.527369392643567,
      3.138368397484731
    ],
    "p_A_better": 0.3787
  },
  "B2_vs_RF": {
    "model_A": "B2",
    "model_B": "RF",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 31.188825657808714,
    "ci": [
      -2.2102876780825276,
      62.79661283063943
    ],
    "p_A_better": 0.7259
  },
  "B3_vs_RF": {
    "model_A": "B3",
    "model_B": "RF",
    "convention": "sqrtMSE(B)-sqrtMSE(A); positive => A better",
    "observed_diff": 32.05949931201194,
    "ci": [
      -4.495748611125891,
      68.17511317349337
    ],
    "p_A_better": 0.6758
  }
}

Model RMSE bootstrap CIs:
{"B2": [12.371643034066723, 27.885498118213953], "B3": [13.350680784424307, 26.31299014713174], "B4": [13.350680784424307, 26.31299014713174], "RF": [12.446667725049831, 86.75625958929739]}


RMSE: {"B2": 20.627200150351193, "B3": 19.75652649614797, "B4": 19.75652649614797, "RF": 51.81602580815991}

## Decision

The selected model for the final UPDATE-10 inference is **B3** based on pooled RMSE with the
short-horizon validation re-run on the actual UPDATE-10 training source. See `reports/update10/FORECAST_HORIZON_AUDIT.md`
and `reports/MISSION_03_FINAL_READINESS.md` for the full chain of evidence.
