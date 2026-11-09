# Update 09 — Feature Engineering Validation

Existing architecture reused unchanged (`src/mission01.py: build_series/matrix`, 28 FULL features: station, calendar, lead, lags, rollings, city means; Mission-02.1/02.2 feature sets extend it without breaking it).

Verification executed: `build_series` ran against the new parquet-backed frame (cleaned copy) — 4 station series, lag1 non-null 80.7% (matches ~80% coverage). Temporal/causal definitions unchanged; no future leakage introduced; no test targets used.

Feature count: **28 (FULL)**. No pipeline logic modified — only data source swapped to the versioned parquet. Detail: `reports/update09/feature_validation.json`.
