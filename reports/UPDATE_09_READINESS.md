# Update 09 — Training Readiness

**Ready:** versioned training dataset (`data/processed/UPDATE-09/training_dataset.parquet`, 48,133 rows) is pipeline-compatible; Test/Sample frozen; baselines preserved.

**New operating point:** cutoff Nov-9 07:00 IST → private leads **42–89 h** (vs 194–241 before). This is the correct regime to re-validate for. The Nov-10 09:00 update is still pending — expect cutoff ≈ Nov-10 and leads ≈ 33–80 h.

**Do NOT (yet):** overwrite `submissions/baseline_v0.csv`, launch full training, change validation methodology ad hoc.

**Exact final commands (after Nov-10 release lands in `UPDATE-10/`):**
```bash
python src/update09_integrate.py          # 1. re-ingest UPDATE-10 (adapt script paths)
python src/mission02_2.py                 # 2. re-validate baselines at new auto leads
python src/finalize.py                    # 3. retrain on versioned parquet + build submission
```
(For a B4 submission: generate with B4 formula from `reports/m022/shrinkage.json`, not RF.)

**Unresolved (human review):**
1. Nov-10 files not yet present — re-run this integration on `UPDATE-10/` and re-derive leads; confirm `DataDictionary.txt` scoring windows.
2. Dec/junk-value handling (>1,000) untouched at ingestion — decide log vs exclude before final (M02.1 best: log).
3. `.gitignore` still absent — prior commit included `src/__pycache__/*.pyc`; add an ignore before the next commit (no commit made here per instructions).
