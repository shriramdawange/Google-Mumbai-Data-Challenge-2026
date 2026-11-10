from mission02_3_stage2 import final_infer

# Preserve the historical interim candidate: B2 from interim UPDATE-09-parquet-equivalent source, leads 42-89
out, diag = final_infer(
    parquet_path="data/processed/UPDATE-09/training_dataset.parquet",
    test_path="UPDATE-10/Test.csv",
    selected_model="B2",
    out_path="submissions/candidate_update10.csv",
    diag_path="reports/update10/submission_diagnostics.json",
)
import json
print(json.dumps(diag, indent=2))
print(f"OLD_CANDIDATE_RESTORED rows={len(out)} mean={out['Target'].mean():.3f} min={out['Target'].min():.3f} max={out['Target'].max():.3f}")
