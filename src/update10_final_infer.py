from mission02_3_stage2 import final_infer

out, diag = final_infer(
    parquet_path="data/processed/UPDATE-10/training_dataset.parquet",
    test_path="UPDATE-10/Test.csv",
    selected_model="B2",
    out_path="submissions/candidate_update10_B2.csv",
    diag_path="reports/update10/submission_diagnostics_B2.json",
)
import json
print(json.dumps(diag, indent=2))
print(f"FINAL_INFER_B2_DONE rows={len(out)} mean={out['Target'].mean():.3f} min={out['Target'].min():.3f} max={out['Target'].max():.3f}")
