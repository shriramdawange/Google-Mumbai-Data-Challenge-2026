"""MISSION 03 Stage G: independent validation of submissions/candidate_update10.csv against UPDATE-10 Test.csv."""
import os, json, hashlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "submissions", "candidate_update10.csv")
TEST = os.path.join(ROOT, "UPDATE-10", "Test.csv")
RPT = os.path.join(ROOT, "reports", "update10")
V = os.path.join(RPT, "validation")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

cand = pd.read_csv(CAND)
te = pd.read_csv(TEST, parse_dates=["target_start_ist", "target_end_ist"])
checks = {}
checks["row_count_ok"] = (len(cand) == len(te))
checks["columns_ok"] = (list(cand.columns) == ["ID", "Target"])
checks["id_sequence_ok"] = (cand["ID"].astype(str).values == te["ID"].astype(str).values).all()
checks["id_unique_ok"] = bool(cand["ID"].is_unique and te["ID"].is_unique)
checks["missing_predictions_ok"] = (cand["Target"].notna().all() and len(cand) == len(te))
checks["nan_inf_ok"] = bool(np.isfinite(cand["Target"].to_numpy(dtype=float)).all())
checks["nonnegative_ok"] = bool((cand["Target"].to_numpy(dtype=float) >= 0).all())
cand_ids = set(cand["ID"].astype(str))
priv_ids = set(te.loc[te["scoring_block"] == "private", "ID"].astype(str).tolist())
pub_ids = set(te.loc[te["scoring_block"] == "public", "ID"].astype(str).tolist())
checks["private_scored_ids_covered"] = bool(priv_ids.issubset(cand_ids))
checks["public_scored_ids_covered"] = bool(pub_ids.issubset(cand_ids))
checks["all_test_ids_present"] = bool(cand_ids >= set(te["ID"].astype(str).tolist()))
t = cand["Target"].to_numpy(dtype=float)
checks["target_range_ok"] = bool((t >= 0).all() and np.isfinite(t).all())
diag = json.load(open(os.path.join(RPT, "submission_diagnostics.json")))
lines = ["# UPDATE-10 Submission Candidate Validation (Mission 03, Stage G)", "",
         f"- Candidate: `submissions/candidate_update10.csv`", f"- SHA256: `{sha(CAND)}`", "",
         "## Independent validation checks", "",
         "| check | status |", "|---|---|"]
for k, v in checks.items():
    lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
lines += ["", "## Candidate summary", "",
          f"- rows: {len(cand)}", f"- columns: {list(cand.columns)}",
          f"- Target min: {float(np.min(t)):.4f}", f"- Target max: {float(np.max(t)):.4f}",
          f"- Target mean: {float(np.mean(t)):.4f}",
          f"- private rows covered: {len(priv_ids & cand_ids)} / {len(priv_ids)}", "",
          "## Submission diagnostics", "",
          json.dumps(diag, indent=2), "", ""]
open(os.path.join(RPT, "SUBMISSION_VALIDATION.md"), "w").write("\n".join(lines))
json.dump(dict(candidate_sha256=sha(CAND), checks={k: bool(v) for k, v in checks.items()}, target_min=float(np.min(t)), target_max=float(np.max(t)), target_mean=float(np.mean(t)), rows=int(len(cand))), open(os.path.join(RPT, "submission_validation.json"), "w"), indent=2)
print("\n".join(lines))
print("ALL_PASS=", all(checks.values()))
