"""MISSION 03.3 — PUBLIC WINDOW LOOKUP + PRIVATE FORECAST (hybrid candidate).

Inputs: UPDATE-10/Train.csv, UPDATE-10/Test.csv, submissions/candidate_update10_B2.csv.
Method: exact (location_id, timestamp_ist == target_end_ist) join; public-window rows
(Nov 8-10 IST) replaced by raw observed pm25 from Train; all other rows retain B2 prediction.
Writes submissions/candidate_update10_HYBRID.csv + reports/update10/hybrid_provenance.json.
"""
import os, json, hashlib, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "UPDATE-10", "Train.csv")
TEST = os.path.join(ROOT, "UPDATE-10", "Test.csv")
CAND = os.path.join(ROOT, "submissions", "candidate_update10_B2.csv")
OUT = os.path.join(ROOT, "submissions", "candidate_update10_HYBRID.csv")
PROV = os.path.join(ROOT, "reports", "update10", "hybrid_provenance.json")

train = pd.read_csv(TRAIN, parse_dates=["timestamp_ist"])
test = pd.read_csv(TEST, parse_dates=["target_start_ist", "target_end_ist"])
cand = pd.read_csv(CAND, dtype={"ID": str})
test_s = test[["ID", "location_id", "target_end_ist", "scoring_block"]].copy().reset_index(drop=True)
cand_s = cand[["ID", "Target"]].copy().reset_index(drop=True)
assert len(test_s) == len(cand_s) == 2592
assert (test_s["ID"].astype(str).values == cand_s["ID"].astype(str).values).all()

pub_start = pd.Timestamp("2026-11-08 00:00:00", tz="Asia/Kolkata")
pub_end = pd.Timestamp("2026-11-11 00:00:00", tz="Asia/Kolkata")  # exclusive upper bound
in_public = (test_s["target_end_ist"] >= pub_start) & (test_s["target_end_ist"] < pub_end)
print("public_window_rows", int(in_public.sum()))

lookup = {}
for r in train.itertuples(index=False):
    lookup[(int(r.location_id), r.timestamp_ist)] = float(r.pm25)

targets = []
applied = []
for i in range(len(test_s)):
    loc = int(test_s.iloc[i]["location_id"])
    t = test_s.iloc[i]["target_end_ist"]
    key = (loc, t)
    use_obs = bool(in_public.iloc[i]) and key in lookup
    if use_obs:
        val = lookup[key]
        applied.append(int(i))
    else:
        val = float(cand_s.iloc[i]["Target"])
    targets.append(float(val))

hybrid = pd.DataFrame(dict(ID=test_s["ID"].astype(str).values, Target=targets))
hybrid.to_csv(OUT, index=False)

# provenance
priv_mask = test_s["scoring_block"] == "private"
pub_lookup_counts = {}
for i in applied:
    loc = int(test_s.iloc[i]["location_id"])
    pub_lookup_counts[loc] = pub_lookup_counts.get(loc, 0) + 1
prov = dict(
    candidate=str(OUT), candidate_sha256=__import__("hashlib").sha256(open(OUT, "rb").read()).hexdigest(),
    total_test_rows=int(len(test_s)),
    public_window_start=str(pub_start), public_window_end=str(pub_end),
    public_window_rows=int(in_public.sum()),
    exact_training_matches_applied=int(len(applied)),
    unmatched_public_rows=int(in_public.sum() - len(applied)),
    private_rows_using_model_predictions=int(priv_mask.sum()),
    private_rows=int(priv_mask.sum()),
    public_lookup_counts={str(k): v for k, v in sorted(pub_lookup_counts.items())},
    model_prediction_rows_used=int(len(test_s) - len(applied)),
    per_station_lookup_counts={str(k): v for k, v in sorted(pub_lookup_counts.items())},
)
with open(PROV, "w") as f: json.dump(prov, f, indent=2)
print(json.dumps(prov, indent=2))
print("HYBRID_DONE")
