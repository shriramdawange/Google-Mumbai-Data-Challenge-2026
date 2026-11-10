"""MISSION 03.1 RESUME — Step 4: audit the actual UPDATE-10/Train.csv."""
import os, json, hashlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "UPDATE-10", "Train.csv")
RPT = os.path.join(ROOT, "reports", "update10")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

df = pd.read_csv(P, parse_dates=["timestamp_ist", "available_at"])
audit = dict(
    path=P, exists=True, sha256=sha(P), size_bytes=int(os.path.getsize(P)),
    raw_shape=[int(df.shape[0]), int(df.shape[1])], columns=list(df.columns),
    timestamp_min=str(df["timestamp_ist"].min()), timestamp_max=str(df["timestamp_ist"].max()),
    n_stations=int(df["location_id"].nunique()), stations=sorted(df["location_id"].unique().tolist()),
    n_rows=int(len(df)),
    dup_keys=int(df.duplicated(["location_id", "timestamp_ist"]).sum()),
    nulls={c: int(df[c].isna().sum()) for c in df.columns},
    pm25_min=float(df["pm25"].min()), pm25_max=float(df["pm25"].max()),
    pm25_negative=int((df["pm25"] < 0).sum()), pm25_mean=float(df["pm25"].mean()),
    flag_high_1000=int(df["flag_high_1000"].sum()) if "flag_high_1000" in df.columns else None,
    count_min=int(df["count"].min()) if "count" in df.columns else None,
)
with open(os.path.join(RPT, "TRAIN_AUDIT.json"), "w") as f: json.dump(audit, f, indent=2)
print(json.dumps(audit, indent=2))
print("TRAIN_AUDIT_DONE")
