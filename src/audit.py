import pandas as pd, numpy as np, hashlib
tr = pd.read_csv("data/Train.csv", parse_dates=["timestamp_ist"])
te = pd.read_csv("data/Test.csv", parse_dates=["target_start_ist", "target_end_ist"])
sub = pd.read_csv("submissions/baseline_v0.csv")
clean = tr[~tr.flag_high_1000]
print("raw mean", tr.pm25.mean(), "clean mean", clean.pm25.mean(), "clean std", clean.pm25.std())
print(clean.pm25.quantile([0.5, 0.9, 0.95, 0.99, 0.999]))
print("count>1000", (tr.pm25 > 1000).sum(), "flag true", tr.flag_high_1000.sum())
clean2 = clean.copy()
clean2["m"] = pd.to_datetime(clean2.timestamp_ist).dt.month
print(clean2.groupby("m").pm25.mean().to_string())
print(clean.groupby("location_id").pm25.agg(["mean", "std", "median", "max", "count"]).to_string())
for loc, g in tr.groupby("location_id"):
    s = g.set_index("timestamp_ist")["pm25"].asfreq("h")
    print(loc, "slots", len(s), "obs", int(s.notna().sum()), "miss", int(s.isna().sum()), "cov", round(float(s.notna().mean()), 4))
te["lead"] = (pd.to_datetime(te.target_end_ist) - pd.Timestamp("2026-11-02 23:00:00+05:30")).dt.total_seconds() / 3600
print(te.groupby("scoring_block")["lead"].agg(["count", "min", "max"]))
print("test range", te.target_start_ist.min(), te.target_end_ist.max())
data = open("submissions/baseline_v0.csv", "rb").read()
print("sha256", hashlib.sha256(data).hexdigest(), "bytes", len(data))
print(open("submissions/baseline_v0.csv").readline().strip())
oof = pd.read_csv("reports/oof_validation.csv")
print("oof lead", oof.lead.min(), oof.lead.max(), len(oof))
print(oof[["y", "pred_rf", "pred_hgb"]].describe().to_string())
