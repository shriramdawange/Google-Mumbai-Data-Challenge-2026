import os, json, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
te = pd.read_csv(os.path.join(ROOT, "UPDATE-10", "Test.csv"), parse_dates=["target_start_ist", "target_end_ist"])
tr = pd.read_parquet(os.path.join(ROOT, "data", "processed", "UPDATE-10", "training_dataset.parquet"))
tm = tr["timestamp_ist"].max()
priv = te[te["scoring_block"] == "private"]
pub = te[te["scoring_block"] == "public"]
unsc = te[te["scoring_block"] == "unscored"]
priv_leads = sorted(int((t - tm).total_seconds() // 3600) for t in priv["target_end_ist"].unique())
pub_leads = sorted(int((t - tm).total_seconds() // 3600) for t in pub["target_end_ist"].unique())
out = dict(
    train_source="data/processed/UPDATE-10/training_dataset.parquet (from ACTUAL UPDATE-10/Train.csv)",
    train_max=str(tm), train_rows=int(len(tr)),
    private_rows=int(len(priv)), private_hours=int(priv["target_end_ist"].nunique()),
    private_leads=priv_leads, private_window=[str(priv["target_end_ist"].min()), str(priv["target_end_ist"].max())],
    public_rows=int(len(pub)), public_hours=int(pub["target_end_ist"].nunique()), public_leads=pub_leads,
    unscored_rows=int(len(unsc)), unscored_hours=int(unsc["target_end_ist"].nunique()), unscored_leads=sorted(int((t - tm).total_seconds() // 3600) for t in unsc["target_end_ist"].unique()),
)
with open(os.path.join(ROOT, "reports", "update10", "FORECAST_HORIZON.json"), "w") as f: json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("HORIZON_RECALC_DONE")
