"""MISSION 03.2 — FINAL MODEL CHALLENGER: compare B2 vs B3 on last seven untouched validation origins."""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "reports", "update10", "validation")
o = pd.read_csv(os.path.join(V, "validation_oof.csv"))
UNTOUCHED_FOLDS = [5, 6, 7, 8, 9, 10, 11]
o_u = o[o["fold"].isin(UNTOUCHED_FOLDS)].copy()

def rmse(a, b): return float(np.sqrt(mean_squared_error(np.asarray(a, dtype=float), np.asarray(b, dtype=float))))
def mae(a, b): return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
def bias(a, b): return float(np.mean(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))

def report(df):
    rows = {}
    for m in ["B2", "B3", "B4", "pred_rf"]:
        rows[m] = dict(rmse=rmse(df["y"], df[m]), mae=mae(df["y"], df[m]), bias=bias(df["y"], df[m]), n=int(len(df)))
    return rows

res = {}
res["last_seven"] = report(o_u)
per_station = []
for loc, g in o_u.groupby("location_id"):
    per_station.append(dict(location_id=int(loc), n=int(len(g)),
                            B2=float(rmse(g["y"], g["B2"])), B3=float(rmse(g["y"], g["B3"])),
                            B4=float(rmse(g["y"], g["B4"])), pred_rf=float(rmse(g["y"], g["pred_rf"])),
                            mae_B2=float(mae(g["y"], g["B2"])), mae_B3=float(mae(g["y"], g["B3"]))))
res["per_station_last_seven"] = per_station

per_origin = []
for f, g in o_u.groupby("fold"):
    per_origin.append(dict(fold=int(f), origin=str(g["origin"].iloc[0]), n=int(len(g)),
                           B2=float(rmse(g["y"], g["B2"])), B3=float(rmse(g["y"], g["B3"])), B4=float(rmse(g["y"], g["B4"])), pred_rf=float(rmse(g["y"], g["pred_rf"]))))
res["per_origin_last_seven"] = per_origin

latest = o[o["origin"] == "2026-11-05 03:00:00+05:30"].copy()
res["latest_origin_2026_11_05"] = report(latest)
res["latest_per_origin"] = [dict(fold=int(f), origin=str(g["origin"].iloc[0]), n=int(len(g)),
                                 B2=float(rmse(g["y"], g["B2"])), B3=float(rmse(g["y"], g["B3"])), B4=float(rmse(g["y"], g["B4"])), pred_rf=float(rmse(g["y"], g["pred_rf"]))) for f, g in latest.groupby("fold")]

o_u["se_B2"] = (o_u["B2"] - o_u["y"]) ** 2
o_u["se_B3"] = (o_u["B3"] - o_u["y"]) ** 2
o_u["se_B4"] = (o_u["B4"] - o_u["y"]) ** 2
o_u["se_RF"] = (o_u["pred_rf"] - o_u["y"]) ** 2
rng = np.random.RandomState(0)
origins_u = sorted(o_u["origin"].unique())
pairs = [("B2", "B3", "se_B2", "se_B3", "B2_vs_B3"), ("B2", "B4", "se_B2", "se_B4", "B2_vs_B4"), ("B2", "pred_rf", "se_B2", "se_RF", "B2_vs_RF"), ("B3", "pred_rf", "se_B3", "se_RF", "B3_vs_RF")]
sets = {p[4]: [] for p in pairs}
rm = {c: [] for c in ["B2", "B3", "B4", "RF"]}
B = 10000
for _ in range(B):
    samp = list(rng.choice(origins_u, size=len(origins_u), replace=True))
    rr = pd.concat([o_u[o_u["origin"] == x] for x in samp], ignore_index=True)
    for c in rm:
        rm[c].append(float(np.sqrt(rr[f"se_{c}"].mean())))
    for (a, b, sa, sb, key) in pairs:
        sets[key].append(float(np.sqrt(rr[sb].mean()) - np.sqrt(rr[sa].mean())))
boot = dict(n_boot=B, n_origins=int(len(origins_u)), n_rows=int(len(o_u)), comparisons={})
comparisons = {}
for p, k in [(p, p[4]) for p in pairs]:
    comparisons[k] = dict(
        model_A=p[0], model_B=p[1], convention="sqrtMSE(B)-sqrtMSE(A); positive => A better",
        observed_diff=float(rmse(o_u["y"], o_u[p[1]]) - rmse(o_u["y"], o_u[p[0]])),
        ci=[float(np.percentile(sets[k], 2.5)), float(np.percentile(sets[k], 97.5))],
        p_A_better=float((np.array(sets[k]) > 0).mean()),
    )
boot["comparisons"] = comparisons
boot["model_rmse_ci"] = {c: [float(np.percentile(rm[c], 2.5)), float(np.percentile(rm[c], 97.5))] for c in rm}
boot["model_rmse"] = {c: rmse(o_u["y"], o_u["B2" if c == "B2" else ("B3" if c == "B3" else ("B4" if c == "B4" else "pred_rf"))]) for c in rm}
with open(os.path.join(V, "challenger_compare.json"), "w") as f: json.dump(res, f, indent=2, default=str)
with open(os.path.join(V, "challenger_bootstrap.json"), "w") as f: json.dump(boot, f, indent=2)
print(json.dumps(res, indent=2, default=str))
print(json.dumps(boot, indent=2))
print("CHALLENGER_COMPARE_DONE")
