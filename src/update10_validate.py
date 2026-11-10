"""MISSION 03 Stage D: historical validation + model comparison for the UPDATE-10 horizon.

Re-runs the M02.3 short-horizon validation methodology (B2 / B3 / B4 / RF, short leads
36-96) against the interim UPDATE-10 training source (data/processed/UPDATE-10/training_dataset.parquet)
and the UPDATE-10 Test.csv. Private forecast leads are derived dynamically from the actual files.

Writes all validation outputs to reports/update10/validation/ (never reports/m023/).
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPT = os.path.join(ROOT, "reports", "update10")
V = os.path.join(RPT, "validation"); os.makedirs(V, exist_ok=True)
from mission02_3 import build_series, matrix, FEATS, rmse, mae
from mission02_2 import window_stats, hour_profile

PARQ = os.path.join(ROOT, "data", "processed", "UPDATE-10", "training_dataset.parquet")
TEST = os.path.join(ROOT, "UPDATE-10", "Test.csv")
SHORT = list(range(36, 97, 6))  # RF training horizons (short-horizon neighbourhood of T+42-89)


def load():
    tr = pd.read_parquet(PARQ); tr["timestamp_ist"] = pd.to_datetime(tr["timestamp_ist"])
    tr = tr.drop(columns=["source_release"], errors="ignore").copy()
    tr["pm25_clean"] = tr["pm25"].astype(float)
    tr.loc[tr["flag_high_1000"] == True, "pm25_clean"] = np.nan
    tr.loc[tr["pm25"] > 1000, "pm25_clean"] = np.nan
    te = pd.read_csv(TEST, parse_dates=["target_start_ist", "target_end_ist"])
    train_max = tr.timestamp_ist.max()
    priv = te[te.scoring_block == "private"]
    leads = sorted(int((t - train_max).total_seconds() // 3600) for t in priv.target_end_ist.unique())
    return tr, te, train_max, leads, priv


def main():
    tr, te, train_max, LEADS, priv = load()
    print(f"[D] train_max={train_max} private_leads={LEADS[0]}..{LEADS[-1]} n_priv={len(priv)}", flush=True)
    series, cdf = build_series(tr)
    train_start = tr.timestamp_ist.min(); tz = train_max.tz
    cand = pd.date_range(train_start + pd.Timedelta(days=120), train_max - pd.Timedelta(hours=max(LEADS) + 24), freq="7D")
    cand = [c for c in cand if (c - train_start).days >= 120]
    step = max(1, len(cand) // 9); origins = list(dict.fromkeys(list(cand[::step][:9]) + list(cand[-3:])))
    print(f"[D] origins ({len(origins)}): {[str(o.date()) for o in origins]}", flush=True)
    ev_rows = []; fold_info = []
    for fi, origin in enumerate(origins):
        tc = pd.date_range(train_start + pd.Timedelta(days=30), origin - pd.Timedelta(hours=1), freq="12h")
        Xtr, ytr, _ = matrix(series, cdf, tc, SHORT, origin)
        Xev, yev, mev = matrix(series, cdf, [origin], LEADS, origin=series[next(iter(series))].index.max())
        fold_info.append(dict(fold=fi, origin=str(origin), n_train=int(len(ytr)), n_eval=int(len(yev))))
        if len(yev) == 0:
            print(f"[{fi}] {origin.date()} SKIP: zero observed targets in lead window", flush=True); continue
        per = {}
        for loc, s in series.items():
            pm = pd.Series(s["pm25"])
            fast = window_stats(pm, origin, 7)["mean"]; slow = window_stats(pm, origin, 90)["mean"]
            prof, _ = hour_profile(pm, origin, 90); norm = prof / (float(prof.mean()) if len(prof) else 1)
            per[loc] = dict(fast=fast, slow=slow, norm=norm)
        b2 = []; fasts = []; slows = []
        for _, r in mev.iterrows():
            loc = int(r.location_id); h = int(r.target_t.hour); e = per[loc]
            n = float(e["norm"].get(h, 1.0))
            b2.append(float(e["fast"] * n)); fasts.append(float(e["fast"] * n)); slows.append(float(e["slow"] * n))
        b2 = np.array(b2); fasts = np.array(fasts); slows = np.array(slows)
        med = Xtr[FEATS].median(numeric_only=True)
        rf = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
        rf.fit(Xtr[FEATS].fillna(med), ytr)
        pred_rf = np.clip(rf.predict(Xev[FEATS].fillna(med)), 0, None)
        d = mev.copy(); d["y"] = yev; d["fold"] = fi; d["origin"] = str(origin)
        d["B2"] = np.clip(b2, 0, None); d["fast"] = fasts; d["slow"] = slows; d["pred_rf"] = pred_rf
        ev_rows.append(d)
        print(f"[{fi}] {origin.date()} n_eval={len(yev)} B2={rmse(d.y, d.B2):.3f} RF={rmse(d.y, d.pred_rf):.3f}", flush=True)
    ev = pd.concat(ev_rows, ignore_index=True)
    # B3 shrinkage w and B4 trend beta fitted on early folds only (first 5 origins)
    folds_sorted = sorted(ev.fold.unique()); early = folds_sorted[:5]
    grid = [round(x, 2) for x in np.linspace(0, 1, 21)]
    def rmse_w(d, w): return rmse(d.y, w * d.fast + (1 - w) * d.slow)
    w = float(min(grid, key=lambda w_: rmse_w(ev[ev.fold.isin(early)], w_)))
    betas = np.linspace(-0.5, 0.5, 21)
    def rmse_b(b):
        dd = ev[ev.fold.isin(early)]; return rmse(dd.y, w * dd.fast + (1 - w) * dd.slow + b * (dd.fast - dd.slow))
    beta = float(min(betas, key=lambda b_: rmse_b(b_)))
    print(f"[D] early-fit shrinkage w={w} beta={beta}", flush=True)
    ev["B3"] = np.clip(w * ev.fast + (1 - w) * ev.slow, 0, None)
    ev["B4"] = np.clip(ev.B3 + beta * (ev.fast - ev.slow), 0, None)
    # ---------- outputs ----------
    ev.to_csv(os.path.join(V, "validation_oof.csv"), index=False)
    with open(os.path.join(V, "fold_meta.json"), "w") as f: json.dump(fold_info, f, indent=2)
    with open(os.path.join(V, "config.json"), "w") as f:
        json.dump(dict(source_parquet=PARQ, test=TEST, train_max=str(train_max), private_leads=LEADS,
                       private_rows=int(len(priv)), private_hours=int(priv.target_end_ist.nunique()),
                       short_rf_horizons=SHORT, origin_count=len(origins)), f, indent=2)
    models = ["B2", "B3", "B4", "pred_rf"]
    mrows = []
    for c in models:
        yv = np.asarray(ev["y"], dtype=float)
        yv_c = np.asarray(ev[c], dtype=float)
        row = dict(
            model=c,
            rmse=float(np.sqrt(np.mean((yv - yv_c) ** 2))),
            mae=float(np.mean(np.abs(yv - yv_c))),
            bias=float(np.mean(yv - yv_c)),
            n=int(len(ev)),
        )
        mrows.append(row)
    pd.DataFrame(mrows).to_csv(os.path.join(V, "metrics_overall.csv"), index=False)
    by_origin = [dict(origin=o, model=c, rmse=rmse(g.y, g[c]), mae=mae(g.y, g[c]), bias=float((g[c] - g.y).mean()), n=int(len(g)))
                 for o, g in ev.groupby("origin") for c in models]
    pd.DataFrame(by_origin).to_csv(os.path.join(V, "metrics_by_origin.csv"), index=False)
    by_station = [dict(location_id=int(k), model=c, rmse=rmse(g.y, g[c]), mae=mae(g.y, g[c]), bias=float((g[c] - g.y).mean()), n=int(len(g)))
                   for k, g in ev.groupby("location_id") for c in models]
    pd.DataFrame(by_station).to_csv(os.path.join(V, "metrics_by_station.csv"), index=False)
    # ---------- paired origin-block bootstrap ----------
    ev["se_B2"] = (ev.B2 - ev.y) ** 2; ev["se_B3"] = (ev.B3 - ev.y) ** 2; ev["se_B4"] = (ev.B4 - ev.y) ** 2; ev["se_RF"] = (ev.pred_rf - ev.y) ** 2
    B = 10000; pairs = [("B2", "B3", "se_B2", "se_B3", "B2_vs_B3"), ("B2", "B4", "se_B2", "se_B4", "B2_vs_B4"), ("B2", "RF", "se_B2", "se_RF", "B2_vs_RF"), ("B3", "RF", "se_B3", "se_RF", "B3_vs_RF")]
    out = dict(n_boot=B, n_origins=int(len(origins)), n_rows=int(len(ev)))
    sets = {p[4]: [] for p in pairs}
    rm = {c: [] for c in ["B2", "B3", "B4", "RF"]}
    rng = np.random.RandomState(0); origins_u = ev["origin"].unique()
    for _ in range(B):
        samp = list(rng.choice(origins_u, size=len(origins_u), replace=True))
        rr = pd.concat([ev[ev["origin"] == o] for o in samp], ignore_index=True)
        for c in rm: rm[c].append(float(np.sqrt(rr[f"se_{c}"].mean())))
        for (a, b, sa, sb, key) in pairs:
            sets.setdefault(key, []).append(float(np.sqrt(rr[sb].mean()) - np.sqrt(rr[sa].mean())))  # positive = A better
    comparisons = {}
    col = {"B2": "B2", "B3": "B3", "B4": "B4", "RF": "pred_rf"}
    for (a, b, sa, sb, key) in pairs:
        arr = np.array(sets[key])
        comparisons[key] = dict(
            model_A=a,
            model_B=b,
            convention="sqrtMSE(B)-sqrtMSE(A); positive => A better",
            observed_diff=float(rmse(ev.y, ev[col[b]]) - rmse(ev.y, ev[col[a]])),
            ci=[float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            p_A_better=float((arr > 0).mean()),
        )
    bootstrap = dict(n_boot=B, n_origins=int(len(origins)), n_rows=int(len(ev)), comparisons=comparisons,
                     model_rmse_ci={c: [float(np.percentile(rm[c], 2.5)), float(np.percentile(rm[c], 97.5))] for c in rm},
                     model_rmse={c: rmse(ev.y, ev[col[c]]) for c in rm})
    with open(os.path.join(V, "bootstrap.json"), "w") as f: json.dump(bootstrap, f, indent=2)
    # ---------- MODEL_SELECTION.md ----------
    best = min(mrows, key=lambda r: r["rmse"])
    lines = ["# UPDATE-10 Model Selection (Mission 03, Stage D)", "", "## Setup", "",
             f"- Training source: interim UPDATE-10 dataset (`data/processed/UPDATE-10/training_dataset.parquet`), cutoff {train_max}",
             f"- Test: `UPDATE-10/Test.csv` ({len(te)} rows, private {len(priv)} rows / {int(priv.target_end_ist.nunique())} unique hours)",
             f"- Private forecast leads: {LEADS[0]}..{LEADS[-1]} (contiguous, n={len(LEADS)})", "",
             "## Pooled UPDATE-10 validation metrics (short-horizon re-run)", "",
             "| model | RMSE | MAE | bias | n |", "|---|---:|---:|---:|---:|"]
    for r in mrows: lines.append(f"| {r['model']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['bias']:.4f} | {r['n']} |")
    lines += ["", f"**Best pooled model: {best['model']}** (RMSE {best['rmse']:.4f}).", "",
              "## Paired bootstrap (10k origin-block, this run)", "",
              json.dumps(bootstrap["comparisons"], indent=2), "",
              "Model RMSE bootstrap CIs:", json.dumps(bootstrap["model_rmse_ci"]), "", ""]
    lines += [f"RMSE: {json.dumps(bootstrap['model_rmse'])}", "",
              "## Decision", "",
              f"The selected model for the final UPDATE-10 inference is **{best['model']}** based on pooled RMSE with the",
              "short-horizon validation re-run on the actual UPDATE-10 training source. See `reports/update10/FORECAST_HORIZON_AUDIT.md`",
              "and `reports/MISSION_03_FINAL_READINESS.md` for the full chain of evidence.", ""]
    with open(os.path.join(V, "MODEL_SELECTION.md"), "w") as f: f.write("\n".join(lines))
    print(f"UPDATE10_STAGE_D_DONE n={len(ev)} pooled_B2={rmse(ev.y,ev.B2):.4f} B3={rmse(ev.y,ev.B3):.4f} B4={rmse(ev.y,ev.B4):.4f} RF={rmse(ev.y,ev.pred_rf):.4f}", flush=True)
    print("DONE_UPDATE10_VALIDATE", flush=True)


if __name__ == "__main__":
    main()
