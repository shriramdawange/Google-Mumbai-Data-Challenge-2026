"""Finalize: train final model (RF best) + submission. Reuses OOF validation."""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mission01 import load, build_series, cal_feats, train_matrix, FEATURES
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports"); SUBS = os.path.join(ROOT, "submissions")
os.makedirs(REPORTS, exist_ok=True); os.makedirs(SUBS, exist_ok=True)

def sample_cutoffs(train_max, horizon_max, step_h=6):
    # normalize to same tz as train_max
    s = pd.Timestamp("2025-03-01", tz=train_max.tz)
    return pd.date_range(s, train_max - pd.Timedelta(hours=horizon_max), freq=f"{step_h}h")

def main():
    tr, te, st = load()
    idx, series, city_df, _ = build_series(tr)
    train_max = tr.timestamp_ist.max()
    HMAX=240
    final_cutoffs = sample_cutoffs(train_max, horizon_max=HMAX, step_h=6)
    print(f"final cutoffs {len(final_cutoffs)} train_max={train_max}", flush=True)
    Xf, yf, mf = train_matrix(series, city_df, final_cutoffs)
    print(f"FINAL train {Xf.shape}", flush=True)
    med_f = Xf.median(numeric_only=True); Xf_f = Xf[FEATURES].fillna(med_f)
    rf_f = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
    hgb_f = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.1, random_state=42)
    rf_f.fit(Xf_f, yf); hgb_f.fit(Xf_f, yf)
    with open(os.path.join(REPORTS,"final_models.pkl"),"wb") as f:
        pickle.dump(dict(rf=rf_f, hgb=hgb_f, med=med_f), f)
    # best = RF per OOF (rmse 10.57 vs 11.68)
    te2 = te.copy(); te2["target_end_ist"] = pd.to_datetime(te2["target_end_ist"])
    preds_rf=[]; preds_hgb=[]; leads=[]
    for _, row in te2.iterrows():
        loc=int(row["location_id"]); t=row["target_end_ist"]; c=train_max
        lead=float((t-c).total_seconds()/3600.0); leads.append(lead)
        df=series[loc]; crow=df.loc[c] if c in df.index else df.iloc[-1]
        r=dict(station_code=int(crow["station_code"]), lag1=crow["lag1"], lag2=crow["lag2"], lag3=crow["lag3"],
               lag6=crow["lag6"], lag12=crow["lag12"], lag24=crow["lag24"], lag48=crow["lag48"],
               roll_mean_6=crow["roll_mean_6"], roll_mean_24=crow["roll_mean_24"], roll_mean_72=crow["roll_mean_72"],
               roll_mean_168=crow["roll_mean_168"], roll_std_24=crow["roll_std_24"], roll_std_72=crow["roll_std_72"],
               trend_24=crow["trend_24"],
               city_lag1=float(city_df.loc[c,"city_lag1"]) if c in city_df.index else np.nan,
               city_roll24=float(city_df.loc[c,"city_roll24"]) if c in city_df.index else np.nan,
               city_roll72=float(city_df.loc[c,"city_roll72"]) if c in city_df.index else np.nan)
        r.update(cal_feats(pd.Timestamp(t))); r["lead"]=lead; r["log_lead"]=float(np.log1p(max(lead,0)))
        x=pd.DataFrame([r])[FEATURES].fillna(med_f)
        preds_rf.append(float(max(0, rf_f.predict(x)[0])))
        preds_hgb.append(float(max(0, hgb_f.predict(x)[0])))
    sub=pd.DataFrame(dict(ID=te2.ID.values, Target=preds_rf))
    assert len(sub)==2592 and sub.ID.is_unique
    assert sub.Target.notna().all() and np.isfinite(sub.Target).all() and (sub.Target>=0).all()
    assert (sub.ID.values==te2.ID.values).all(), "order must match Test"
    sub.to_csv(os.path.join(SUBS,"baseline_v0.csv"), index=False)
    print("saved baseline_v0 (RF)", sub.Target.describe().to_string())
    # also save hgb variant for reference
    pd.DataFrame(dict(ID=te2.ID.values, Target=preds_hgb)).to_csv(os.path.join(SUBS,"baseline_v0_hgb.csv"), index=False)
    # submission diagnostics
    te2["Target_rf"]=preds_rf; te2["Target_hgb"]=preds_hgb; te2["lead"]=leads
    te2.to_csv(os.path.join(REPORTS,"submission_diagnostics.csv"), index=False)
    print(te2.groupby("scoring_block").agg(n=("Target_rf","size"), mean_rf=("Target_rf","mean"), max_rf=("Target_rf","max"),
          mean_hgb=("Target_hgb","mean")).to_string())
    print(te2.groupby("location_id").Target_rf.agg(["mean","min","max"]).to_string())
    # extreme analysis from OOF: top-decile + correlation; no obs >100 in val — document
    oof=pd.read_csv(os.path.join(REPORTS,"oof_validation.csv"))
    from sklearn.metrics import mean_squared_error
    for thr in [50, 62, 75]:
        sub_e=oof[oof.y>thr]
        if len(sub_e)>0:
            print(f"y>{thr}: n={len(sub_e)} rf_rmse={np.sqrt(mean_squared_error(sub_e.y, sub_e.pred_rf)):.3f} hgb={np.sqrt(mean_squared_error(sub_e.y, sub_e.pred_hgb)):.3f}")
    print("OOF y>100 n=", (oof.y>100).sum(), " y pct", oof.y.describe().to_dict())

if __name__=="__main__":
    main()
