"""MISSION 01: causal direct-horizon baseline pipeline.
Cutoff-based features: all history features use data <= cutoff only.
No future target info. Excludes flag_high_1000 (>1000) from training
targets/features as sensor anomalies; reported separately.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
SUBS = os.path.join(ROOT, "submissions")
os.makedirs(REPORTS, exist_ok=True); os.makedirs(SUBS, exist_ok=True)
import pickle

def load():
    tr = pd.read_csv(os.path.join(DATA,"Train.csv"), parse_dates=["timestamp_ist","available_at"])
    te = pd.read_csv(os.path.join(DATA,"Test.csv"), parse_dates=["target_start_ist","target_end_ist"])
    st = pd.read_csv(os.path.join(DATA,"Stations.csv"))
    tr["timestamp_ist"] = pd.to_datetime(tr["timestamp_ist"])
    # Clean extreme sensor anomalies: flag_high_1000 or pm25>1000 -> NaN for features/targets
    tr["pm25_clean"] = tr["pm25"].astype(float)
    tr.loc[tr["flag_high_1000"]==True, "pm25_clean"] = np.nan
    tr.loc[tr["pm25"]>1000, "pm25_clean"] = np.nan
    tr = tr.sort_values(["location_id","timestamp_ist"])
    return tr, te, st

def build_series(tr):
    """Hourly grid per station with lags/rolling precomputed (causal)."""
    stations = sorted(tr.location_id.unique().tolist())
    station_code = {s:i for i,s in enumerate(stations)}
    start = tr.timestamp_ist.min().floor("h"); end = tr.timestamp_ist.max().floor("h")
    idx = pd.date_range(start, end, freq="h")
    series = {}
    for loc in stations:
        g = tr[tr.location_id==loc].set_index("timestamp_ist")["pm25_clean"].sort_index()
        g = g[~g.index.duplicated(keep="last")].reindex(idx)
        df = pd.DataFrame({"pm25": g}, index=idx)
        for L in [1,2,3,6,12,24,48]:
            df[f"lag{L}"] = df["pm25"].shift(L)
        for W in [6,24,72,168]:
            df[f"roll_mean_{W}"] = df["pm25"].rolling(W, min_periods=1).mean()
        for W in [24,72]:
            df[f"roll_std_{W}"] = df["pm25"].rolling(W, min_periods=1).std()
        df["roll_mean_24_prev"] = df["roll_mean_24"].shift(24)
        df["trend_24"] = df["roll_mean_24"] - df["roll_mean_24_prev"]
        df["station_code"] = station_code[loc]
        df["location_id"] = loc
        series[loc] = df
    # city mean (mean across stations at same hour) + its lags/roll
    city = pd.DataFrame({loc: series[loc]["pm25"] for loc in stations}, index=idx).mean(axis=1)
    city_df = pd.DataFrame({"city_pm25": city}, index=idx)
    city_df["city_lag1"] = city_df["city_pm25"].shift(1)
    city_df["city_roll24"] = city_df["city_pm25"].rolling(24, min_periods=1).mean()
    city_df["city_roll72"] = city_df["city_pm25"].rolling(72, min_periods=1).mean()
    return idx, series, city_df, station_code

FEATURES = ["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
            "lead","log_lead",
            "lag1","lag2","lag3","lag6","lag12","lag24","lag48",
            "roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168",
            "roll_std_24","roll_std_72","trend_24","city_lag1","city_roll24","city_roll72"]

def cal_feats(t):
    h=t.hour; d=t.dayofweek; m=t.month
    return dict(hour_t=h, dow_t=d, month_t=m,
                hour_sin=np.sin(2*np.pi*h/24), hour_cos=np.cos(2*np.pi*h/24),
                dow_sin=np.sin(2*np.pi*d/7), dow_cos=np.cos(2*np.pi*d/7),
                is_weekend=int(d>=5))

def sample_cutoffs(train_max, horizon_max, start="2025-03-01", step_h=6):
    s = pd.Timestamp(start, tz="Asia/Kolkata")
    return pd.date_range(s, train_max - pd.Timedelta(hours=horizon_max), freq=f"{step_h}h")

HORIZONS = [1,3,6,12,24,36,48,72,96,168,240]

def train_matrix(series, city_df, cutoffs, horizons=HORIZONS):
    rows=[]; ys=[]; meta=[]
    for loc, df in series.items():
        scode = int(df["station_code"].iloc[0])
        for c in cutoffs:
            if c not in df.index: continue
            crow = df.loc[c]; city_row = city_df.loc[c] if c in city_df.index else None
            base = dict(station_code=scode,
                        lag1=crow["lag1"], lag2=crow["lag2"], lag3=crow["lag3"],
                        lag6=crow["lag6"], lag12=crow["lag12"], lag24=crow["lag24"], lag48=crow["lag48"],
                        roll_mean_6=crow["roll_mean_6"], roll_mean_24=crow["roll_mean_24"],
                        roll_mean_72=crow["roll_mean_72"], roll_mean_168=crow["roll_mean_168"],
                        roll_std_24=crow["roll_std_24"], roll_std_72=crow["roll_std_72"],
                        trend_24=crow["trend_24"],
                        city_lag1=float(city_row["city_lag1"]) if city_row is not None else np.nan,
                        city_roll24=float(city_row["city_roll24"]) if city_row is not None else np.nan,
                        city_roll72=float(city_row["city_roll72"]) if city_row is not None else np.nan)
            for h in horizons:
                t = c + pd.Timedelta(hours=h)
                if t not in df.index: continue
                y = df.loc[t, "pm25"]
                if pd.isna(y): continue
                r = dict(base); r.update(cal_feats(t)); r["lead"]=h; r["log_lead"]=np.log1p(h)
                rows.append(r); ys.append(float(y)); meta.append((int(loc), c, t, h))
    X = pd.DataFrame(rows); y = np.array(ys); meta = pd.DataFrame(meta, columns=["location_id","cutoff","target_t","lead"])
    return X, y, meta

def eval_matrix(series, city_df, cutoff, horizons):
    return train_matrix(series, city_df, [cutoff], horizons)

def rmse(a,b):
    return float(np.sqrt(mean_squared_error(a,b)))

def main():
    tr, te, st = load()
    print("train", tr.shape, "test", te.shape, flush=True)
    idx, series, city_df, scode_map = build_series(tr)
    train_max = tr.timestamp_ist.max()
    print("train_max", train_max, flush=True)
    HMAX=240
    # ---- Rolling-origin validation: 4 folds ----
    folds = [pd.Timestamp(d, tz="Asia/Kolkata") for d in ["2026-08-20","2026-09-20","2026-10-05","2026-10-20"]]
    eval_h = list(range(1,73))
    results=[]
    oof_store=[]  # (fold, location_id, cutoff, target, lead, y, pred_model, pred_persist, pred_rollmean)
    for fi, cut in enumerate(folds):
        train_cutoffs = sample_cutoffs(cut - pd.Timedelta(hours=HMAX), horizon_max=HMAX, step_h=6)
        print(f"fold {fi} cutoff {cut} n_train_cutoffs={len(train_cutoffs)}", flush=True)
        Xtr, ytr, mtr = train_matrix(series, city_df, train_cutoffs)
        Xev, yev, mev = eval_matrix(series, city_df, cut, eval_h)
        # drop eval rows with all-NaN history still kept (models handle); keep as is
        print(f"  train {Xtr.shape} eval {Xev.shape}", flush=True)
        # median impute for RF (fit on train)
        med = Xtr.median(numeric_only=True)
        Xtr_f = Xtr.fillna(med); Xev_f = Xev.fillna(med)
        # models
        rf = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
        hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.1, random_state=42)
        rf.fit(Xtr_f, ytr); hgb.fit(Xtr_f, ytr)
        pred_rf = np.clip(rf.predict(Xev_f), 0, None)
        # HGB can take NaN; use filled for consistency
        pred_hgb = np.clip(hgb.predict(Xev_f), 0, None)
        # baselines: persistence = lag1 at cutoff; rollmean24
        pred_p = np.clip(np.where(np.isnan(Xev["lag1"].values), (Xtr["lag1"].median(),), Xev["lag1"].values), 0, None)
        pred_r = np.clip(Xev["roll_mean_24"].fillna(np.nanmedian(ytr)).values, 0, None)
        for name, pred in [("rf",pred_rf),("hgb",pred_hgb),("persistence",pred_p),("rollmean24",pred_r)]:
            r = rmse(yev, pred)
            results.append(dict(fold=fi, cutoff=str(cut), model=name, rmse=r, n=len(yev)))
            print(f"    {name} rmse={r:.3f}", flush=True)
        mev2 = mev.copy(); mev2["y"]=yev; mev2["pred_rf"]=pred_rf; mev2["pred_hgb"]=pred_hgb; mev2["pred_persist"]=pred_p; mev2["pred_rollmean"]=pred_r; mev2["fold"]=fi
        oof_store.append(mev2)
        # also report per-station for HGB
        for loc in sorted(mev2.location_id.unique()):
            sub = mev2[mev2.location_id==loc]
            print(f"      HGB loc {loc} rmse={rmse(sub.y, sub.pred_hgb):.3f} n={len(sub)}", flush=True)
    oof = pd.concat(oof_store, ignore_index=True)
    oof.to_csv(os.path.join(REPORTS,"oof_validation.csv"), index=False)
    with open(os.path.join(REPORTS,"validation_results.json"),"w") as f:
        json.dump(results,f,indent=2)
    # overall + per-station + lead + extreme for HGB and best
    overall={}
    for col in ["pred_rf","pred_hgb","pred_persist","pred_rollmean"]:
        overall[col]= dict(rmse=rmse(oof.y, oof[col]), mae=float(np.mean(np.abs(oof.y-oof[col]))), bias=float(np.mean(oof[col]-oof.y)))
    per_station = oof.groupby("location_id").apply(lambda d: {c: rmse(d.y, d[c]) for c in ["pred_hgb","pred_rf","pred_persist","pred_rollmean"]}, include_groups=False)
    def lead_bucket(x):
        if x<=6: return "1-6"
        if x<=24: return "7-24"
        if x<=48: return "25-48"
        return "49-72"
    oof["lead_bucket"]=oof.lead.apply(lead_bucket)
    lead_rmse = oof.groupby("lead_bucket").apply(lambda d: {c: rmse(d.y,d[c]) for c in ["pred_hgb","pred_rf","pred_persist","pred_rollmean"]}, include_groups=False)
    ext = oof[oof.y>100]
    ext_rmse = ({"n": len(ext), **{c: rmse(ext.y, ext[c]) if len(ext)>0 else None for c in ["pred_hgb","pred_rf","pred_persist","pred_rollmean"]}} if len(ext)>0 else {"n":0})
    high = oof[oof.y>200]
    with open(os.path.join(REPORTS,"overall_metrics.json"),"w") as f:
        json.dump({"overall":overall,"per_station":per_station.to_dict(),"lead_rmse":lead_rmse.to_dict(),"extreme_gt100":ext_rmse,"n_high_gt200":len(high)},f,indent=2)
    print("OVERALL", overall, flush=True)
    print("PER_STATION\n", per_station.to_string(), flush=True)
    print("LEAD\n", lead_rmse.to_string(), flush=True)
    print("EXTREME", ext_rmse, flush=True)
    # pick best model by overall rmse between rf/hgb
    best_col = "pred_hgb" if overall["pred_hgb"]["rmse"]<=overall["pred_rf"]["rmse"] else "pred_rf"
    best_model_name = "HistGradientBoosting" if best_col=="pred_hgb" else "RandomForest"

    # ---- Final model: train on all history up to train_max - HMAX ----
    final_cutoffs = sample_cutoffs(train_max, horizon_max=HMAX, step_h=6)
    Xf, yf, mf = train_matrix(series, city_df, final_cutoffs)
    print(f"FINAL train {Xf.shape}", flush=True)
    med_f = Xf.median(numeric_only=True); Xf_f = Xf.fillna(med_f)
    rf_f = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
    hgb_f = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0.1, random_state=42)
    rf_f.fit(Xf_f, yf); hgb_f.fit(Xf_f, yf)
    with open(os.path.join(REPORTS,"final_models.pkl"),"wb") as f:
        pickle.dump(dict(rf=rf_f, hgb=hgb_f, med=med_f), f)
    # ---- Test inference: cutoff = train_max (per station same) ----
    te2 = te.copy(); te2["target_end_ist"] = pd.to_datetime(te2["target_end_ist"])
    preds=[]
    for _, row in te2.iterrows():
        loc = int(row["location_id"]); t = row["target_end_ist"]; c=train_max
        lead = (t - c).total_seconds()/3600.0
        df = series[loc]; crow = df.loc[c] if c in df.index else df.iloc[-1]
        r = dict(station_code=int(crow["station_code"]),
                 lag1=crow["lag1"], lag2=crow["lag2"], lag3=crow["lag3"], lag6=crow["lag6"],
                 lag12=crow["lag12"], lag24=crow["lag24"], lag48=crow["lag48"],
                 roll_mean_6=crow["roll_mean_6"], roll_mean_24=crow["roll_mean_24"],
                 roll_mean_72=crow["roll_mean_72"], roll_mean_168=crow["roll_mean_168"],
                 roll_std_24=crow["roll_std_24"], roll_std_72=crow["roll_std_72"], trend_24=crow["trend_24"],
                 city_lag1=float(city_df.loc[c,"city_lag1"]) if c in city_df.index else np.nan,
                 city_roll24=float(city_df.loc[c,"city_roll24"]) if c in city_df.index else np.nan,
                 city_roll72=float(city_df.loc[c,"city_roll72"]) if c in city_df.index else np.nan)
        r.update(cal_feats(pd.Timestamp(t))); r["lead"]=float(lead); r["log_lead"]=float(np.log1p(max(lead,0)))
        x = pd.DataFrame([r])[FEATURES].fillna(med_f)
        p = hgb_f.predict(x)[0] if best_col=="pred_hgb" else rf_f.predict(x)[0]
        preds.append(dict(ID=row["ID"], Target=float(max(0, p)), lead=lead))
    sub = pd.DataFrame(preds)[["ID","Target"]]
    # preserve Test.csv original order (already in order since iteration order kept)
    assert len(sub)==2592 and sub.ID.is_unique
    assert sub.Target.notna().all() and np.isfinite(sub.Target).all() and (sub.Target>=0).all()
    sub.to_csv(os.path.join(SUBS,"baseline_v0.csv"), index=False)
    print("saved submission", SUBS, sub.Target.describe().to_string(), flush=True)
    # save summary for report generation
    diag = dict(best_model=best_model_name, overall=overall, per_station=per_station.to_dict(), lead_rmse=lead_rmse.to_dict(), extreme=ext_rmse, fold_results=results, train_max=str(train_max), n_oof=len(oof))
    with open(os.path.join(REPORTS,"mission01_summary.json"),"w") as f:
        json.dump(diag,f,indent=2)

if __name__=="__main__":
    main()
