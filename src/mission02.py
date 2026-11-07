"""MISSION 02: long-horizon validation, blending, lookback, seasonal, update simulation.
Does NOT overwrite Mission 01 artifacts. All experiments causal/direct-horizon.
RF config for M02: n_estimators=150, max_depth=14, min_samples_leaf=4 (speed).
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)
M02_DIR = os.path.join(REPORTS, "m02"); os.makedirs(M02_DIR, exist_ok=True)

def load():
    tr = pd.read_csv(os.path.join(DATA,"Train.csv"), parse_dates=["timestamp_ist","available_at"])
    tr["pm25_clean"] = tr["pm25"].astype(float)
    tr.loc[tr["flag_high_1000"]==True, "pm25_clean"] = np.nan
    tr.loc[tr["pm25"]>1000, "pm25_clean"] = np.nan
    return tr.sort_values(["location_id","timestamp_ist"])

LOOKBACKS = [6,12,24,48,72,120,168,336]
LEADS = [24,48,72,120,168,194,216,240,241]
HORIZONS = [1,3,6,12,24,36,48,72,96,168,240]

def build_series2(tr):
    stations = sorted(tr.location_id.unique().tolist())
    code = {s:i for i,s in enumerate(stations)}
    idx = pd.date_range(tr.timestamp_ist.min().floor("h"), tr.timestamp_ist.max().floor("h"), freq="h")
    series = {}
    for loc in stations:
        g = tr[tr.location_id==loc].set_index("timestamp_ist")["pm25_clean"].sort_index()
        g = g[~g.index.duplicated(keep="last")].reindex(idx)
        df = pd.DataFrame({"pm25": g}, index=idx)
        for L in [1,2,3,6,12,24,48,120,168,336]:
            df[f"lag{L}"] = df["pm25"].shift(L)
        for W in LOOKBACKS:
            df[f"roll_mean_{W}"] = df["pm25"].rolling(W, min_periods=1).mean()
            df[f"roll_std_{W}"] = df["pm25"].rolling(W, min_periods=1).std()
        df["roll_mean_24_prev"] = df["roll_mean_24"].shift(24)
        df["trend_24"] = df["roll_mean_24"] - df["roll_mean_24_prev"]
        df["station_code"]=code[loc]
        series[loc]=df
    city = pd.DataFrame({loc: series[loc]["pm25"] for loc in stations}, index=idx).mean(axis=1)
    cdf = pd.DataFrame({"city_pm25": city}, index=idx)
    cdf["city_lag1"]=cdf["city_pm25"].shift(1)
    for W in LOOKBACKS:
        cdf[f"city_roll{W}"]=cdf["city_pm25"].rolling(W, min_periods=1).mean()
    return idx, series, cdf

def cal_feats(t):
    h,d,m = t.hour, t.dayofweek, t.month
    return dict(hour_t=h, dow_t=d, month_t=m,
                hour_sin=np.sin(2*np.pi*h/24), hour_cos=np.cos(2*np.pi*h/24),
                dow_sin=np.sin(2*np.pi*d/7), dow_cos=np.cos(2*np.pi*d/7),
                is_weekend=int(d>=5))

FULL_FEATS = ["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
              "lead","log_lead","lag1","lag2","lag3","lag6","lag12","lag24","lag48",
              "roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168",
              "roll_std_24","roll_std_72","trend_24","city_lag1","city_roll24","city_roll72"]
def lookback_feats(W):
    lagW = f"lag{W}" if f"lag{W}" in [f"lag{x}" for x in [1,2,3,6,12,24,48,120,168,336]] else "lag48"
    # W=6->lag6, 12->lag12, 24->lag24, 48->lag48, 72->lag48 (no lag72 built; use lag48), 120->lag120, 168->lag168, 336->lag336
    m = {6:"lag6",12:"lag12",24:"lag24",48:"lag48",72:"lag48",120:"lag120",168:"lag168",336:"lag336"}
    lagW = m[W]
    return ["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
            "lead","log_lead","lag1", lagW, f"roll_mean_{W}", f"roll_std_{W}", f"city_roll{W}", "city_lag1"]

def base_row(df, c, city_df):
    crow = df.loc[c]; 
    b = dict(station_code=int(crow["station_code"]),
        lag1=crow["lag1"], lag2=crow["lag2"], lag3=crow["lag3"], lag6=crow["lag6"], lag12=crow["lag12"], lag24=crow["lag24"], lag48=crow["lag48"],
        lag120=crow["lag120"], lag168=crow["lag168"], lag336=crow["lag336"],
        city_lag1=float(city_df.loc[c,"city_lag1"]) if c in city_df.index else np.nan)
    for W in LOOKBACKS:
        b[f"roll_mean_{W}"]=crow[f"roll_mean_{W}"]; b[f"roll_std_{W}"]=crow[f"roll_std_{W}"]
        b[f"city_roll{W}"]=float(city_df.loc[c,f"city_roll{W}"]) if c in city_df.index else np.nan
    b["trend_24"]=crow["trend_24"]
    return b

def matrix(series, city_df, cutoffs, horizons):
    rows=[]; ys=[]; meta=[]
    for loc, df in series.items():
        for c in cutoffs:
            if c not in df.index: continue
            base = base_row(df, c, city_df)
            for h in horizons:
                t = c + pd.Timedelta(hours=h)
                if t not in df.index: continue
                y = df.loc[t,"pm25"]
                if pd.isna(y): continue
                r=dict(base); r.update(cal_feats(t)); r["lead"]=h; r["log_lead"]=np.log1p(h)
                rows.append(r); ys.append(float(y)); meta.append((int(loc), c, t, h))
    return pd.DataFrame(rows), np.array(ys), pd.DataFrame(meta, columns=["location_id","cutoff","target_t","lead"])

def rmse(a,b): return float(np.sqrt(mean_squared_error(np.asarray(a), np.asarray(b))))

def train_rf(X, y, feats):
    med = X[feats].median(numeric_only=True)
    rf = RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
    rf.fit(X[feats].fillna(med), y)
    return rf, med

def main():
    tr = load()
    _, series, city_df = build_series2(tr)
    tz = tr.timestamp_ist.max().tz
    # --- LONG-HORIZON folds ---
    folds = [("F0_winter_dec", pd.Timestamp("2025-12-15", tz=tz)),
             ("F1_winter_jan", pd.Timestamp("2026-01-15", tz=tz)),
             ("F2_aug", pd.Timestamp("2026-08-20", tz=tz)),
             ("F3_oct5", pd.Timestamp("2026-10-05", tz=tz)),
             ("F4_oct20", pd.Timestamp("2026-10-20", tz=tz))]
    oof_rows=[]; fold_info=[]
    for fi,(fname, cut) in enumerate(folds):
        print(f"[LH] {fname} cutoff={cut}", flush=True)
        train_cut = pd.date_range(pd.Timestamp("2025-03-01", tz=tz), cut - pd.Timedelta(hours=240), freq="6h")
        Xtr, ytr, _ = matrix(series, city_df, train_cut, HORIZONS)
        Xev, yev, mev = matrix(series, city_df, [cut], LEADS)
        print(f"  train {Xtr.shape} eval {Xev.shape}", flush=True)
        rf, med = train_rf(Xtr, ytr, FULL_FEATS)
        pred_rf = np.clip(rf.predict(Xev[FULL_FEATS].fillna(med)), 0, None)
        pred_p = np.clip(np.where(np.isnan(Xev["lag1"].values), float(Xtr["lag1"].median()), Xev["lag1"].values), 0, None)
        pred_r = np.clip(Xev["roll_mean_24"].fillna(float(np.nanmedian(ytr))).values, 0, None)
        d = mev.copy(); d["y"]=yev; d["pred_rf"]=pred_rf; d["pred_persist"]=pred_p; d["pred_rollmean"]=pred_r
        d["fold"]=fname; oof_rows.append(d)
        fold_info.append(dict(fold=fname, n_train=len(ytr), n_eval=len(yev)))
        for ld in LEADS:
            s = d[d.lead==ld]
            if len(s)==0: continue
            print(f"    lead {ld}: n={len(s)} RF={rmse(s.y,s.pred_rf):.3f} P={rmse(s.y,s.pred_persist):.3f} R={rmse(s.y,s.pred_rollmean):.3f}", flush=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(os.path.join(M02_DIR,"longhorizon_oof.csv"), index=False)
    with open(os.path.join(M02_DIR,"fold_info.json"),"w") as f: json.dump(fold_info,f,indent=2)
    print("LH overall RF", rmse(oof.y, oof.pred_rf), "P", rmse(oof.y, oof.pred_persist), "R", rmse(oof.y, oof.pred_rollmean), flush=True)
    print(oof.groupby("lead").apply(lambda d: pd.Series({"n":len(d),"rf":rmse(d.y,d.pred_rf),"p":rmse(d.y,d.pred_persist),"r":rmse(d.y,d.pred_rollmean),"ymean":d.y.mean(),"ymax":d.y.max()}), include_groups=False).to_string(), flush=True)

    # --- BLENDING: optimize w on early folds F0,F1,F2; evaluate on F3,F4 + all ---
    grid = [0,0.1,0.25,0.4,0.5,0.6,0.75,0.9,1.0]
    early = oof[oof["fold"].isin(["F0_winter_dec","F1_winter_jan","F2_aug"])]
    late = oof[oof["fold"].isin(["F3_oct5","F4_oct20"])]
    def best_w(d, base):
        vals = [(w, rmse(d.y, w*d.pred_rf+(1-w)*d[base])) for w in grid]
        return min(vals, key=lambda x: x[1])
    wP_early = best_w(early, "pred_persist"); wR_early = best_w(early, "pred_rollmean")
    print(f"blend w*RF+(1-w)*persist early-optimal w={wP_early}", flush=True)
    print(f"blend w*RF+(1-w)*rollmean early-optimal w={wR_early}", flush=True)
    blend_rows=[]
    for w in grid:
        for base in ["pred_persist","pred_rollmean"]:
            blend_rows.append(dict(w=w, base=base,
                rmse_early=rmse(early.y, w*early.pred_rf+(1-w)*early[base]),
                rmse_late=rmse(late.y, w*late.pred_rf+(1-w)*late[base]),
                rmse_all=rmse(oof.y, w*oof.pred_rf+(1-w)*oof[base])))
    pd.DataFrame(blend_rows).to_csv(os.path.join(M02_DIR,"blend_grid.csv"), index=False)
    # per-lead best w (oracle, for diagnosis) + honest per-lead using early-w
    for ld in LEADS:
        d = oof[oof.lead==ld]
        vals = [(w, rmse(d.y, w*d.pred_rf+(1-w)*d.pred_persist)) for w in grid]
        print(f"lead {ld} oracle blend(w*RF+(1-w)P) best={min(vals,key=lambda x:x[1])}", flush=True)
    # --- Seasonal / extreme ---
    print("winter F0,F1 y stats", oof[oof["fold"].isin(['F0_winter_dec','F1_winter_jan'])].y.describe().to_dict(), flush=True)
    for thr in [75,100,150]:
        s = oof[oof.y>thr]
        if len(s)>0:
            print(f"y>{thr} n={len(s)} rf={rmse(s.y,s.pred_rf):.3f} p={rmse(s.y,s.pred_persist):.3f} r={rmse(s.y,s.pred_rollmean):.3f}", flush=True)
        else:
            print(f"y>{thr} n=0", flush=True)

    # --- LOOKBACK ---
    lb_origins = ["F1_winter_jan","F2_aug","F3_oct5","F4_oct20"]
    lb_rows=[]
    for W in LOOKBACKS:
        feats = lookback_feats(W)
        for fname in lb_origins:
            cut = dict(folds)[fname]
            train_cut = pd.date_range(pd.Timestamp("2025-03-01", tz=tz), cut - pd.Timedelta(hours=240), freq="6h")
            Xtr, ytr, _ = matrix(series, city_df, train_cut, HORIZONS)
            Xev, yev, mev = matrix(series, city_df, [cut], [24,72,168,240])
            rf2, med2 = train_rf(Xtr, ytr, feats)
            pred = np.clip(rf2.predict(Xev[feats].fillna(med2)),0,None)
            d = mev.copy(); d["y"]=yev; d["pred"]=pred; d["fold"]=fname; d["lookback"]=W
            lb_rows.append(d)
            print(f"[LB] W={W} {fname} n={len(d)} rmse={rmse(d.y,d.pred):.3f}", flush=True)
    lb = pd.concat(lb_rows, ignore_index=True)
    lb.to_csv(os.path.join(M02_DIR,"lookback_oof.csv"), index=False)
    print(lb.groupby("lookback").apply(lambda d: rmse(d.y,d.pred), include_groups=False).to_string(), flush=True)

    # --- UPDATE SIMULATION ---
    # Target: fixed 48h window from T. Compare long-lead cutoff (T-10d) vs refreshed (T-2d).
    # Also winter scenario.
    scenarios = []
    for T_str in ["2026-10-15","2025-12-20"]:
        T = pd.Timestamp(T_str, tz=tz)
        for label, C in [("long", T-pd.Timedelta(days=10)), ("refreshed", T-pd.Timedelta(days=2))]:
            train_cut = pd.date_range(pd.Timestamp("2025-03-01", tz=tz), C - pd.Timedelta(hours=240), freq="6h")
            Xtr, ytr, _ = matrix(series, city_df, train_cut, HORIZONS)
            # eval: T .. T+47 (leads relative to C computed below)
            leads = [int((T+pd.Timedelta(hours=h)-C).total_seconds()//3600) for h in range(48)]
            Xev, yev, mev = matrix(series, city_df, [C], leads)
            rf3, med3 = train_rf(Xtr, ytr, FULL_FEATS)
            pred = np.clip(rf3.predict(Xev[FULL_FEATS].fillna(med3)),0,None)
            d = mev.copy(); d["y"]=yev; d["pred"]=pred; d["T"]=str(T); d["mode"]=label
            scenarios.append(d)
            print(f"[UPD] T={T_str} {label} C={C} n={len(d)} rmse={rmse(d.y,d.pred):.3f}", flush=True)
    upd = pd.concat(scenarios, ignore_index=True)
    upd.to_csv(os.path.join(M02_DIR,"update_sim.csv"), index=False)
    print(upd.groupby(["T","mode"]).apply(lambda d: rmse(d.y,d.pred), include_groups=False).to_string(), flush=True)
    print("DONE_M02", flush=True)

if __name__=="__main__":
    main()
