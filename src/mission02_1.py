"""MISSION 02.1: contiguous 48h window validation, dense horizons, B0-B3, residual,
lookback v2, missingness, ablation, extreme sensitivity, corrected update sim.
Preserves M01/M02 artifacts. Outputs to reports/m021/.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); REPORTS = os.path.join(ROOT, "reports")
M = os.path.join(REPORTS, "m021"); os.makedirs(M, exist_ok=True)

LOOKBACKS=[6,12,24,48,72,120,168,336]
# Dense grid: short + 72..288 step 6
DENSE=[1,6,12,24,48]+list(range(72,289,6))  # 42 horizons
LONG_ONLY=list(range(168,289,6))  # specialized
def load():
    tr=pd.read_csv(os.path.join(DATA,"Train.csv"), parse_dates=["timestamp_ist","available_at"])
    tr["pm25_clean"]=tr["pm25"].astype(float)
    tr.loc[tr["flag_high_1000"]==True,"pm25_clean"]=np.nan
    tr.loc[tr["pm25"]>1000,"pm25_clean"]=np.nan
    return tr.sort_values(["location_id","timestamp_ist"])
def build_series(tr):
    stations=sorted(tr.location_id.unique().tolist()); code={s:i for i,s in enumerate(stations)}
    idx=pd.date_range(tr.timestamp_ist.min().floor("h"), tr.timestamp_ist.max().floor("h"), freq="h")
    series={}
    for loc in stations:
        g=tr[tr.location_id==loc].set_index("timestamp_ist")["pm25_clean"].sort_index()
        g=g[~g.index.duplicated(keep="last")].reindex(idx)
        df=pd.DataFrame({"pm25":g}, index=idx)
        for L in [1,2,3,6,12,24,48,72,120,168,336]:
            df[f"lag{L}"]=df["pm25"].shift(L)
        for W in LOOKBACKS:
            df[f"roll_mean_{W}"]=df["pm25"].rolling(W,min_periods=1).mean()
            df[f"roll_med_{W}"]=df["pm25"].rolling(W,min_periods=1).median()
            df[f"roll_std_{W}"]=df["pm25"].rolling(W,min_periods=1).std()
        df["roll_mean_24_prev"]=df["roll_mean_24"].shift(24)
        df["trend_24"]=df["roll_mean_24"]-df["roll_mean_24_prev"]
        # missingness (causal, at hour t using history <= t): valid mask
        valid=df["pm25"].notna().astype(int)
        df["miss_hours_since"]=None  # computed per-cutoff via valid history (see base_row)
        df["valid_cnt_24"]=valid.rolling(24,min_periods=1).sum()
        df["valid_cnt_72"]=valid.rolling(72,min_periods=1).sum()
        df["valid_cnt_168"]=valid.rolling(168,min_periods=1).sum()
        df["cov_72"]=df["valid_cnt_72"]/72.0; df["cov_168"]=df["valid_cnt_168"]/168.0
        df["recent_median_24"]=df["pm25"].rolling(24,min_periods=1).median()
        df["station_code"]=code[loc]; df["_valid"]=valid
        series[loc]=df
    city=pd.DataFrame({loc:series[loc]["pm25"] for loc in stations},index=idx).mean(axis=1)
    cdf=pd.DataFrame({"city_pm25":city},index=idx); cdf["city_lag1"]=cdf["city_pm25"].shift(1)
    for W in LOOKBACKS: cdf[f"city_roll{W}"]=cdf["city_pm25"].rolling(W,min_periods=1).mean()
    return idx,series,cdf
def cal_feats(t):
    h,d,m=t.hour,t.dayofweek,t.month
    return dict(hour_t=h,dow_t=d,month_t=m,hour_sin=np.sin(2*np.pi*h/24),hour_cos=np.cos(2*np.pi*h/24),
                dow_sin=np.sin(2*np.pi*d/7),dow_cos=np.cos(2*np.pi*d/7),is_weekend=int(d>=5))
def hours_since_last(df, c):
    # hours since last valid obs <= c ; 99 if none
    v=df.loc[:c,"_valid"]; 
    if v.sum()==0: return 99
    last=v[v==1].index.max(); return int((c-last).total_seconds()//3600)
FULL=["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
      "lead","log_lead","lag1","lag2","lag3","lag6","lag12","lag24","lag48","lag72","lag120","lag168","lag336",
      "roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168","roll_mean_336","roll_std_24","roll_std_72",
      "trend_24","city_lag1","city_roll24","city_roll72",
      "hours_since","valid_cnt_24","cov_72","cov_168","recent_median_24","roll_med_24"]
def base_row(df,c,cdf):
    crow=df.loc[c]
    b=dict(station_code=int(crow["station_code"]),
        lag1=crow["lag1"],lag2=crow["lag2"],lag3=crow["lag3"],lag6=crow["lag6"],lag12=crow["lag12"],
        lag24=crow["lag24"],lag48=crow["lag48"],lag72=crow["lag72"],lag120=crow["lag120"],lag168=crow["lag168"],lag336=crow["lag336"],
        roll_mean_6=crow["roll_mean_6"],roll_mean_12=crow["roll_mean_12"],roll_mean_24=crow["roll_mean_24"],
        roll_mean_48=crow["roll_mean_48"],roll_mean_72=crow["roll_mean_72"],
        roll_mean_120=crow["roll_mean_120"],roll_mean_168=crow["roll_mean_168"],roll_mean_336=crow["roll_mean_336"],
        roll_std_6=crow["roll_std_6"],roll_std_12=crow["roll_std_12"],roll_std_24=crow["roll_std_24"],
        roll_std_48=crow["roll_std_48"],roll_std_72=crow["roll_std_72"],roll_std_120=crow["roll_std_120"],
        roll_std_168=crow["roll_std_168"],roll_std_336=crow["roll_std_336"],
        roll_med_24=crow["roll_med_24"],trend_24=crow["trend_24"],
        city_lag1=float(cdf.loc[c,"city_lag1"]) if c in cdf.index else np.nan,
        city_roll24=float(cdf.loc[c,"city_roll24"]) if c in cdf.index else np.nan,
        city_roll72=float(cdf.loc[c,"city_roll72"]) if c in cdf.index else np.nan,
        hours_since=hours_since_last(df,c), valid_cnt_24=float(crow["valid_cnt_24"]),
        cov_72=float(crow["cov_72"]), cov_168=float(crow["cov_168"]),
        recent_median_24=float(crow["recent_median_24"]))
    return b
def matrix(series,cdf,cutoffs,horizons,origin,subsample_h=None,rng=None):
    rows=[];ys=[];meta=[]
    for loc,df in series.items():
        for c in cutoffs:
            if c not in df.index: continue
            base=base_row(df,c,cdf)
            hs=horizons
            if subsample_h and len(hs)>subsample_h:
                # stratified: keep 2 short + subsample long
                short=[h for h in hs if h<72]; long=[h for h in hs if h>=72]
                take=rng.choice(long, size=min(subsample_h-len(short), len(long)), replace=False).tolist()
                hs=sorted(short+take)
            for h in hs:
                t=c+pd.Timedelta(hours=h)
                if t>origin or t not in df.index: continue  # per-example availability: target <= origin
                y=df.loc[t,"pm25"]
                if pd.isna(y): continue
                r=dict(base); r.update(cal_feats(t)); r["lead"]=h; r["log_lead"]=np.log1p(h)
                rows.append(r); ys.append(float(y)); meta.append((int(loc),c,t,h))
    X=pd.DataFrame(rows); y=np.array(ys) if rows else np.array([])
    meta=pd.DataFrame(meta,columns=["location_id","cutoff","target_t","lead"]) if rows else pd.DataFrame(columns=["location_id","cutoff","target_t","lead"])
    return X,y,meta
def eval_window(series,cdf,origin,leads):
    return matrix(series,cdf,[origin],list(leads),origin=pd.Timestamp.max.tz_localize(None) if False else series[next(iter(series))].index.max())
def rmse(a,b): return float(np.sqrt(mean_squared_error(np.asarray(a),np.asarray(b))))
def train_rf(X,y,feats,n=150):
    med=X[feats].median(numeric_only=True)
    rf=RandomForestRegressor(n_estimators=n,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=42)
    rf.fit(X[feats].fillna(med),y); return rf,med
def save(df,name):
    df.to_csv(os.path.join(M,name),index=False); print(f"saved {name} {df.shape}",flush=True)

def main():
    tr=load(); _,series,cdf=build_series(tr)
    train_max=tr.timestamp_ist.max(); train_start=tr.timestamp_ist.min(); tz=train_max.tz
    # Auto private leads: target_end from Nov11 01:00 to Nov13 00:00 (48h) minus train_max
    priv_start=pd.Timestamp("2026-11-11 01:00",tz=tz); priv_end=pd.Timestamp("2026-11-13 00:00",tz=tz)
    LEADS=[int((priv_start+pd.Timedelta(hours=i)-train_max).total_seconds()//3600) for i in range(48)]
    print(f"train_max={train_max} private leads={LEADS[0]}..{LEADS[-1]} n={len(LEADS)}",flush=True)
    with open(os.path.join(M,"config.json"),"w") as f:
        json.dump(dict(train_start=str(train_start),train_max=str(train_max),private_leads=LEADS,dense=DENSE,lookbacks=LOOKBACKS),f,indent=2)
    # Origins: weekly sampling, >=120d history, target window within train_max
    cand=pd.date_range(pd.Timestamp("2025-06-23",tz=tz), train_max-pd.Timedelta(hours=max(LEADS)+24), freq="7D")
    cand=[c for c in cand if (c-train_start).days>=120]
    # subsample to 12 spanning seasons, always including the 3 most recent origins (autumn relevance)
    step=max(1,len(cand)//9); origins=list(dict.fromkeys(list(cand[::step][:9])+list(cand[-3:])))
    print(f"origins ({len(origins)}): {[str(o.date()) for o in origins]}",flush=True)
    rng=np.random.RandomState(42)
    # ---------- Stage 1: contiguous validation: baselines B0-B3 + general RF + specialized RF ----------
    # B-profiles need history-only computation per origin: compute profile from series <= origin
    oof=[]
    fold_meta=[]
    for fi,origin in enumerate(origins):
        # training cutoffs: every 12h from max(train_start+7d) to origin-1h
        tc=pd.date_range(train_start+pd.Timedelta(days=7), origin-pd.Timedelta(hours=1), freq="12h")
        Xtr,ytr,mtr=matrix(series,cdf,tc,DENSE,origin,subsample_h=12,rng=rng)
        Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
        print(f"[S1] {fi} {origin.date()} train={len(ytr)} eval={len(yev)}",flush=True)
        fold_meta.append(dict(fold=fi,origin=str(origin),n_train=len(ytr),n_eval=len(yev)))
        if len(yev)==0:
            print(f"  SKIP fold {fi}: zero observed targets in 194-241h window (missing coverage gap)",flush=True)
            continue
        # weights: inverse frequency of target_t (balance duplicates from overlapping cutoff/lead)
        if len(mtr)>0:
            cnt=mtr["target_t"].value_counts()
            w=mtr["target_t"].map(cnt).rdiv(1.0).values; w=w/w.mean()
            dup_pearson=len(mtr)-mtr["target_t"].nunique()
            print(f"  dup-target extra rows={dup_pearson} maxfreq={cnt.max()}",flush=True)
        else:
            w=np.array([])
        # baselines: compute from history <= origin only
        # B0: station x hour profile
        preds={}
        # build per-station profile + recent means from series pm25 <= origin (use df.loc[:origin])
        for loc,df in series.items():
            pass
        # vectorized per eval row:
        # precompute dictionaries
        b0_map={}; b1_7={}; b1_14={}; b1_30={}; levels={}; diur={}
        for loc,df in series.items():
            hist=df.loc[:origin,"pm25"].dropna()
            by_hour=hist.groupby(hist.index.hour).mean()
            tot=hist.mean()
            b0_map[loc]=by_hour
            levels[loc]=hist.iloc[-24:].mean() if len(hist)>=24 else hist.mean()
            b1_7[loc]=hist.iloc[-168:].mean() if len(hist)>=168 else hist.mean()
            b1_14[loc]=hist.iloc[-336:].mean() if len(hist)>=336 else hist.mean()
            b1_30[loc]=hist.iloc[-720:].mean() if len(hist)>=720 else hist.mean()
            diur[loc]=(by_hour/(tot if tot!=0 else 1))
        p_b0=[];p_b1=[];p_b2=[];p_b3=[]
        for _,r in mev.iterrows():
            loc=int(r.location_id); h=int(r.target_t.hour)
            prof=float(b0_map[loc].get(h, np.nan))
            p_b0.append(prof)
            r7,b1_14v,b1_30v=b1_7[loc],b1_14[loc],b1_30[loc]
            # B1: mean of 7/14/30 (equal)
            p_b1.append(float(np.nanmean([r7,b1_14v,b1_30v])))
            # B2: recent level (24h) * normalised diurnal
            d=float(diur[loc].get(h,1.0))
            p_b2.append(float(levels[loc]*d))
            # B3: seasonal = 30d mean * diurnal, adjusted 50% toward recent level
            seas=float(b1_30v*d)
            p_b3.append(float(0.5*seas+0.5*levels[loc]*d))
        gmean=float(yev.mean()) if len(yev)>0 else 30.0
        def _fill(a): 
            a=np.array(a,dtype=float); a=np.where(np.isnan(a),gmean,a); return np.clip(a,0,None)
        preds["B0_hour_profile"]=_fill(p_b0)
        preds["B1_recent_means"]=_fill(p_b1)
        preds["B2_level_x_diurnal"]=_fill(p_b2)
        preds["B3_seasonal_adj"]=_fill(p_b3)
        # models
        rf,med=train_rf(Xtr,ytr,FULL)
        # sample_weight variant: train with weights
        med_w=Xtr[FULL].median(numeric_only=True)
        rf_w=RandomForestRegressor(n_estimators=150,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=42)
        rf_w.fit(Xtr[FULL].fillna(med_w),ytr,sample_weight=w)
        # specialized: train only long horizons 168-288
        tc2=pd.date_range(train_start+pd.Timedelta(days=7), origin-pd.Timedelta(hours=1), freq="12h")
        Xtr2,ytr2,_=matrix(series,cdf,tc2,LONG_ONLY,origin,subsample_h=10,rng=rng)
        rfs,meds=train_rf(Xtr2,ytr2,FULL)
        d=mev.copy(); d["y"]=yev; d["fold"]=fi; d["origin"]=str(origin)
        d["pred_general_rf"]=np.clip(rf.predict(Xev[FULL].fillna(med)),0,None)
        d["pred_general_rf_w"]=np.clip(rf_w.predict(Xev[FULL].fillna(med_w)),0,None)
        d["pred_specialised_rf"]=np.clip(rfs.predict(Xev[FULL].fillna(meds)),0,None)
        for k,v in preds.items(): d[f"pred_{k}"]=v
        d["pred_persist"]=np.clip(Xev["lag1"].values,0,None)
        oof.append(d)
        print(f"  general={rmse(d.y,d.pred_general_rf):.3f} spec={rmse(d.y,d.pred_specialised_rf):.3f} B0={rmse(d.y,d.pred_B0_hour_profile):.3f} B1={rmse(d.y,d.pred_B1_recent_means):.3f} B2={rmse(d.y,d.pred_B2_level_x_diurnal):.3f} B3={rmse(d.y,d.pred_B3_seasonal_adj):.3f}",flush=True)
    oof=pd.concat(oof,ignore_index=True)
    save(oof,"contiguous_oof.csv")
    with open(os.path.join(M,"fold_meta.json"),"w") as f: json.dump(fold_meta,f,indent=2)
    # residual stage reuses Stage-1 matrices? Refit per origin residual models using best baseline (pick best pooled B1-B3 by overall S1)
    # Determine best baseline pooled
    cand_cols=["pred_B0_hour_profile","pred_B1_recent_means","pred_B2_level_x_diurnal","pred_B3_seasonal_adj"]
    best_B=min(cand_cols,key=lambda c: rmse(oof.y,oof[c]))
    print(f"best baseline={best_B} rmse={rmse(oof.y,oof[best_B]):.3f}",flush=True)
    res_rows=[]
    for _,row in oof.head(0).iterrows(): pass
    # Rerun per origin for residuals (forest small): reuse origin loop quickly with stored? Refactor: loop again
    for fi,origin in enumerate(origins):
        tc=pd.date_range(train_start+pd.Timedelta(days=7), origin-pd.Timedelta(hours=1), freq="12h")
        Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=12,rng=np.random.RandomState(99))
        Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
        if len(yev)==0: continue
        d0=oof[oof.fold==fi].sort_values(["location_id","lead"]).reset_index(drop=True)
        # recompute best baseline preds aligned: use d0's baseline preds for training? Need baseline values for training rows: approximate by mapping best_B prediction function? Training baseline must be causal per-sample cutoff.
        # Approximation: for residual training, compute baseline for each training row from its cutoff's 24h level*diurnal using precomputed? Use last-24 mean at cutoff * B0 profile? We approximate with Xtr roll_mean_24 adjusted by hour profile from training period only (leak-free-ish). For honesty, compute B2-like: roll_mean_24 * (global hour profile normalized, computed on data <= origin only)
        # Build normalized profile from history <= origin (station-level pooled normalized)
        prof_by_station={}
        for loc,df in series.items():
            hist=df.loc[:origin,"pm25"].dropna()
            byh=hist.groupby(hist.index.hour).mean(); prof_by_station[loc]=byh/hist.mean()
        base_tr=[]
        # need station per row: Xtr station_code -> location (map code back)
        code2loc={v:k for k,v in {s:i for i,s in enumerate(sorted(series.keys()))}.items()}
        for i in range(len(Xtr)):
            sc=int(Xtr["station_code"].iloc[i]); loc=code2loc[sc]; h=int(Xtr["hour_t"].iloc[i])
            base_tr.append(float(Xtr["roll_mean_24"].iloc[i]*float(prof_by_station[loc].get(h,1.0))))
        base_tr=np.array(base_tr); base_ev=d0[best_B].values
        resid_tr=ytr-base_tr
        # Ridge
        med=Xtr[FULL].median(numeric_only=True); Xm=Xtr[FULL].fillna(med); Xem=Xev[FULL].fillna(med)
        ridge=Ridge(alpha=10.0); ridge.fit(Xm,resid_tr); pr_ridge=np.clip(base_ev+ridge.predict(Xem),0,None)
        # Shallow HGB
        hgb=HistGradientBoostingRegressor(max_iter=150,learning_rate=0.08,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=0.5,random_state=42)
        hgb.fit(Xm,resid_tr); pr_hgb=np.clip(base_ev+hgb.predict(Xem),0,None)
        # RF residual
        rfr=RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_leaf=6,n_jobs=-1,random_state=42)
        rfr.fit(Xm,resid_tr); pr_rfr=np.clip(base_ev+rfr.predict(Xem),0,None)
        d=d0.copy()
        d["pred_ridge_resid"]=pr_ridge; d["pred_hgb_resid"]=pr_hgb; d["pred_rf_resid"]=pr_rfr
        res_rows.append(d)
        print(f"[RES] {fi} {origin.date()} ridge={rmse(d.y,d.pred_ridge_resid):.3f} hgb={rmse(d.y,d.pred_hgb_resid):.3f} rf={rmse(d.y,d.pred_rf_resid):.3f} base={rmse(d.y,d[best_B]):.3f}",flush=True)
    reso=pd.concat(res_rows,ignore_index=True)
    save(reso,"residual_oof.csv")
    with open(os.path.join(M,"best_baseline.json"),"w") as f: json.dump(dict(best_B=best_B),f)
    # ---------- Stage 2: lookback v2 (subset 6 origins) ----------
    lb_origins=origins[::2]  # 6 origins
    lb_rows=[]
    lagmap={6:"lag6",12:"lag12",24:"lag24",48:"lag48",72:"lag72",120:"lag120",168:"lag168",336:"lag336"}
    for W in LOOKBACKS:
        feats=["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","lead","log_lead","lag1",lagmap[W],f"roll_mean_{W}",f"roll_std_{W}","city_roll24","hours_since","valid_cnt_24","cov_72","cov_168","recent_median_24"]
        # ensure roll_std_W exists for all W (built)
        for fi,origin in enumerate(lb_origins):
            tc=pd.date_range(train_start+pd.Timedelta(days=7), origin-pd.Timedelta(hours=1), freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=10,rng=np.random.RandomState(7))
            Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
            if len(yev)==0: continue
            rf2,med2=train_rf(Xtr,ytr,feats,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rf2.predict(Xev[feats].fillna(med2)),0,None); d["lookback"]=W; d["origin"]=str(origin)
            lb_rows.append(d)
            print(f"[LB2] W={W} {origin.date()} n={len(d)} rmse={rmse(d.y,d.pred):.3f}",flush=True)
    lb=pd.concat(lb_rows,ignore_index=True); save(lb,"lookback2_oof.csv")
    # ---------- Stage 3: ablation (4 origins) ----------
    ab_origins=origins[::3]  # 4 origins
    groups={"lead":["lead","log_lead"],"month":["month_t"],"dow":["dow_t","dow_sin","dow_cos","is_weekend"],
            "hour":["hour_t","hour_sin","hour_cos"],"longterm_level":["roll_mean_168","roll_mean_336","lag168","lag336"],
            "recent_level":["roll_mean_6","roll_mean_24","lag1","lag2","lag3","lag6","lag12","lag24","lag48","lag72","lag120"],
            "diurnal":["hour_t","hour_sin","hour_cos"],"station":["station_code"]}
    ab_rows=[]
    for gname,drop in groups.items():
        feats=[c for c in FULL if c not in drop]
        # diurnal and hour same drop intentionally (reported separately as same ablation); keep both entries
        for origin in ab_origins:
            tc=pd.date_range(train_start+pd.Timedelta(days=7),origin-pd.Timedelta(hours=1),freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(11))
            Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
            if len(yev)==0: continue
            rf2,med2=train_rf(Xtr,ytr,feats,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rf2.predict(Xev[feats].fillna(med2)),0,None); d["ablation"]=gname; d["origin"]=str(origin)
            ab_rows.append(d)
        print(f"[ABL] drop {gname} pooled rmse={rmse(pd.concat(ab_rows).query('ablation==@gname').y, pd.concat(ab_rows).query('ablation==@gname').pred):.3f}",flush=True)
    ab=pd.concat(ab_rows,ignore_index=True); save(ab,"ablation_oof.csv")
    # ---------- Stage 4: extreme sensitivity (3 origins: winter, summer, autumn) ----------
    # pick empirically: first origin (winter/autumn?), middle, last
    ext_origins=[origins[0],origins[len(origins)//2],origins[-1]]
    # raw train targets include >1000 (rebuild with raw? here pm25_clean already excludes; variant 'raw' needs original >1000 rows)
    tr_raw=pd.read_csv(os.path.join(DATA,"Train.csv"), parse_dates=["timestamp_ist"])
    _,series_raw,_=build_series_raw(tr_raw)
    variants={}
    for origin in ext_origins:
        print(f"[EXT] {origin.date()}",flush=True)
        # variant A: current (>1000 excl, mean rolling) = FULL
        tc=pd.date_range(train_start+pd.Timedelta(days=7),origin-pd.Timedelta(hours=1),freq="12h")
        Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(13))
        Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
        if len(yev)==0: continue
        # eval targets preserved from clean series (same for all); note raw targets identical except >1000 which are NaN in clean (not scored anyway)
        rfA,medA=train_rf(Xtr,ytr,FULL,n=100)
        pA=np.clip(rfA.predict(Xev[FULL].fillna(medA)),0,None)
        # variant B: median rolling (replace roll_mean with roll_med)
        featsB=[c if not c.startswith("roll_mean") else f"roll_med_{c.split('_')[-1]}" for c in FULL]
        featsB=[f if f in Xtr.columns else "roll_med_24" for f in featsB]
        rfB,medB=train_rf(Xtr,ytr,featsB,n=100); pB=np.clip(rfB.predict(Xev[featsB].fillna(medB)),0,None)
        # variant C: robust - log1p target training then expm1
        rfC,medC=RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=42),Xtr[FULL].median(numeric_only=True)
        rfC.fit(Xtr[FULL].fillna(medC),np.log1p(ytr)); pC=np.clip(np.expm1(rfC.predict(Xev[FULL].fillna(medC))),0,None)
        # variant D: raw (include >1000) - train on series_raw
        XtrR,ytrR,_=matrix(series_raw,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(13))
        rfD,medD=train_rf(XtrR,ytrR,FULL,n=80); pD=np.clip(rfD.predict(Xev[FULL].fillna(medD)),0,None)
        d=mev.copy(); d["y"]=yev; d["origin"]=str(origin)
        d["pred_excl_mean"]=pA; d["pred_excl_median"]=pB; d["pred_robust_log"]=pC; d["pred_raw"]=pD
        variants[str(origin)]=d
        print(f"  excl_mean={rmse(d.y,d.pred_excl_mean):.3f} median={rmse(d.y,d.pred_excl_median):.3f} log={rmse(d.y,d.pred_robust_log):.3f} raw={rmse(d.y,d.pred_raw):.3f}",flush=True)
    ext=pd.concat(variants.values(),ignore_index=True); save(ext,"extreme_oof.csv")
    # ---------- Stage 5: corrected update sim (within trained domain) ----------
    # Fixed target window 48h; origins: C_long = T-10d? Must keep leads inside [1,240]. Choose so both modes <=240.
    # T example: pick mid-history T; long cutoff = T-8d (leads 192-239), refreshed = T-2d (48-95). Both inside 240.
    upd_rows=[]
    for T_str in ["2026-09-15","2025-12-25"]:
        T=pd.Timestamp(T_str,tz=tz)
        for label,C in [("old",T-pd.Timedelta(days=8)),("refresh",T-pd.Timedelta(days=2))]:
            tc=pd.date_range(train_start+pd.Timedelta(days=7),C-pd.Timedelta(hours=1),freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,C,subsample_h=10,rng=np.random.RandomState(17))
            leads=[int((T+pd.Timedelta(hours=h)-C).total_seconds()//3600) for h in range(48)]
            assert max(leads)<=240 and min(leads)>=1, f"leads outside trained domain {min(leads)}-{max(leads)}"
            Xev,yev,mev=matrix(series,cdf,[C],leads,origin=series[next(iter(series))].index.max())
            rfU,medU=train_rf(Xtr,ytr,FULL,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rfU.predict(Xev[FULL].fillna(medU)),0,None); d["T"]=str(T); d["mode"]=label
            upd_rows.append(d); print(f"[UPD2] T={T_str} {label} C={C.date()} leads {min(leads)}-{max(leads)} rmse={rmse(d.y,d.pred):.3f}",flush=True)
    upd=pd.concat(upd_rows,ignore_index=True); save(upd,"update_sim2.csv")
    print("DONE_M021",flush=True)

def build_series_raw(tr_raw):
    # same as build_series but no >1000 exclusion (for raw variant); keep flag rows as observed
    tr=tr_raw.copy(); tr["pm25_clean"]=tr["pm25"].astype(float)
    return build_series(tr)

if __name__=="__main__":
    main()
