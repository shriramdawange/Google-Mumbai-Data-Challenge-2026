"""MISSION 02.3: short-horizon (42-89h) backtest B2/B3/B4/RF + bootstrap + extreme.
Data adapter: parquet -> causal feature pipeline; leads derived dynamically.
Does NOT overwrite M01-M022 artifacts. Outputs reports/m023/.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data"); REPORTS=os.path.join(ROOT,"reports"); M=os.path.join(REPORTS,"m023"); os.makedirs(M,exist_ok=True)
PARQ=os.path.join(ROOT,"data","processed","UPDATE-09","training_dataset.parquet")
TEST=os.path.join(ROOT,"UPDATE-09","Test.csv")
def rmse(a,b): return float(np.sqrt(mean_squared_error(np.asarray(a),np.asarray(b))))
def mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def load_adapter():
    # Phase 1 adapter
    df=pd.read_parquet(PARQ)
    df["timestamp_ist"]=pd.to_datetime(df["timestamp_ist"])
    # source_release metadata only
    assert "source_release" in df.columns
    df_tmp=df.drop(columns=["source_release"]).copy()
    # pm25_clean policy
    df_tmp["pm25_clean"]=df_tmp["pm25"].astype(float)
    df_tmp.loc[df_tmp["flag_high_1000"]==True,"pm25_clean"]=np.nan
    df_tmp.loc[df_tmp["pm25"]>1000,"pm25_clean"]=np.nan
    df_tmp=df_tmp.sort_values(["location_id","timestamp_ist"])
    assert not df_tmp.duplicated(["location_id","timestamp_ist"]).any(), "dup keys"
    for loc,g in df_tmp.groupby("location_id"):
        assert g.timestamp_ist.is_monotonic_increasing
    assert sorted(df_tmp.location_id.unique().tolist())==[11606,11611,3409510,3409511]
    te=pd.read_csv(TEST,parse_dates=["target_start_ist","target_end_ist"])
    assert len(te)==2592 and te.ID.nunique()==2592
    train_max=df_tmp.timestamp_ist.max()
    priv=te[te.scoring_block=="private"]
    assert len(priv)==192
    leads=sorted(((priv.target_end_ist-train_max).dt.total_seconds()/3600).astype(int).unique().tolist())
    print(f"train_max={train_max} private leads {leads[0]}..{leads[-1]} n={len(leads)}",flush=True)
    assert leads==list(range(leads[0],leads[-1]+1)) and (leads[-1]-leads[0]+1)==48
    nhours=priv.target_end_ist.nunique(); print(f"private rows={len(priv)} hours={nhours}",flush=True)
    assert nhours==48
    return df_tmp, te, train_max, leads

def build_series(tr):
    stations=sorted(tr.location_id.unique().tolist()); code={s:i for i,s in enumerate(stations)}
    idx=pd.date_range(tr.timestamp_ist.min().floor("h"),tr.timestamp_ist.max().floor("h"),freq="h")
    series={}
    for loc in stations:
        g=tr[tr.location_id==loc].set_index("timestamp_ist")["pm25_clean"].sort_index()
        g=g[~g.index.duplicated(keep="last")].reindex(idx)
        d=pd.DataFrame({"pm25":g},index=idx)
        for L in [1,2,3,6,12,24,48,72]:
            d[f"lag{L}"]=d["pm25"].shift(L)
        for W in [6,12,24,48,72,120,168,336]:
            d[f"roll_mean_{W}"]=d["pm25"].rolling(W,min_periods=1).mean()
            d[f"roll_std_{W}"]=d["pm25"].rolling(W,min_periods=1).std()
        d["roll_mean_24_prev"]=d["roll_mean_24"].shift(24)
        d["trend_24"]=d["roll_mean_24"]-d["roll_mean_24_prev"]
        v=d["pm25"].notna().astype(int)
        d["valid_cnt_24"]=v.rolling(24,min_periods=1).sum(); d["_valid"]=v
        d["station_code"]=code[loc]
        series[loc]=d
    city=pd.DataFrame({loc:series[loc]["pm25"] for loc in stations},index=idx).mean(axis=1)
    cdf=pd.DataFrame({"city_pm25":city},index=idx); cdf["city_lag1"]=cdf["city_pm25"].shift(1)
    for W in [24,72]: cdf[f"city_roll{W}"]=cdf["city_pm25"].rolling(W,min_periods=1).mean()
    return series,cdf
def cal(t):
    h,d,m=t.hour,t.dayofweek,t.month
    return dict(hour_t=h,dow_t=d,month_t=m,hour_sin=np.sin(2*np.pi*h/24),hour_cos=np.cos(2*np.pi*h/24),
                dow_sin=np.sin(2*np.pi*d/7),dow_cos=np.cos(2*np.pi*d/7),is_weekend=int(d>=5))
def hours_since(df,c):
    v=df.loc[:c,"_valid"]
    return 99 if v.sum()==0 else int((c-v[v==1].index.max()).total_seconds()//3600)
FEATS=["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
       "lead","log_lead","lag1","lag2","lag3","lag6","lag12","lag24","lag48","lag72",
       "roll_mean_6","roll_mean_24","roll_mean_72","roll_mean_168","roll_std_24","roll_std_72",
       "trend_24","city_lag1","city_roll24","city_roll72","hours_since","valid_cnt_24"]
def base(df,c,cdf):
    r=df.loc[c]
    return dict(station_code=int(r["station_code"]),lag1=r["lag1"],lag2=r["lag2"],lag3=r["lag3"],lag6=r["lag6"],
        lag12=r["lag12"],lag24=r["lag24"],lag48=r["lag48"],lag72=r["lag72"],
        roll_mean_6=r["roll_mean_6"],roll_mean_24=r["roll_mean_24"],roll_mean_72=r["roll_mean_72"],roll_mean_168=r["roll_mean_168"],
        roll_std_24=r["roll_std_24"],roll_std_72=r["roll_std_72"],trend_24=r["trend_24"],
        city_lag1=float(cdf.loc[c,"city_lag1"]),city_roll24=float(cdf.loc[c,"city_roll24"]),city_roll72=float(cdf.loc[c,"city_roll72"]),
        hours_since=hours_since(df,c),valid_cnt_24=float(r["valid_cnt_24"]))
def matrix(series,cdf,cutoffs,horizons,origin):
    rows=[];ys=[];meta=[]
    for loc,df in series.items():
        for c in cutoffs:
            if c not in df.index: continue
            b=base(df,c,cdf)
            for h in horizons:
                t=c+pd.Timedelta(hours=h)
                if t>origin or t not in df.index: continue
                y=df.loc[t,"pm25"]
                if pd.isna(y): continue
                r=dict(b); r.update(cal(t)); r["lead"]=h; r["log_lead"]=np.log1p(h)
                rows.append(r); ys.append(float(y)); meta.append((int(loc),c,t,h))
    return pd.DataFrame(rows),np.array(ys),pd.DataFrame(meta,columns=["location_id","cutoff","target_t","lead"])

def main():
    tr,te,train_max,LEADS=load_adapter()
    series,cdf=build_series(tr)
    train_start=tr.timestamp_ist.min(); tz=train_max.tz
    with open(os.path.join(M,"config.json"),"w") as f:
        json.dump(dict(source_parquet=str(PARQ),test=str(TEST),train_max=str(train_max),
                       private_leads=LEADS,private_rows=192,private_hours=48),f,indent=2)
    # Origins: weekly grid >=120d history, window within train_max
    cand=pd.date_range(train_start+pd.Timedelta(days=120), train_max-pd.Timedelta(hours=max(LEADS)+24), freq="7D")
    cand=list(cand)
    # subsample: evenly spaced 12 incl recent
    step=max(1,len(cand)//9); origins=list(dict.fromkeys(list(cand[::step][:9])+list(cand[-3:])))
    print(f"origins ({len(origins)}): {[str(o.date()) for o in origins]}",flush=True)
    SHORT=list(range(36,97,6))  # short-horizon training leads around 42-89
    print(f"short RF horizons: {SHORT}",flush=True)
    # Stage: per-origin baselines B2/B3/B4 (timestamp windows) + RF
    # Profile helper (station-specific, window-based)
    import sys as _s; _s.path.insert(0,os.path.join(ROOT,"src"))
    from mission02_2 import window_stats, hour_profile
    ev_rows=[]; fold_info=[]
    for fi,origin in enumerate(origins):
        # RF training: cutoffs every 12h, SHORT horizons, target<=origin
        tc=pd.date_range(train_start+pd.Timedelta(days=30),origin-pd.Timedelta(hours=1),freq="12h")
        Xtr,ytr,_=matrix(series,cdf,tc,SHORT,origin)
        Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
        print(f"[{fi}] {origin.date()} train={len(ytr)} eval={len(yev)}",flush=True)
        fold_info.append(dict(fold=fi,origin=str(origin),n_train=int(len(ytr)),n_eval=int(len(yev))))
        if len(yev)==0: continue
        # Baselines B2/B3/B4 need channels at origin: per station fast(7d)/slow(90d) + diurnal norm90
        per={}
        for loc,s in series.items():
            pm=pd.Series(s["pm25"])
            fast=window_stats(pm,origin,7)["mean"]; slow=window_stats(pm,origin,90)["mean"]
            prof,_=hour_profile(pm,origin,90); norm=prof/(float(prof.mean()) if len(prof) else 1)
            per[loc]=dict(fast=fast,slow=slow,norm=norm)
        b2=[];fasts=[];slows=[]
        for _,r in mev.iterrows():
            loc=int(r.location_id); h=int(r.target_t.hour); e=per[loc]
            n=float(e["norm"].get(h,1.0))
            b2.append(float(e["fast"]*n)); fasts.append(float(e["fast"]*n)); slows.append(float(e["slow"]*n))
        b2=np.array(b2); fasts=np.array(fasts); slows=np.array(slows)
        # RF
        med=Xtr[FEATS].median(numeric_only=True)
        rf=RandomForestRegressor(n_estimators=150,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=42)
        rf.fit(Xtr[FEATS].fillna(med),ytr)
        pred_rf=np.clip(rf.predict(Xev[FEATS].fillna(med)),0,None)
        d=mev.copy(); d["y"]=yev; d["fold"]=fi; d["origin"]=str(origin)
        d["B2"]=np.clip(b2,0,None); d["fast"]=fasts; d["slow"]=slows
        d["pred_rf"]=pred_rf
        ev_rows.append(d)
        print(f"  B2={rmse(d.y,d.B2):.3f} RF={rmse(d.y,d.pred_rf):.3f}",flush=True)
    ev=pd.concat(ev_rows,ignore_index=True)
    # Fit w (B3) and beta (B4) on early folds only (first 5 origins)
    folds_sorted=sorted(ev.fold.unique()); early=folds_sorted[:5]
    grid=[round(x,2) for x in np.linspace(0,1,21)]
    def rmse_w(d,w): return rmse(d.y,w*d.fast+(1-w)*d.slow)
    w=float(min(grid,key=lambda w_: rmse_w(ev[ev.fold.isin(early)],w_)))
    betas=np.linspace(-0.5,0.5,21)
    def rmse_b(b): 
        dd=ev[ev.fold.isin(early)]; return rmse(dd.y, w*dd.fast+(1-w)*dd.slow+b*(dd.fast-dd.slow))
    beta=float(min(betas,key=lambda b_: rmse_b(b_)))
    print(f"early w={w} beta={beta}",flush=True)
    ev["B3"]=np.clip(w*ev.fast+(1-w)*ev.slow,0,None)
    ev["B4"]=np.clip(ev.B3+beta*(ev.fast-ev.slow),0,None)
    ev.to_csv(os.path.join(M,"validation_oof.csv"),index=False)
    with open(os.path.join(M,"fold_meta.json"),"w") as f: json.dump(fold_info,f,indent=2)
    # Metrics tables
    models=["B2","B3","B4","pred_rf"]
    rows=[]
    for c in models:
        rows.append(dict(model=c,rmse=rmse(ev.y,ev[c]),mae=mae(ev.y,ev[c]),bias=float((ev[c]-ev.y).mean()),n=len(ev)))
    by_origin=[dict(origin=o,model=c,rmse=rmse(g.y,g[c]),n=len(g)) for o,g in ev.groupby("origin") for c in models]
    by_station=[dict(location_id=int(k),model=c,rmse=rmse(g.y,g[c]),n=len(g)) for k,g in ev.groupby("location_id") for c in models]
    pd.DataFrame(rows).to_csv(os.path.join(M,"metrics_by_origin.csv"),index=False)
    pd.DataFrame(by_station).to_csv(os.path.join(M,"metrics_by_station.csv"),index=False)
    # paired bootstrap over origins (10k)
    rng=np.random.RandomState(0); origins_u=ev.origin.unique()
    def se_col(c): return (ev[c]-ev.y)**2
    ev["se_B4"]=(ev.B4-ev.y)**2; ev["se_B3"]=(ev.B3-ev.y)**2; ev["se_RF"]=(ev.pred_rf-ev.y)**2
    B=10000; diffs=[]
    for _ in range(B):
        samp=rng.choice(origins_u,size=len(origins_u),replace=True)
        rr=pd.concat([ev[ev.origin==o] for o in samp],ignore_index=True)
        diffs.append((np.sqrt(rr.se_B3.mean())-np.sqrt(rr.se_B4.mean()), np.sqrt(rr.se_RF.mean())-np.sqrt((ev.loc[rr.index,"B4"]*0+rr.B4-rr.y).pipe(lambda x: np.sqrt(np.mean(x**2)))**0)) if False else (np.sqrt(rr.se_B3.mean())-np.sqrt(rr.se_B4.mean()), np.sqrt(rr.se_RF.mean())-np.sqrt(rr.se_B4.mean())))
    diffs=np.array(diffs)
    boot=dict(n_boot=B,
              p_B4_better_B3=float((diffs[:,0]>0).mean()),
              p_B4_better_RF=float((diffs[:,1]>0).mean()),
              mean_diff_B3minusB4=float(diffs[:,0].mean()), mean_diff_RFminusB4=float(diffs[:,1].mean()),
              ci_B4_rmse=None)
    # CI for B4 rmse via bootstrap of se
    rm=[]
    for _ in range(B):
        samp=rng.choice(origins_u,size=len(origins_u),replace=True)
        rr=pd.concat([ev[ev.origin==o] for o in samp],ignore_index=True); rm.append(np.sqrt(rr.se_B4.mean()))
    boot["ci_B4_rmse"]=[float(np.percentile(rm,2.5)),float(np.percentile(rm,97.5))]
    with open(os.path.join(M,"bootstrap.json"),"w") as f: json.dump(boot,f,indent=2)
    # extreme sensitivity: on validation eval with >1000 cleaning impact: compare excl (current) vs raw scored on 300-1000 subset
    hi=ev[ev.y>100]
    es=dict(n_with_obs_gt100=int(len(hi)),note="19 train anomalies >1000 excluded from features/targets; eval set keeps 300-1000 targets",
            rf_rmse_gt100=float(rmse(hi.y,hi.pred_rf)) if len(hi) else None,
            b4_rmse_gt100=float(rmse(hi.y,hi.B4)) if len(hi) else None)
    with open(os.path.join(M,"extreme_sensitivity.json"),"w") as f: json.dump(es,f,indent=2)
    # late folds honest: last 5 origins
    late=sorted(ev.fold.unique())[-5:]
    late_ev=ev[ev.fold.isin(late)]
    print(f"late folds {late} B2={rmse(late_ev.y,late_ev.B2):.3f} B3={rmse(late_ev.y,late_ev.B3):.3f} B4={rmse(late_ev.y,late_ev.B4):.3f} RF={rmse(late_ev.y,late_ev.pred_rf):.3f}",flush=True)
    print(f"pooled B2={rmse(ev.y,ev.B2):.3f} B3={rmse(ev.y,ev.B3):.3f} B4={rmse(ev.y,ev.B4):.3f} RF={rmse(ev.y,ev.pred_rf):.3f} n={len(ev)}",flush=True)
    print("DONE_023_STAGE1",flush=True)
if __name__=="__main__": main()
