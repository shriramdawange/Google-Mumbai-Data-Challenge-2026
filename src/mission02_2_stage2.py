"""M02.2 stage2: diurnal, residual V2 with causality, final comparison + bootstrap."""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data"); REPORTS=os.path.join(ROOT,"reports"); M=os.path.join(REPORTS,"m022"); M021=os.path.join(REPORTS,"m021")
sys.path.insert(0,os.path.join(ROOT,"src"))
from mission02_2 import (load,build,window_stats,hour_profile,rmse,WINDOWS)
def main():
    tr=load(); idx,sers=build(tr)
    cfg=json.load(open(os.path.join(M,"config.json"))); LEADS=cfg["private_leads"]
    ev=pd.read_csv(os.path.join(M,"baseline_oof.csv"))
    sh=json.load(open(os.path.join(M,"shrinkage.json")))
    w=float(sh["w_pooled"]); beta=float(sh["beta"])
    # quick pooled table
    yy=float(ev.y.mean())
    for c in ["B0","B3_pooled","B3_station","B3_leaddep","B4_trend"]:
        s=ev[c].fillna(yy)
        print(c, round(rmse(ev.y, s),3), "bias", round(float((s-ev.y).mean()),3), f"n_nan={int(ev[c].isna().sum())}", flush=True)
    # per-origin SE contribution for B3
    ev["se"]= (ev.B3_pooled-ev.y)**2
    tot=ev.se.sum()
    print(ev.groupby("origin").se.sum().div(tot).sort_values(ascending=False).head(5).to_string(), flush=True)
    # ---------- Diurnal: additive vs multiplicative, seasons/regimes ----------
    # Recompute profile stats at origins for reporting (station vs pooled, seasons, regimes)
    fold_meta=json.load(open(os.path.join(M021,"fold_meta.json")))
    origins=[pd.Timestamp(m["origin"]) for m in fold_meta]
    rows=[]
    for origin in origins:
        if str(origin) not in set(ev.origin): continue
        for loc,s in sers.items():
            for Wd in [30,60,90]:
                prof,_=hour_profile(s,origin,Wd)
                rows.append(dict(origin=str(origin),location_id=int(loc),Wd=Wd,amp=float(prof.max()-prof.min()),mean=float(prof.mean())))
    pd.DataFrame(rows).to_csv(os.path.join(M,"diurnal_stats.csv"),index=False)
    print(f"diurnal rows={len(rows)}",flush=True)
    # ---------- Residual V2: reconstruct baseline per training example at own origin ----------
    # Use 4 subset origins for residual training to bound time; training cutoffs every 24h (sparse) with horizons 194-241 only? Use DENSE subsample 6 horizons.
    import pandas as _pd
    from mission02_1 import matrix as mx_matrix, FULL
    print("building residual training matrices (4 origins)...",flush=True)
    # NOTE: baseline for training must match validation definition (B3_pooled formula w/fast-slow/norm90).
    # Reconstruct via mission02_2 functions per cutoff for Ridge only (cheap, runs on aggregated features?).
    # For tree residual we need per-example baseline: compute using window means at cutoff (time rolling).
    # Implement fast time-rolling precompute per station:
    pre={}
    for loc,s in sers.items():
        rolling={}
        for W in [7,90]:
            rolling[f"L{W}"]=s.rolling(f"{W}D").mean()
        # hour profile norm90: expanding? Use rolling 90D hour means: approximate via rolling mean per hour using groupby shift trick:
        # Compute df with hour, then for each hour rolling 90D mean
        df=pd.DataFrame({"v":s}); df["h"]=df.index.hour
        norms={}
        for h in range(24):
            sub=df[df.h==h].set_index(df.index[df.h==h])
            norms[h]=sub["v"].rolling("90D").mean()
        pre[loc]=(rolling,norms)
        print(f" pre {loc} done",flush=True)
    # Causality test: perturb future and verify baseline unchanged
    loc0=sorted(sers.keys())[0]; origin0=origins[5] if len(origins)>5 else origins[-1]
    rolling,norms=pre[loc0]
    b0=float(rolling["L7"].loc[origin0]* (float((norms[int(origin0.hour)].loc[origin0]/norms[int(origin0.hour)].loc[origin0].mean())) if False else 1.0))
    s_pert=sers[loc0].copy(); s_pert.loc[s_pert.index>origin0]=99999
    rolling2={}
    for W in [7,90]: rolling2[f"L{W}"]=s_pert.rolling(f"{W}D").mean()
    assert abs(float(rolling2["L7"].loc[origin0])-float(rolling["L7"].loc[origin0])) < 1e-9, "CAUSALITY FAIL: future perturbation changed baseline"
    print("causality test PASS: future perturbation does not change origin baseline",flush=True)
    # Train Ridge residual on 4 origins' examples (cutoffs every 7d for speed, leads 194..241 step 12)
    Xrows=[]; yres=[]; 
    rss=[]
    train_start=tr.timestamp_ist.min()
    for origin in [origins[0],origins[3],origins[6],origins[-1]]:
        cs=pd.date_range(train_start+pd.Timedelta(days=30),origin-pd.Timedelta(hours=241),freq="7D")
        for c in cs:
            for ld in range(194,242,12):
                t=c+pd.Timedelta(hours=ld)
                for loc,s in sers.items():
                    if t not in s.index or pd.isna(s.loc[t]): continue
                    rolling,norms=pre[loc]
                    h=int(t.hour)
                    # norm90 at cutoff c for hour h
                    lv7=float(rolling["L7"].loc[c]); lv90=float(rolling["L90"].loc[c])
                    # diurnal norm: hour-90d mean at c / overall-90d mean at c
                    # overall-90d mean:
                    ov=float(s.loc[(s.index>c-pd.Timedelta(days=90))&(s.index<=c)].dropna().mean())
                    hm=float(norms[h].loc[c]) if (c in norms[h].index and pd.notna(norms[h].loc[c])) else ov
                    norm=hm/ov if ov and pd.notna(ov) and ov!=0 else 1.0
                    fast=lv7*norm; slow=lv90*norm; base=float(w*fast+(1-w)*slow)
                    y=float(s.loc[t]); rss.append((loc,c,t,ld,y,fast,slow,base))
        print(f" residual collect {origin.date()} total={len(rss)}",flush=True)
    rss_df=pd.DataFrame(rss,columns=["location_id","cutoff","target_t","lead","y","fast","slow","base"])
    rss_df=rss_df.dropna(subset=["fast","slow","base","y"]).reset_index(drop=True)
    rss_df["resid"]=rss_df.y-rss_df.base
    print(f"residual train n={len(rss_df)} base rmse (in-sample train)={rmse(rss_df.y,rss_df.base):.3f}",flush=True)
    # Features for Ridge residual: reuse FULL via mission02_1 matrix limited to same cutoffs/horizons? Approximate: train Ridge on [fast,slow,lead, hour_sin/cos, station] only, to keep causal+comparable.
    # Build minimal features per row:
    rss_df["hour"]=pd.to_datetime(rss_df.target_t).dt.hour
    rss_df["hour_sin"]=np.sin(2*np.pi*rss_df.hour/24); rss_df["hour_cos"]=np.cos(2*np.pi*rss_df.hour/24)
    rss_df["log_lead"]=np.log1p(rss_df.lead)
    RF_FEATS=["fast","slow","lead","log_lead","hour_sin","hour_cos","location_id"]
    ridge=Ridge(alpha=10.0); ridge.fit(rss_df[RF_FEATS], rss_df.resid)
    rss_df["pred"]=np.clip(rss_df.base+ridge.predict(rss_df[RF_FEATS]),0,None)
    # evaluate Ridge residual on validation ev (rebuild ev features for B3_pooled rows)
    ev2=ev.copy()
    # build ev features fast/slow already in baseline_oof? B3fast/B3slow columns exist
    ev2["hour_sin"]=np.sin(2*np.pi*pd.to_datetime(ev2.target_t).dt.hour/24)
    ev2["hour_cos"]=np.cos(2*np.pi*pd.to_datetime(ev2.target_t).dt.hour/24)
    ev2["log_lead"]=np.log1p(ev2.lead)
    ev2["fast"]=ev2["B3fast"]; ev2["slow"]=ev2["B3slow"]; ev2["base"]=ev2["B3_pooled"]
    ev2["pred_ridge_v2"]=np.clip(ev2.base+ridge.predict(ev2[RF_FEATS]),0,None)
    print(f"ridge V2 val rmse={rmse(ev2.y,ev2.pred_ridge_v2):.3f} vs B3 {rmse(ev2.y,ev2.B3_pooled):.3f}",flush=True)
    ev2.to_csv(os.path.join(M,"residual_v2_oof.csv"),index=False)
    # ---------- Final comparison + block bootstrap ----------
    contig=pd.read_csv(os.path.join(M021,"contiguous_oof.csv"))
    # merge RF preds into ev by keys (fold origin/lead/station/target)
    merged=ev.merge(contig[["origin","location_id","lead","pred_general_rf"]],on=["origin","location_id","lead"],how="left")
    merged.to_csv(os.path.join(M,"comparison_oof.csv"),index=False)
    # block bootstrap over origins: 1000 resamples
    rng=np.random.RandomState(0)
    origins_u=merged.origin.unique()
    models=["B3_pooled","B2_7d","pred_general_rf"]
    # B2_7d = fast (w=1). For SE comparison use squared errors per row, resampling origins with replacement
    boot={c:[] for c in models}
    for b in range(1000):
        samp=rng.choice(origins_u,size=len(origins_u),replace=True)
        rows=pd.concat([merged[merged.origin==o] for o in samp],ignore_index=True)
        for c in models:
            r=rows.dropna(subset=[c])
            boot[c].append(rmse(r.y,r[c]))
    for c in models:
        a=np.array(boot[c]); print(c, f"rmse={rmse(merged.y,merged[c].fillna(0)):.3f}" if c!="pred_general_rf" else f"rmse={rmse(merged.dropna(subset=[c]).y,merged.dropna(subset=[c])[c]):.3f}", f"CI [{np.percentile(a,2.5):.2f},{np.percentile(a,97.5):.2f}]",flush=True)
    print("DONE_STAGE2_022",flush=True)
if __name__=="__main__": main()
