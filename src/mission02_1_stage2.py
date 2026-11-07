import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data"); REPORTS=os.path.join(ROOT,"reports"); M=os.path.join(REPORTS,"m021")
sys.path.insert(0,os.path.join(ROOT,"src"))
from mission02_1 import (load,build_series,matrix,train_rf,FULL,LOOKBACKS,DENSE,rmse,build_series_raw)
def save(df,name): df.to_csv(os.path.join(M,name),index=False); print(f"saved {name} {df.shape}",flush=True)
def main():
    tr=load(); _,series,cdf=build_series(tr)
    train_start=tr.timestamp_ist.min(); tz=tr.timestamp_ist.max().tz
    cfg=json.load(open(os.path.join(M,"config.json"))); LEADS=cfg["private_leads"]
    fold_meta=json.load(open(os.path.join(M,"fold_meta.json")))
    origins=[pd.Timestamp(m["origin"]) for m in fold_meta if m["n_eval"]>0]
    # lookback
    lb_origins=origins[::2]
    lb_rows=[]; lagmap={6:"lag6",12:"lag12",24:"lag24",48:"lag48",72:"lag72",120:"lag120",168:"lag168",336:"lag336"}
    for W in LOOKBACKS:
        feats=["station_code","hour_t","dow_t","month_t","hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","lead","log_lead","lag1",lagmap[W],f"roll_mean_{W}",f"roll_std_{W}","city_roll24","hours_since","valid_cnt_24","cov_72","cov_168","recent_median_24"]
        for origin in lb_origins:
            tc=pd.date_range(train_start+pd.Timedelta(days=7),origin-pd.Timedelta(hours=1),freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=10,rng=np.random.RandomState(7))
            Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
            if len(yev)==0: continue
            rf2,med2=train_rf(Xtr,ytr,feats,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rf2.predict(Xev[feats].fillna(med2)),0,None); d["lookback"]=W; d["origin"]=str(origin)
            lb_rows.append(d); print(f"[LB2] W={W} {origin.date()} n={len(d)} rmse={rmse(d.y,d.pred):.3f}",flush=True)
    save(pd.concat(lb_rows,ignore_index=True),"lookback2_oof.csv")
    # ablation
    ab_origins=origins[::3]
    groups={"lead":["lead","log_lead"],"month":["month_t"],"dow":["dow_t","dow_sin","dow_cos","is_weekend"],
            "hour":["hour_t","hour_sin","hour_cos"],"longterm_level":["roll_mean_168","roll_mean_336","lag168","lag336"],
            "recent_level":["roll_mean_6","roll_mean_24","lag1","lag2","lag3","lag6","lag12","lag24","lag48","lag72","lag120"],
            "diurnal":["hour_t","hour_sin","hour_cos"],"station":["station_code"]}
    ab_rows=[]
    for gname,drop in groups.items():
        feats=[c for c in FULL if c not in drop]
        for origin in ab_origins:
            tc=pd.date_range(train_start+pd.Timedelta(days=7),origin-pd.Timedelta(hours=1),freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(11))
            Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
            if len(yev)==0: continue
            rf2,med2=train_rf(Xtr,ytr,feats,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rf2.predict(Xev[feats].fillna(med2)),0,None); d["ablation"]=gname; d["origin"]=str(origin)
            ab_rows.append(d)
        tmp=pd.concat(ab_rows); sub=tmp[tmp.ablation==gname]
        print(f"[ABL] drop {gname} pooled rmse={rmse(sub.y,sub.pred):.3f} n={len(sub)}",flush=True)
    save(pd.concat(ab_rows,ignore_index=True),"ablation_oof.csv")
    # extreme
    ext_origins=[origins[0],origins[len(origins)//2],origins[-1]]
    tr_raw=pd.read_csv(os.path.join(DATA,"Train.csv"),parse_dates=["timestamp_ist"])
    _,series_raw,_=build_series_raw(tr_raw)
    outs=[]
    for origin in ext_origins:
        print(f"[EXT] {origin.date()}",flush=True)
        tc=pd.date_range(train_start+pd.Timedelta(days=7),origin-pd.Timedelta(hours=1),freq="12h")
        Xtr,ytr,_=matrix(series,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(13))
        Xev,yev,mev=matrix(series,cdf,[origin],LEADS,origin=series[next(iter(series))].index.max())
        if len(yev)==0: continue
        rfA,medA=train_rf(Xtr,ytr,FULL,n=100); pA=np.clip(rfA.predict(Xev[FULL].fillna(medA)),0,None)
        featsB=[c if not c.startswith("roll_mean") else f"roll_med_{c.split('_')[-1]}" for c in FULL]
        featsB=[f if f in Xtr.columns else "roll_med_24" for f in featsB]
        rfB,medB=train_rf(Xtr,ytr,featsB,n=100); pB=np.clip(rfB.predict(Xev[featsB].fillna(medB)),0,None)
        rfC=RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=4,n_jobs=-1,random_state=42)
        medC=Xtr[FULL].median(numeric_only=True); rfC.fit(Xtr[FULL].fillna(medC),np.log1p(ytr)); pC=np.clip(np.expm1(rfC.predict(Xev[FULL].fillna(medC))),0,None)
        XtrR,ytrR,_=matrix(series_raw,cdf,tc,DENSE,origin,subsample_h=8,rng=np.random.RandomState(13))
        rfD,medD=train_rf(XtrR,ytrR,FULL,n=80); pD=np.clip(rfD.predict(Xev[FULL].fillna(medD)),0,None)
        d=mev.copy(); d["y"]=yev; d["origin"]=str(origin)
        d["pred_excl_mean"]=pA; d["pred_excl_median"]=pB; d["pred_robust_log"]=pC; d["pred_raw"]=pD
        outs.append(d); print(f"  excl_mean={rmse(d.y,d.pred_excl_mean):.3f} median={rmse(d.y,d.pred_excl_median):.3f} log={rmse(d.y,d.pred_robust_log):.3f} raw={rmse(d.y,d.pred_raw):.3f}",flush=True)
    save(pd.concat(outs,ignore_index=True),"extreme_oof.csv")
    # update
    upd_rows=[]
    for T_str in ["2026-09-15","2025-12-25"]:
        T=pd.Timestamp(T_str,tz=tz)
        for label,C in [("old",T-pd.Timedelta(days=8)),("refresh",T-pd.Timedelta(days=2))]:
            tc=pd.date_range(train_start+pd.Timedelta(days=7),C-pd.Timedelta(hours=1),freq="12h")
            Xtr,ytr,_=matrix(series,cdf,tc,DENSE,C,subsample_h=10,rng=np.random.RandomState(17))
            leads=[int((T+pd.Timedelta(hours=h)-C).total_seconds()//3600) for h in range(48)]
            assert max(leads)<=240, f"{min(leads)}-{max(leads)}"
            Xev,yev,mev=matrix(series,cdf,[C],leads,origin=series[next(iter(series))].index.max())
            rfU,medU=train_rf(Xtr,ytr,FULL,n=100)
            d=mev.copy(); d["y"]=yev; d["pred"]=np.clip(rfU.predict(Xev[FULL].fillna(medU)),0,None); d["T"]=str(T); d["mode"]=label
            upd_rows.append(d); print(f"[UPD2] T={T_str} {label} leads {min(leads)}-{max(leads)} rmse={rmse(d.y,d.pred):.3f} n={len(d)}",flush=True)
    save(pd.concat(upd_rows,ignore_index=True),"update_sim2.csv")
    print("DONE_STAGE2",flush=True)
if __name__=="__main__": main()
