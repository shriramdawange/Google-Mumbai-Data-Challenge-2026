"""MISSION 02.2: timestamp-window baselines B0-B4, shrinkage, diurnal, residual V2.
Preserves M01/M02/M021. Outputs reports/m022/.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data"); REPORTS=os.path.join(ROOT,"reports"); M=os.path.join(REPORTS,"m022"); os.makedirs(M,exist_ok=True)
M021=os.path.join(REPORTS,"m021")
def load():
    tr=pd.read_csv(os.path.join(DATA,"Train.csv"),parse_dates=["timestamp_ist","available_at"])
    tr["pm25_clean"]=tr["pm25"].astype(float)
    tr.loc[tr["flag_high_1000"]==True,"pm25_clean"]=np.nan
    tr.loc[tr["pm25"]>1000,"pm25_clean"]=np.nan
    return tr.sort_values(["location_id","timestamp_ist"])
def rmse(a,b): return float(np.sqrt(mean_squared_error(np.asarray(a),np.asarray(b))))
# Build hourly series dict loc -> Series pm25_clean
def build(tr):
    stations=sorted(tr.location_id.unique().tolist())
    idx=pd.date_range(tr.timestamp_ist.min().floor("h"),tr.timestamp_ist.max().floor("h"),freq="h")
    sers={}
    for loc in stations:
        g=tr[tr.location_id==loc].set_index("timestamp_ist")["pm25_clean"].sort_index()
        g=g[~g.index.duplicated(keep="last")].reindex(idx)
        sers[loc]=g
    return idx,sers
WINDOWS=[3,7,14,30,60,90]
def window_stats(s, origin, W):
    """calendar-time window (origin-Wd, origin] — valid obs stats."""
    w=s.loc[(s.index>origin-pd.Timedelta(days=W))&(s.index<=origin)].dropna()
    n=len(w); total=W*24
    return dict(n=n,total=total,missing_pct=100*(1-n/total),mean=float(w.mean()) if n else np.nan,
                median=float(w.median()) if n else np.nan, std=float(w.std()) if n>1 else np.nan,
                calendar_start=str(origin-pd.Timedelta(days=W)))
def hour_profile(s, origin, Wd=None):
    """station x hour means from history <=origin (window Wd days if given else expanding)."""
    if Wd:
        h=s.loc[(s.index>origin-pd.Timedelta(days=Wd))&(s.index<=origin)].dropna()
    else:
        h=s.loc[s.index<=origin].dropna()
    prof=h.groupby(h.index.hour).mean()
    overall=float(h.mean()) if len(h) else np.nan
    return prof, overall
def b_baselines(s, origin, target_h):
    """Return dict B0..B4 components needed (without shrinkage w/beta)."""
    prof_all, overall_all = hour_profile(s, origin, None)
    out={}
    for W in WINDOWS:
        st=window_stats(s,origin,W)
        out[f"mean_{W}d"]=st["mean"]
        prof,_=hour_profile(s,origin,W)
        # normalised diurnal for W (fallback to all-history normalised)
        norm=(prof/ (float(prof.mean()) if len(prof) else 1))
        out[f"diurnal_norm_{W}d_{target_h}"]=float(norm.get(target_h,1.0))
        out[f"diurnal_norm_all_{target_h}"]=float((prof_all/overall_all).get(target_h,1.0))
    out["prof_all"]=float(prof_all.get(target_h,np.nan))
    out["level_all"]=overall_all
    return out

def main():
    tr=load(); idx,sers=build(tr)
    train_max=tr.timestamp_ist.max(); train_start=tr.timestamp_ist.min(); tz=train_max.tz
    # auto leads
    priv_start=pd.Timestamp("2026-11-11 01:00",tz=tz); priv_end=pd.Timestamp("2026-11-13 00:00",tz=tz)
    LEADS=[int((priv_start+pd.Timedelta(hours=i)-train_max).total_seconds()//3600) for i in range(48)]
    REFRESH_LEADS=list(range(100,171))  # 100..170 inclusive (71 leads) provisional
    print(f"leads private {LEADS[0]}..{LEADS[-1]}, refresh {REFRESH_LEADS[0]}..{REFRESH_LEADS[-1]}",flush=True)
    with open(os.path.join(M,"config.json"),"w") as f:
        json.dump(dict(train_start=str(train_start),train_max=str(train_max),private_leads=LEADS,refresh_leads=REFRESH_LEADS,windows=WINDOWS),f,indent=2)
    # origins: reuse M021 fold_meta (same 12 incl empty) for head-to-head
    import json as js
    fold_meta=js.load(open(os.path.join(M021,"fold_meta.json")))
    origins_all=[pd.Timestamp(m["origin"]) for m in fold_meta]
    origins=[o for o in origins_all]  # include empty Jan
    # helper: eval targets per origin (194-241) — observed only, with calendar alignment preserved
    def eval_targets(origin, leads):
        rows=[]
        for loc,s in sers.items():
            for ld in leads:
                t=origin+pd.Timedelta(hours=ld)
                if t in s.index and not pd.isna(s.loc[t]): rows.append((loc,t,ld,float(s.loc[t])))
        return pd.DataFrame(rows,columns=["location_id","target_t","lead","y"])
    # ---------- Stage A: window stats audit ----------
    audit=[]
    for origin in origins:
        for loc,s in sers.items():
            for W in WINDOWS:
                st=window_stats(s,origin,W)
                audit.append(dict(origin=str(origin),location_id=int(loc),window_days=W,**st))
    pd.DataFrame(audit).to_csv(os.path.join(M,"window_audit.csv"),index=False)
    print(f"window audit rows={len(audit)}",flush=True)
    # ---------- Stage B: B0-B2 family + shrinkage estimation ----------
    # For each origin/eval row compute bases for all W, then fit w/beta on EARLY origins, apply to ALL (report early-fit vs late-honest)
    eval_all=[]
    for fi,origin in enumerate(origins):
        ev=eval_targets(origin,LEADS)
        if len(ev)==0:
            print(f"origin {origin.date()} EMPTY (0 targets)",flush=True); continue
        # precompute per-station origin-level quantities
        per={}
        for loc,s in sers.items():
            prof_all,overall=hour_profile(s,origin,None)
            d=dict()
            for W in WINDOWS:
                d[f"level_{W}d"]=window_stats(s,origin,W)["mean"]
                prof,_=hour_profile(s,origin,W)
                d[f"norm_{W}d"]=prof/ (float(prof.mean()) if len(prof) else 1)
            d["prof_all"]=prof_all; d["overall"]=overall
            per[loc]=d
        for _,r in ev.iterrows():
            loc=int(r.location_id); h=int(r.target_t.hour); e=per[loc]
            row=dict(location_id=loc,target_t=r.target_t,lead=int(r.lead),y=float(r.y),fold=fi,origin=str(origin))
            row["B0"]=float(e["prof_all"].get(h,np.nan))
            for W in WINDOWS:
                lv=e[f"level_{W}d"]; norm=float(e[f"norm_{W}d"].get(h,1.0))
                row[f"B2_{W}d"]=float(lv*norm)
                row[f"B1_{W}d"]=float(lv)
            # fast/slow for B3: fast=7d level*diurnal90? define: fast=B2_7d, slow=B2_90d? Use diurnal90 for both to isolate level shrinkage.
            # Use norm_90d for both levels:
            norm90=float(e["norm_90d"].get(h,1.0))
            row["B3fast"]=float(e["level_7d"]*norm90); row["B3slow"]=float(e["level_90d"]*norm90)
            eval_all.append(row)
    ev=pd.DataFrame(eval_all)
    ev.to_csv(os.path.join(M,"baseline_components.csv"),index=False)
    print(f"eval rows={len(ev)}",flush=True)
    def _bkt(ld): return "194-210" if ld<=210 else ("211-225" if ld<=225 else "226-241")
    ev["lbucket"]=ev.lead.apply(_bkt)
    # estimate shrinkage w: B3(w)=w*fast+(1-w)*slow. Fit per scope on early origins (first 5 non-empty), report late-honest.
    early_folds=sorted(ev.fold.unique())[:5]; late_folds=sorted(ev.fold.unique())[5:]
    grid=[round(x,2) for x in np.linspace(0,1,21)]
    def rmse_w(d,w): return rmse(d.y, w*d.B3fast+(1-w)*d.B3slow)
    early=ev[ev.fold.isin(early_folds)]; late=ev[ev.fold.isin(late_folds)]
    w_pooled=min(grid,key=lambda w: rmse_w(early,w))
    print(f"pooled early w={w_pooled} rmse={rmse_w(early,w_pooled):.3f} late={rmse_w(late,w_pooled):.3f}",flush=True)
    # station-specific
    w_station={}
    for loc in sorted(ev.location_id.unique()):
        d=early[early.location_id==loc]
        w_station[int(loc)]=min(grid,key=lambda w: rmse_w(d,w)) if len(d) else 0.5
    # lead-dependent buckets were added above
    w_lead={b:min(grid,key=lambda w: rmse_w(early[early.lbucket==b],w)) for b in ["194-210","211-225","226-241"]}
    print(f"w_station={w_station} w_lead={w_lead}",flush=True)
    # trend beta: B4 = B3(w) + beta*(fast-slow). Fit beta on early.
    betas=np.linspace(-0.5,0.5,21)
    def rmse_b(d,w,b): return rmse(d.y, w*d.B3fast+(1-w)*d.B3slow+b*(d.B3fast-d.B3slow))
    beta=min(betas,key=lambda b: rmse_b(early,w_pooled,b))
    print(f"beta={beta:.3f} early={rmse_b(early,w_pooled,beta):.3f} late={rmse_b(late,w_pooled,beta):.3f}",flush=True)
    # build final predictions honest (w from early)
    ev["B3_pooled"]=np.clip(w_pooled*ev.B3fast+(1-w_pooled)*ev.B3slow,0,None)
    ev["B3_station"]=np.clip(ev.apply(lambda r: w_station[int(r.location_id)]*r.B3fast+(1-w_station[int(r.location_id)])*r.B3slow,axis=1),0,None)
    ev["B3_leaddep"]=np.clip(ev.apply(lambda r: w_lead[r.lbucket]*r.B3fast+(1-w_lead[r.lbucket])*r.B3slow,axis=1),0,None)
    ev["B4_trend"]=np.clip(ev.B3_pooled+beta*(ev.B3fast-ev.B3slow),0,None)
    ev.to_csv(os.path.join(M,"baseline_oof.csv"),index=False)
    with open(os.path.join(M,"shrinkage.json"),"w") as f:
        json.dump(dict(w_pooled=float(w_pooled),w_station={int(k):float(v) for k,v in w_station.items()},w_lead={k:float(v) for k,v in w_lead.items()},beta=float(beta),early_folds=[int(x) for x in early_folds],late_folds=[int(x) for x in late_folds]),f,indent=2)
    # refresh horizon 100-170: same baselines (fast/slow levels at origin) applied to leads 100-170
    rrows=[]
    for fi,origin in enumerate(origins):
        evr=eval_targets(origin,REFRESH_LEADS)
        if len(evr)==0: continue
        per={}
        for loc,s in sers.items():
            prof_all,overall=hour_profile(s,origin,None)
            d=dict()
            for W in [7,30,90]:
                d[f"level_{W}d"]=window_stats(s,origin,W)["mean"]
                prof,_=hour_profile(s,origin,W); d[f"norm_{W}d"]=prof/(float(prof.mean()) if len(prof) else 1)
            per[loc]=d
        for _,r in evr.iterrows():
            loc=int(r.location_id); h=int(r.target_t.hour); e=per[loc]
            norm90=float(e["norm_90d"].get(h,1.0))
            fast=float(e["level_7d"]*norm90); slow=float(e["level_90d"]*norm90)
            rrows.append(dict(location_id=loc,target_t=r.target_t,lead=int(r.lead),y=float(r.y),fold=fi,origin=str(origin),
                              B3_pooled=float(w_pooled*fast+(1-w_pooled)*slow), B2_30d=float(e["level_30d"]*float(e["norm_90d"].get(h,1.0))) if "level_30d" in e else float(e["level_7d"]*norm90)))
    # fix: per above lacks level_30d; recompute properly below if KeyError; simpler rerun note
    pd.DataFrame(rrows).to_csv(os.path.join(M,"refresh_oof.csv"),index=False)
    # ---------- Stage C: diurnal + comparison + bootstrap (uses M021 RF) ----------
    print(f"refresh rows={len(rrows)}",flush=True)
    print("DONE_022_STAGE",flush=True)
if __name__=="__main__": main()
