"""MISSION 02.3 stage2 (v2): UPDATE-10-ready B2 inference interface + validation.
Accepts training Parquet path, Test CSV path, selected model. Fail-loud on schema/horizon checks.
Does NOT write into submissions/. Saves inference diagnostics.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M=os.path.join(ROOT,"reports","m023")
SUPPORTED_MODELS={"B2","B3","B4"}
SUPPORTED_LEADS=(1,400)  # validated forecast-lead domain (hours)

def _load_training(parquet_path):
    df=pd.read_parquet(parquet_path)
    df["timestamp_ist"]=pd.to_datetime(df["timestamp_ist"])
    out=df.drop(columns=["source_release"],errors="ignore").copy()
    assert len(out)>0
    out["pm25_clean"]=out["pm25"].astype(float)
    out.loc[out["flag_high_1000"]==True,"pm25_clean"]=np.nan
    out.loc[out["pm25"]>1000,"pm25_clean"]=np.nan
    return out

def _validate_inputs(df, test_path):
    te=pd.read_csv(test_path,parse_dates=["target_start_ist","target_end_ist"])
    assert len(te)==2592, f"Test must have 2592 rows, got {len(te)}"
    assert te.ID.is_unique and len(te.ID.unique())==2592
    for loc,g in df.groupby("location_id"):
        assert g.timestamp_ist.is_monotonic_increasing, f"non-chrono station {loc}"
    train_max=df.timestamp_ist.max()
    priv=te[te.scoring_block=="private"]
    assert len(priv)==192, f"private must be 192 rows, got {len(priv)}"
    hours=priv.target_end_ist.nunique()
    assert hours==48, f"private must be 48 unique hours, got {hours}"
    leads=sorted(int((t-train_max).total_seconds()//3600) for t in priv.target_end_ist.unique())
    assert leads==list(range(leads[0],leads[-1]+1)), f"private leads not contiguous: {leads}"
    lo,hi=SUPPORTED_LEADS
    assert lo<=leads[0] and leads[-1]<=hi, f"forecast lead {leads[0]}..{leads[-1]} outside supported domain {lo}..{hi}"
    return df, te, train_max, leads, priv

def infer(parquet_path, test_path, selected_model="B2"):
    assert selected_model in SUPPORTED_MODELS, f"Unsupported model {selected_model}"
    df=_load_training(parquet_path)
    df, te, train_max, leads, priv = _validate_inputs(df, test_path)
    from mission02_2 import window_stats, hour_profile
    preds={}
    for loc,g in df.groupby("location_id"):
        s=g.set_index("timestamp_ist")["pm25_clean"].sort_index()
        st90=window_stats(s,train_max,90)
        if pd.isna(st90["mean"]) or st90["n"]<24:
            raise ValueError(f"missing station history for {loc} at cutoff {train_max}")
        lv=window_stats(s,train_max,7)["mean"]
        if pd.isna(lv):
            raise ValueError(f"missing 7d level for {loc} at cutoff {train_max}")
        prof,_=hour_profile(s,train_max,90); norm=prof/(float(prof.mean()) if len(prof) else 1.0)
        preds[int(loc)]=(float(lv),norm)
    rows=[]
    for _,r in te.iterrows():
        loc=int(r["location_id"]); h=int(r["target_end_ist"].hour)
        lv,norm=preds[loc]
        t=float(max(0.0, lv*float(norm.get(h,1.0))))
        rows.append(dict(ID=r["ID"], Target=float(t), location_id=loc,
                         lead=int((r["target_end_ist"]-train_max).total_seconds()//3600)))
    out=pd.DataFrame(rows)
    assert len(out)==2592 and out["ID"].is_unique
    assert (out["ID"].values==te["ID"].values).all()
    assert out["Target"].notna().all() and np.isfinite(out["Target"].values).all() and (out["Target"].values>=0).all()
    diag=dict(selected_model=selected_model,
              parquet_path=parquet_path, test_path=test_path,
              train_max=str(train_max),
              private_rows=int(len(priv)), private_hours=int(priv.target_end_ist.nunique()),
              private_leads=[int(leads[0]),int(leads[-1])],
              inference_target_rows=int(len(out)),
              inference_full_rows=int(len(out)),
              inference_mean=float(out["Target"].mean()), inference_min=float(out["Target"].min()), inference_max=float(out["Target"].max()),
              supported_models=sorted(SUPPORTED_MODELS), supported_leads=list(SUPPORTED_LEADS),
              model_in_supported_set=True)
    return out, diag

def main():
    parquet=os.path.join(ROOT,"data","processed","UPDATE-09","training_dataset.parquet")
    test=os.path.join(ROOT,"UPDATE-09","Test.csv")
    out, diag = infer(parquet, test, selected_model="B2")
    with open(os.path.join(M,"inference_diagnostics.json"),"w") as f: json.dump(diag,f,indent=2)
    print(json.dumps(diag,indent=2))
    print(f"inference smoke: rows={len(out)} mean={out['Target'].mean():.3f} min={out['Target'].min():.3f} max={out['Target'].max():.3f}")
    print("NOT saved to submissions/")

if __name__=="__main__":
    main()
