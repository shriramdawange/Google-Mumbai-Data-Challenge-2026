"""MISSION 02.3 stage2: inference interface (UPDATE-10-ready) + smoke test. No submission files."""
import os, pandas as pd, numpy as np
import sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,"src"))
from mission02_3 import load_adapter, build_series, FEATS
from sklearn.ensemble import RandomForestRegressor

def infer(parquet_path, test_path):
    """Rebuild B2 features + RF-free inference stub? Here: B2 inference smoke (RF requires retrain; interface returns B2 preds + validates format)."""
    df=pd.read_parquet(parquet_path)
    df["timestamp_ist"]=pd.to_datetime(df["timestamp_ist"])
    tmp=df.drop(columns=["source_release"],errors="ignore").copy()
    tmp["pm25_clean"]=tmp["pm25"].astype(float)
    tmp.loc[tmp["flag_high_1000"]==True,"pm25_clean"]=np.nan; tmp.loc[tmp["pm25"]>1000,"pm25_clean"]=np.nan
    train_max=tmp.timestamp_ist.max()
    te=pd.read_csv(test_path,parse_dates=["target_start_ist","target_end_ist"])
    assert len(te)==2592 and te.ID.nunique()==2592
    # B2: per station 7d level * 90d diurnal norm at train_max
    from mission02_2 import window_stats, hour_profile
    preds={}
    for loc,g in tmp.groupby("location_id"):
        s=g.set_index("timestamp_ist")["pm25_clean"].sort_index()
        lv=window_stats(s,train_max,7)["mean"]
        prof,_=hour_profile(s,train_max,90); norm=prof/prof.mean()
        preds[int(loc)]=(lv,norm)
    rows=[]
    for _,r in te.iterrows():
        loc=int(r.location_id); h=int(r.target_end_ist.hour); lv,norm=preds[loc]
        rows.append(dict(ID=r.ID, Target=float(max(0,lv*float(norm.get(h,1.0))))))
    sub=pd.DataFrame(rows)
    assert (sub.ID.values==te.ID.values).all()
    assert (sub.Target.notna() & np.isfinite(sub.Target) & (sub.Target>=0)).all()
    return sub

if __name__=="__main__":
    sub=infer(os.path.join(ROOT,"data","processed","UPDATE-09","training_dataset.parquet"),
              os.path.join(ROOT,"UPDATE-09","Test.csv"))
    print(f"inference smoke OK: {sub.shape} mean={sub.Target.mean():.3f} min={sub.Target.min():.3f} max={sub.Target.max():.3f}")
    print("NOT saved to submissions/ (interface test only)")
