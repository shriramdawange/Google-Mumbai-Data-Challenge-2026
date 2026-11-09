"""UPDATE-09 integration: inventory, comparison, quality audit, versioned output.
Read-only on UPDATE-09/ and data/. Writes to data/processed/UPDATE-09/ + reports/update09/.
"""
import os, json, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD=os.path.join(ROOT,"data"); NEW=os.path.join(ROOT,"UPDATE-09")
OUT=os.path.join(ROOT,"data","processed","UPDATE-09"); RPT=os.path.join(ROOT,"reports","update09")
os.makedirs(OUT,exist_ok=True); os.makedirs(RPT,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()

def main():
    # Phase 1 inventory
    rows=[]
    for fn in sorted(os.listdir(NEW)):
        p=os.path.join(NEW,fn)
        if not os.path.isfile(p): continue
        size=os.path.getsize(p); s=sha(p)
        r=dict(filename=fn,size_bytes=size,sha256=s)
        if fn.endswith(".csv"):
            try:
                df=pd.read_csv(p, nrows=0)  # header only
                with open(p,"rb") as f: nr=sum(1 for _ in f)-1
                r.update(rows=nr,cols=len(df.columns),schema=list(df.columns))
            except Exception as e: r.update(error=str(e))
        elif fn.endswith(".txt"):
            r.update(schema=open(p).read()[:500])
        else:
            r.update(schema="archive/binary (not parsed)")
        rows.append(r)
    inv=pd.DataFrame(rows)
    inv.to_csv(os.path.join(RPT,"update09_inventory.csv"),index=False)
    print(inv.to_string(),flush=True)
    # Hash-check Test/Sample/Stations vs old (keys)
    for fn in ["Test.csv","SampleSubmission.csv","Stations.csv","DataDictionary.txt","Mumbai_Air_Quality_Starter.ipynb","AdditionalHistory.zip"]:
        so=sha(os.path.join(OLD,fn)); sn=sha(os.path.join(NEW,fn))
        print(f"{fn} old={so[:8]} new={sn[:8]} identical={so==sn}",flush=True)
    # Phase 2 comparison (Train)
    old=pd.read_csv(os.path.join(OLD,"Train.csv"),parse_dates=["timestamp_ist","available_at"])
    new=pd.read_csv(os.path.join(NEW,"Train.csv"),parse_dates=["timestamp_ist","available_at"])
    for nm,df in [("old",old),("new",new)]:
        print(nm, df.shape, df.timestamp_ist.min(), df.timestamp_ist.max(), df.pm25.describe().to_dict(),flush=True)
    key=["location_id","timestamp_ist"]
    old_keys=set(map(tuple, old[key].values.astype(str))); new_keys=set(map(tuple, new[key].values.astype(str)))
    added=new_keys-old_keys; removed=old_keys-new_keys
    print(f"added={len(added)} removed={len(removed)}",flush=True)
    # classify added timestamps
    added_df=new[new.apply(lambda r: (str(r.location_id),str(r.timestamp_ist)) in added,axis=1)] if added else new.iloc[0:0]
    print(added_df.groupby(["location_id"]).size() if len(added_df) else "no added",flush=True)
    if len(added_df): print(added_df.timestamp_ist.min(), added_df.timestamp_ist.max(),flush=True)
    # revised: same key, different pm25
    m=old.merge(new,on=key,suffixes=("_old","_new"))
    rev=m[np.abs(m.pm25_old-m.pm25_new)>1e-9] if len(m) else m
    print(f"common keys={len(m)} revised pm25={len(rev)}",flush=True)
    comp=dict(old_shape=list(old.shape),new_shape=list(new.shape),
              old_range=[str(old.timestamp_ist.min()),str(old.timestamp_ist.max())],
              new_range=[str(new.timestamp_ist.min()),str(new.timestamp_ist.max())],
              added=len(added),removed=len(removed),common=len(m),revised=len(rev),
              per_station_added={int(k):int(v) for k,v in added_df.groupby("location_id").size().items()} if len(added_df) else {},
              pm25_old=dict(mean=float(old.pm25.mean()),median=float(old.pm25.median()),std=float(old.pm25.std()),max=float(old.pm25.max()),n1000=int((old.pm25>1000).sum())),
              pm25_new=dict(mean=float(new.pm25.mean()),median=float(new.pm25.median()),std=float(new.pm25.std()),max=float(new.pm25.max()),n1000=int((new.pm25>1000).sum())),
              columns_old=list(old.columns),columns_new=list(new.columns),
              added_ts_range=[str(added_df.timestamp_ist.min()),str(added_df.timestamp_ist.max())] if len(added_df) else [])
    with open(os.path.join(RPT,"comparison_report.json"),"w") as f: json.dump(comp,f,indent=2,default=str)
    # Phase 3 quality audit (new)
    qa={}
    qa["nulls"]=new.isna().sum().to_dict()
    qa["dup_keys"]=int(new.duplicated(key).sum())
    qa["bad_ts"]=int((pd.to_datetime(new.timestamp_ist,errors="coerce").isna()).sum())
    qa["stations"]=sorted(new.location_id.unique().tolist())
    qa["pm25_min"]=float(new.pm25.min()); qa["pm25_max"]=float(new.pm25.max()); qa["pm25_negative"]=int((new.pm25<0).sum())
    qa["flag_true"]=int(new.flag_high_1000.sum()) if "flag_high_1000" in new.columns else None
    qa["sorted_chrono"]=bool(new.sort_values(key).reset_index(drop=True)[key].equals(new[key].reset_index() if False else new[key]))
    # chronological per station
    chrono=True
    for _,g in new.groupby("location_id"):
        if not g.timestamp_ist.is_monotonic_increasing: chrono=False
    qa["mono_per_station"]=chrono
    with open(os.path.join(RPT,"data_quality_report.json"),"w") as f: json.dump(qa,f,indent=2,default=str)
    print(qa,flush=True)
    # Phase 4 integration: versioned training dataset = union with documented rule:
    #   - keep all old rows + added rows; on key conflict use NEW value (newer release wins) and log.
    #   - flag column: use new file's values.
    new_sorted=new.sort_values(key)
    old_sorted=old.sort_values(key)
    combined=new_sorted  # new file is a superset release (contains old + added); verified by sets above.
    # If removed keys existed, keep them from old (documented): union
    if removed:
        extra=old[old.apply(lambda r: (str(r.location_id),str(r.timestamp_ist)) in removed,axis=1)]
        combined=pd.concat([new_sorted,extra],ignore_index=True).sort_values(key)
    else:
        combined=new_sorted.reset_index(drop=True)
    combined["source_release"]="UPDATE-09"
    # dedupe safety
    assert combined.duplicated(key).sum()==0, "duplicate keys after merge"
    pq=os.path.join(OUT,"training_dataset.parquet")
    combined.to_parquet(pq, index=False)
    print(f"saved {pq} {combined.shape}",flush=True)
    manifest=dict(lineage=["data/Train.csv (Nov-2 snapshot)","UPDATE-09/Train.csv (Nov-9 release)"],
                  union_rule="new wins on conflict, old kept if key missing in new; pm25>1000 kept raw, cleaning deferred to pipeline (pm25_clean rule unchanged)",
                  records_before=int(len(old)),records_new=int(len(new)),records_after=int(len(combined)),
                  cutoff_old=str(old.timestamp_ist.max()),cutoff_new=str(new.timestamp_ist.max()),
                  output_sha256=sha(pq), output_bytes=os.path.getsize(pq))
    with open(os.path.join(OUT,"dataset_manifest.json"),"w") as f: json.dump(manifest,f,indent=2,default=str)
    with open(os.path.join(OUT,"data_quality_report.json"),"w") as f: json.dump(qa,f,indent=2,default=str)
    with open(os.path.join(OUT,"comparison_report.json"),"w") as f: json.dump(comp,f,indent=2,default=str)
    # Phase 5 feature validation: run mission01 feature builder against NEW data (train vs inference consistency)
    import sys; sys.path.insert(0,os.path.join(ROOT,"src"))
    from mission01 import load as _l, build_series as _b, FEATURES
    new["_dummy"]=1
    # reuse build_series expecting pm25_clean; construct cleaned copy without touching raw
    import shutil
    tmp=new.copy(); tmp["pm25_clean"]=tmp["pm25"].astype(float)
    tmp.loc[tmp["flag_high_1000"]==True,"pm25_clean"]=np.nan; tmp.loc[tmp["pm25"]>1000,"pm25_clean"]=np.nan
    _,series,city,_c=_b(tmp)
    print(f"feature smoke: series={len(series)} FULL feats={len(FEATURES)} sample lag1 non-null={float(series[next(iter(series))]['lag1'].notna().mean()):.3f}",flush=True)
    with open(os.path.join(RPT,"feature_validation.json"),"w") as f:
        json.dump(dict(full_features=len(FEATURES),features=FEATURES,series_stations=len(series),
                       test_ids_unchanged=bool((pd.read_csv(os.path.join(NEW,"Test.csv")).ID.values==pd.read_csv(os.path.join(OLD,"Test.csv")).ID.values).all())),
                  f,indent=2)
    # Phase 6 tests: Test/Sample unchanged hashes already printed; verify counts
    te_old=pd.read_csv(os.path.join(OLD,"Test.csv")); te_new=pd.read_csv(os.path.join(NEW,"Test.csv"))
    ss_old=pd.read_csv(os.path.join(OLD,"SampleSubmission.csv")); ss_new=pd.read_csv(os.path.join(NEW,"SampleSubmission.csv"))
    tests=dict(test_rows_old=len(te_old),test_rows_new=len(te_new),test_order_same=bool((te_old.ID.values==te_new.ID.values).all()),
               sample_rows_old=len(ss_old),sample_rows_new=len(ss_new),
               output_rows=int(len(combined)),dup_keys=int(combined.duplicated(key).sum()))
    with open(os.path.join(RPT,"test_results.json"),"w") as f: json.dump(tests,f,indent=2)
    print(tests,flush=True)
    # Phase 7 readiness: new cutoff + auto private leads
    train_max=new.timestamp_ist.max(); tz=train_max.tz
    for lab,(ps,pe) in {"private":("2026-11-11 01:00","2026-11-13 00:00"),"public":("2026-11-05 01:00","2026-11-07 00:00")}.items():
        a=pd.Timestamp(ps,tz=tz); b=pd.Timestamp(pe,tz=tz)
        leads=[int((a+pd.Timedelta(hours=i)-train_max).total_seconds()//3600) for i in range(48)]
        print(f"{lab} leads from new cutoff: {leads[0]}..{leads[-1]}",flush=True)
    with open(os.path.join(RPT,"training_readiness.json"),"w") as f:
        json.dump(dict(new_cutoff=str(train_max),old_cutoff=str(old.timestamp_ist.max()),output=str(pq),
                       smoke="feature builder OK (see feature_validation.json)"),f,indent=2)
    print("DONE_UPDATE09",flush=True)
if __name__=="__main__": main()
