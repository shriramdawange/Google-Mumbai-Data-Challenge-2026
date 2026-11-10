"""MISSION 03 / 03.1 Stage B: UPDATE-10 integration from the ACTUAL final Train.csv.

Consumes UPDATE-10/Train.csv (final release) as the primary training source.
Never silently copies UPDATE-09 data: union rule = new (UPDATE-10) wins on key
conflict, old (data/Train.csv snapshot) kept only if key missing in new; cleaning
(pm25>1000 -> NaN via pm25_clean policy) applied at pipeline layer, never imputation.
Writes data/processed/UPDATE-10/* + reports/update10/*.
"""
import os, json, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "data")
NEW = os.path.join(ROOT, "UPDATE-10")
OUT = os.path.join(ROOT, "data", "processed", "UPDATE-10")
RPT = os.path.join(ROOT, "reports", "update10")
os.makedirs(OUT, exist_ok=True); os.makedirs(RPT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

def main():
    # ---------- inventory ----------
    inv = []
    for fn in sorted(os.listdir(NEW)):
        p = os.path.join(NEW, fn)
        if not os.path.isfile(p): continue
        size = os.path.getsize(p); s = sha(p)
        r = dict(filename=fn, size_bytes=int(size), sha256=s,
                 file_type="archive" if fn.endswith(".zip") else ("ipynb" if fn.endswith(".ipynb") else ("txt" if fn.endswith(".txt") else "csv")))
        if fn.endswith(".csv"):
            df = pd.read_csv(p, nrows=0)
            with open(p, "rb") as f: nr = sum(1 for _ in f) - 1
            r.update(rows=int(nr), cols=int(len(df.columns)), schema=list(df.columns))
        inv.append(r)
    # ---------- comparison: Test / Sample / Stations vs UPDATE-09-era data/ ----------
    cmp_rows = {}
    cmp_files = ["Test.csv", "SampleSubmission.csv", "DataDictionary.txt", "AdditionalHistory.zip", "Mumbai_Air_Quality_Starter.ipynb"]
    if os.path.exists(os.path.join(NEW, "Stations (1).csv")):
        cmp_files.append("Stations (1).csv")
    elif os.path.exists(os.path.join(OLD, "Stations.csv")):
        cmp_files.append("Stations.csv")
    for fn in cmp_files:
        pn = os.path.join(NEW, fn)
        po = os.path.join(OLD, fn)
        cmp_rows[fn] = dict(old=sha(po) if os.path.exists(po) else None, new=sha(pn) if os.path.exists(pn) else None,
                            identical=os.path.exists(po) and os.path.exists(pn) and sha(po) == sha(pn))
    # ---------- Train comparison + union ----------
    old = pd.read_csv(os.path.join(OLD, "Train.csv"), parse_dates=["timestamp_ist", "available_at"]) if os.path.exists(os.path.join(OLD, "Train.csv")) else None
    new = pd.read_csv(os.path.join(NEW, "Train.csv"), parse_dates=["timestamp_ist", "available_at"])
    key = ["location_id", "timestamp_ist"]
    old_keys = set() if old is None else set(map(tuple, old[key].values.astype(str)))
    new_keys = set(map(tuple, new[key].values.astype(str)))
    added = new_keys - old_keys; removed = old_keys - new_keys
    new_sorted = new.sort_values(key); combined = new_sorted.copy()
    if removed and old is not None:
        extra = old[old.apply(lambda r: (str(r.location_id), str(r.timestamp_ist)) in removed, axis=1)]
        combined = pd.concat([new_sorted, extra], ignore_index=True).sort_values(key)
    combined["source_release"] = "UPDATE-10"
    assert combined.duplicated(key).sum() == 0, "duplicate keys after union"
    pq = os.path.join(OUT, "training_dataset.parquet")
    combined.to_parquet(pq, index=False)
    # ---------- quality audit (from actual new Train) ----------
    qa = dict(train_rows=int(len(new)), train_sha256=sha(os.path.join(NEW, "Train.csv")), train_cutoff=str(new["timestamp_ist"].max()),
              nulls={c: int(new[c].isna().sum()) for c in new.columns}, dup_keys=int(new.duplicated(key).sum()),
              stations=sorted(new["location_id"].unique().tolist()), pm25_min=float(new["pm25"].min()),
              pm25_max=float(new["pm25"].max()), pm25_negative=int((new["pm25"] < 0).sum()),
              flag_high_1000=int(new["flag_high_1000"].sum()) if "flag_high_1000" in new.columns else None,
              mono_per_station=all(g["timestamp_ist"].is_monotonic_increasing for _, g in new.groupby("location_id")))
    comp = dict(train_old_shape=list(old.shape) if old is not None else None, train_new_shape=list(new.shape),
                train_old_max=str(old["timestamp_ist"].max()) if old is not None else None, train_new_max=str(new["timestamp_ist"].max()),
                added=int(len(added)), removed=int(len(removed)), old_keys=len(old_keys) if old is not None else 0, new_keys=len(new_keys),
                old_rows=int(len(old)) if old is not None else 0, new_rows=int(len(new)),
                pm25_old=dict(mean=float(old["pm25"].mean()), max=float(old["pm25"].max())) if old is not None else None,
                pm25_new=dict(mean=float(new["pm25"].mean()), max=float(new["pm25"].max())))
    # ---------- reports ----------
    with open(os.path.join(RPT, "INTEGRATION.md"), "w") as f:
        f.write("# UPDATE-10 Integration Report (Mission 03.1 RESUME)\n\n")
        f.write("Training source: ACTUAL `UPDATE-10/Train.csv` (final release, primary).\n")
        f.write(f"Train rows (actual): {len(new)} | cutoff: {new['timestamp_ist'].max()} | sha256: {sha(os.path.join(NEW,'Train.csv'))}\n")
        f.write(f"Train old snapshot rows: {len(old) if old is not None else 'n/a'} | old cutoff: {old['timestamp_ist'].max() if old is not None else None}\n")
        f.write(f"Added keys: {len(added)} | Removed keys: {len(removed)}\n")
        f.write(f"Output: {pq} shape={list(combined.shape)}\n")
    with open(os.path.join(RPT, "DATA_QUALITY.md"), "w") as f:
        f.write("# UPDATE-10 Data Quality Audit (Mission 03.1 RESUME)\n\n")
        f.write(json.dumps(qa, indent=2, default=str))
    with open(os.path.join(RPT, "UPDATE09_VS_UPDATE10.md"), "w") as f:
        f.write("# UPDATE-09 vs UPDATE-10 Comparison (Mission 03.1 RESUME)\n\n")
        f.write(json.dumps(cmp_rows, indent=2, default=str))
        f.write(f"\nTRAIN: old={comp['train_old_shape']} new={comp['train_new_shape']} old_max={comp['train_old_max']} new_max={comp['train_new_max']} added={comp['added']} removed={comp['removed']}\n")
    with open(os.path.join(RPT, "MANIFEST.json"), "w") as f:
        json.dump(dict(update10_train_actual=dict(path=os.path.join(NEW, "Train.csv"), rows=int(len(new)), sha256=sha(os.path.join(NEW, "Train.csv")), cutoff=str(new["timestamp_ist"].max()), source_release="UPDATE-10"),
                        output_training_dataset=pq, output_sha256=sha(pq), output_rows=int(len(combined)), quality=qa, comparison=comp), f, indent=2, default=str)
    print(f"DONE_UPDATE10_INTEGRATE_ACTUAL rows={len(new)} cutoff={new['timestamp_ist'].max()} added={len(added)} removed={len(removed)}", flush=True)
    print("DONE_UPDATE10_STAGE_B_ACTUAL", flush=True)

if __name__ == "__main__":
    main()
