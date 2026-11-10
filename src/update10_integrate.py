"""MISSION 03 Stage B/C: UPDATE-10 integration, quality, comparison, forecast-horizon audit.
Writes data/processed/UPDATE-10/* and reports/update10/*.
Note: UPDATE-10/Train.csv was absent at audit time; interim training source = UPDATE-09 parquet.
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
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def main():
    inv = []
    for fn in sorted(os.listdir(NEW)):
        p = os.path.join(NEW, fn)
        if not os.path.isfile(p):
            continue
        sz = os.path.getsize(p); h = sha(p)
        row = dict(filename=fn, size_bytes=int(sz), sha256=h, file_type="archive" if fn.endswith(".zip") else ("ipynb" if fn.endswith(".ipynb") else ("txt" if fn.endswith(".txt") else "csv")))
        if fn.endswith(".csv"):
            df = pd.read_csv(p)
            row["shape"] = list(df.shape); row["columns"] = list(df.columns); row["nulls"] = int(df.isna().sum().sum()); row["dups"] = int(df.duplicated().sum())
        inv.append(row)
    with open(os.path.join(RPT, "INTEGRATION.md"), "w") as f:
        f.write("# UPDATE-10 Integration Report (Mission 03)\n\nFiles inventoried:\n")
        for r in inv:
            f.write(f"- {r['filename']}: size={r['size_bytes']} sha256={r['sha256']}\n")
        f.write("\nTRAIN_PRESENT=" + str(os.path.exists(os.path.join(NEW, "Train.csv"))) + "\n")
    with open(os.path.join(RPT, "INVENTORY.md"), "w") as f:
        f.write("# UPDATE-10 Inventory\n\n")
        for r in inv:
            f.write(f"- {r['filename']}: size={r['size_bytes']} sha256={r['sha256']} shape={r.get('shape')} columns={r.get('columns')}\n")
    # Comparison vs UPDATE-09 (filename aliases handled)
    pairs = [
        ("data/Test.csv", "UPDATE-10/Test.csv"),
        ("data/SampleSubmission.csv", "UPDATE-10/SampleSubmission.csv"),
        ("data/Stations.csv", "UPDATE-10/Stations (1).csv"),
        ("data/DataDictionary.txt", "UPDATE-10/DataDictionary.txt"),
        ("data/AdditionalHistory.zip", "UPDATE-10/AdditionalHistory.zip"),
        ("data/Mumbai_Air_Quality_Starter.ipynb", "UPDATE-10/Mumbai_Air_Quality_Starter.ipynb"),
    ]
    cmp_rows = {}
    for po, pn in pairs:
        po = po.replace("data/", OLD + "/").replace("UPDATE-10/", NEW + "/")
        po = os.path.join(ROOT, os.path.relpath(po, ROOT)) if not po.startswith("data/") else po
        po = OLD if po == OLD else os.path.join(ROOT, po)
        pn = NEW if pn == NEW else os.path.join(ROOT, pn)
        a = sha(po); b = sha(pn)
        key = os.path.basename(pn)
        cmp_rows[key] = dict(old=a, new=b, identical=(a == b))
    with open(os.path.join(RPT, "UPDATE09_VS_UPDATE10.md"), "w") as f:
        f.write("# UPDATE-09 vs UPDATE-10 Comparison\n\n")
        for k, v in cmp_rows.items():
            f.write(f"{k}: old={v['old']} new={v['new']} identical={v['identical']}\n")
    tr_old = pd.read_csv(os.path.join(OLD, "Train.csv"), parse_dates=["timestamp_ist", "available_at"])
    tr_new_path = os.path.join(NEW, "Train.csv")
    tr_new = pd.read_csv(tr_new_path, parse_dates=["timestamp_ist", "available_at"]) if os.path.exists(tr_new_path) else None
    comp = dict(train_old_shape=list(tr_old.shape), train_new_shape=(list(tr_new.shape) if tr_new is not None else None),
                train_old_max=str(tr_old.timestamp_ist.max()), train_new_max=(str(tr_new.timestamp_ist.max()) if tr_new is not None else None))
    with open(os.path.join(RPT, "UPDATE09_VS_UPDATE10.md"), "a") as f:
        f.write(f"\nTRAIN: old_shape={comp['train_old_shape']} new_shape={comp['train_new_shape']} old_max={comp['train_old_max']} new_max={comp['train_new_max']}\n")
    qa = dict(test_shape=list(pd.read_csv(os.path.join(NEW, "Test.csv")).shape), sample_shape=list(pd.read_csv(os.path.join(NEW, "SampleSubmission.csv")).shape),
              stations_shape=list(pd.read_csv(os.path.join(NEW, "Stations (1).csv")).shape), train_present=tr_new is not None)
    with open(os.path.join(RPT, "DATA_QUALITY.md"), "w") as f:
        f.write("# UPDATE-10 Data Quality Audit\n\n")
        f.write(f"Test shape: {qa['test_shape']}\nSample shape: {qa['sample_shape']}\nStations shape: {qa['stations_shape']}\nTrain present: {qa['train_present']}\n")
    import shutil
    interim = os.path.join(OUT, "training_dataset.parquet")
    shutil.copy(os.path.join(ROOT, "data", "processed", "UPDATE-09", "training_dataset.parquet"), interim)
    man = dict(source_release="UPDATE-10-final-release-interim", interim_source=os.path.join(ROOT, "data", "processed", "UPDATE-09", "training_dataset.parquet"),
               note="UPDATE-10/Train.csv was absent at Mission 03 audit; interim training dataset used from UPDATE-09 parquet pending final Train.csv delivery.",
               train_rows=int(48133), final_release_claim="Train contains 48,400 observed station-hours (incl. 7-8 Nov); reconcile when Train.csv delivered.",
               cutoff=str(tr_old.timestamp_ist.max()))
    with open(os.path.join(OUT, "dataset_manifest.json"), "w") as f: json.dump(man, f, indent=2)
    with open(os.path.join(OUT, "comparison_report.json"), "w") as f: json.dump(cmp_rows, f, indent=2)
    with open(os.path.join(OUT, "data_quality_report.json"), "w") as f: json.dump(qa, f, indent=2)
    with open(os.path.join(RPT, "MANIFEST.json"), "w") as f:
        json.dump(dict(update10_files=inv, interim_training=interim, sha256_interim=sha(interim)), f, indent=2)
    print("DONE_UPDATE10_STAGE_B", flush=True)

if __name__ == "__main__":
    main()
