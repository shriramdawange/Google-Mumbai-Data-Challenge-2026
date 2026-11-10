import os, hashlib, json, pandas as pd
root = "UPDATE-10"
out = []
for fn in sorted(os.listdir(root)):
    p = os.path.join(root, fn)
    if not os.path.isfile(p):
        continue
    sz = os.path.getsize(p); h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    row = dict(filename=fn, size_bytes=int(sz), sha256=h)
    if fn.endswith(".csv"):
        df = pd.read_csv(p)
        row["shape"] = list(df.shape); row["columns"] = list(df.columns)
        row["nulls"] = int(df.isna().sum().sum()); row["dups"] = int(df.duplicated().sum())
    out.append(row)
with open("reports/update10_inventory_raw.json", "w") as f:
    json.dump(out, f, indent=2)
for r in out:
    print(r)
