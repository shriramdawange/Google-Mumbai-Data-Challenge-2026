import os, json, hashlib, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()
files = {
    "submissions/baseline_v0.csv": "a5e709dce45eaf4cb199765cd5f5b009e43e2465c4a65724b9678c79a1725360",
    "submissions/baseline_v0_hgb.csv": "eaf4dfc0af17ee0a378637b48aec30381813db492de26348f543ae33a9eecbdc",
    "reports/m023/validation_oof.csv": "b96201eba625c79a36e957b13bceb0f225ceabec291c0e2de067363bdd89af87",
    "data/processed/UPDATE-09/training_dataset.parquet": "5a50da1a0065af35674d5fd2025747a3e25f572418f6e0d1931b7eede4b054c5",
    "UPDATE-10/Test.csv": "e7b9cc8859beb939276c958393e148adae55229c34f99dcb2555fe4ac13b28db",
    "UPDATE-10/SampleSubmission.csv": "ee4682ce0e546c1b62beb4a08414a5ddc831fa962cff30e8a823d6b6b0e0c72f",
    "UPDATE-10/Stations (1).csv": "383269fa04abf8f2eb49e5ec0bb5a77c2df11acb6aca996b7edf269a109ee9cf",
    "UPDATE-10/AdditionalHistory.zip": "3233061ca4ac2af39e9f6e80e21d9261b1b8d2fd3b037e5d034bcec059a7e80e",
    "UPDATE-10/DataDictionary.txt": "8786b98f78784e1b53a27cb23c8df64ab92ef76d3be2885098b00c0830a8c985",
    "UPDATE-10/Mumbai_Air_Quality_Starter.ipynb": "e824643f479fe22a4fe67d43c6b73d815b459394584d6d1ff2f0f6cbd797b8b1",
}
res = {}
for f, exp in files.items():
    got = sha(os.path.join(ROOT, f))
    res[f] = dict(expected=exp, actual=got, unchanged=(got == exp))
# M01-M02.2 protected artifacts: just confirm presence in git ls-files (no overwrite expected)
tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
res["git_tracked_protected"] = [t for t in ["submissions/baseline_v0.csv", "submissions/baseline_v0_hgb.csv", "reports/m023/validation_oof.csv", "data/processed/UPDATE-09/training_dataset.parquet"] if t in tracked]
out = {"protected_hashes": res, "all_unchanged": all(v["unchanged"] for k, v in res.items() if isinstance(v, dict) and "unchanged" in v)}
with open(os.path.join(ROOT, "reports/update10/protected_audit.json"), "w") as f: json.dump(out, f, indent=2)
for k, v in res.items():
    if isinstance(v, dict): print(k, v["unchanged"], v["actual"])
    else: print(k, v)
print("ALL_PROTECTED_UNCHANGED=", out["all_unchanged"])
