"""M02.3.1 audit: fix metrics, rebuild B2 bootstrap, verify/configure inference.
No new large experiment; reads validation_oof.csv, writes new files.
"""
import os, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M=os.path.join(ROOT,"reports","m023")
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda:f.read(1<<20),b""): h.update(ch)
    return h.hexdigest()
oof=pd.read_csv(os.path.join(M,"validation_oof.csv"))
models=["B2","B3","B4","pred_rf"]
# Task1: overall + by-origin + by-station
rows=[]
for c in models:
    rows.append(dict(model=c,rmse=rmse(oof.y,oof[c]),mae=float(np.mean(np.abs(oof.y-oof[c]))),
                     bias=float((oof[c]-oof.y).mean()),n=int(len(oof))))
pd.DataFrame(rows).to_csv(os.path.join(M,"metrics_overall.csv"),index=False)
bo=[dict(origin=o,model=c,rmse=rmse(g.y,g[c]),mae=float(np.mean(np.abs(g.y-g[c]))),bias=float((g[c]-g.y).mean()),n=int(len(g)))
    for o,g in oof.groupby("origin") for c in models]
pd.DataFrame(bo).to_csv(os.path.join(M,"metrics_by_origin.csv"),index=False)
bs=[dict(location_id=int(k),model=c,rmse=rmse(g.y,g[c]),mae=float(np.mean(np.abs(g.y-g[c]))),bias=float((g[c]-g.y).mean()),n=int(len(g)))
    for k,g in oof.groupby("location_id") for c in models]
pd.DataFrame(bs).to_csv(os.path.join(M,"metrics_by_station.csv"),index=False)
print(pd.DataFrame(rows).to_string(),flush=True)
# Task2: paired bootstrap B2-centric, 10k, origin-block
oof["se_B2"]=(oof.B2-oof.y)**2; oof["se_B3"]=(oof.B3-oof.y)**2; oof["se_B4"]=(oof.B4-oof.y)**2; oof["se_RF"]=(oof.pred_rf-oof.y)**2
rng=np.random.RandomState(0); origins=oof.origin.unique()
pairs=[("B2","B3","se_B2","se_B3"),("B2","B4","se_B2","se_B4"),("B2","RF","se_B2","se_RF"),("B3","RF","se_B3","se_RF")]
B=10000
out={}
sets={k:[] for k in ["B2_vs_B3","B2_vs_B4","B2_vs_RF","B3_vs_RF"]}
rm={c:[] for c in ["B2","B3","B4","RF"]}
for _ in range(B):
    samp=list(rng.choice(origins,size=len(origins),replace=True))
    rr=pd.concat([oof[oof.origin==o] for o in samp],ignore_index=True)
    for k in rm: rm[k].append(float(np.sqrt(rr[f"se_{k}"].mean())))
    for (a,b,sa,sb),key in zip(pairs,["B2_vs_B3","B2_vs_B4","B2_vs_RF","B3_vs_RF"]):
        sets[key].append(float(np.sqrt(rr[sb].mean())-np.sqrt(rr[sa].mean())))  # positive = A better
out["n_boot"]=B; out["n_origins"]=int(len(origins)); out["n_rows"]=int(len(oof))
out["comparisons"]={}
col={"B2":"B2","B3":"B3","B4":"B4","RF":"pred_rf"}
for (a,b,sa,sb),key in zip(pairs,["B2_vs_B3","B2_vs_B4","B2_vs_RF","B3_vs_RF"]):
    arr=np.array(sets[key])
    out["comparisons"][key]=dict(model_A=a,model_B=b,convention="sqrtMSE(B)-sqrtMSE(A); positive => A better",
        observed_diff=float(rmse(oof.y,oof[col[b]])-rmse(oof.y,oof[col[a]])),
        ci=[float(np.percentile(arr,2.5)),float(np.percentile(arr,97.5))],
        p_A_better=float((arr>0).mean()))
out["model_rmse_ci"]={c:[float(np.percentile(rm[c],2.5)),float(np.percentile(rm[c],97.5))] for c in rm}
out["model_rmse"]={c:rmse(oof.y,oof["B2" if c=="B2" else ("B3" if c=="B3" else ("B4" if c=="B4" else "pred_rf"))]) for c in rm}
with open(os.path.join(M,"bootstrap.json"),"w") as f: json.dump(out,f,indent=2)
print(json.dumps(out["comparisons"],indent=1),flush=True)
# difference explainer: bootstrap script diffs use (B - A); report both
print("DONE_AUDIT_STAGES",flush=True)
