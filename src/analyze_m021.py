import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error
def R(a,b): return float(np.sqrt(mean_squared_error(a,b)))
m="reports/m021/"
o=pd.read_csv(m+"contiguous_oof.csv")
cols=["pred_general_rf","pred_general_rf_w","pred_specialised_rf","pred_B0_hour_profile","pred_B1_recent_means","pred_B2_level_x_diurnal","pred_B3_seasonal_adj"]
for c in cols: print(c, round(R(o.y,o[c]),3), "bias", round(float((o[c]-o.y).mean()),3), "n", len(o))
m2=o.pred_persist.notna()
print("persist", round(R(o.loc[m2,"y"],o.loc[m2,"pred_persist"]),3), "n", int(m2.sum()))
print("y>100 rows", int((o.y>100).sum()), "y>300", int((o.y>300).sum()), "even missing window check: Jan fold skipped")
print(o.groupby("origin").apply(lambda d: R(d.y,d.pred_general_rf), include_groups=False).to_string())
# calibration: regress y on pred
for c in ["pred_general_rf","pred_specialised_rf","pred_B3_seasonal_adj"]:
    X=o[c].values.reshape(-1,1); A=np.hstack([X,np.ones((len(X),1))]); coef,_ ,_,_=np.linalg.lstsq(A,o.y.values,rcond=None)
    print(c, "y = a*pred+b a=",round(float(coef[0]),3),"b=",round(float(coef[1]),3))
print(o.groupby("location_id").apply(lambda d: pd.Series({"gen":R(d.y,d.pred_general_rf),"b3":R(d.y,d.pred_B3_seasonal_adj),"bias_gen":float((d.pred_general_rf-d.y).mean())}), include_groups=False).to_string())
# residual
r=pd.read_csv(m+"residual_oof.csv")
for c in ["pred_ridge_resid","pred_hgb_resid","pred_rf_resid"]: print(c, round(R(r.y,r[c]),3))
# lookback
lb=pd.read_csv(m+"lookback2_oof.csv")
print(lb.groupby("lookback").apply(lambda d: R(d.y,d.pred), include_groups=False).to_string())
# ablation: compare to full on same origins? full pooled on ablation origins from contiguous
ab=pd.read_csv(m+"ablation_oof.csv")
print(ab.groupby("ablation").apply(lambda d: R(d.y,d.pred), include_groups=False).to_string())
# full on same origins for reference
ab_origins=sorted(ab.origin.unique())
full_sub=o[o.origin.isin(ab_origins)]
print("full on ablation origins", round(R(full_sub.y,full_sub.pred_general_rf),3), len(full_sub))
# extreme
e=pd.read_csv(m+"extreme_oof.csv")
for c in ["pred_excl_mean","pred_excl_median","pred_robust_log","pred_raw"]: print(c, round(R(e.y,e[c]),3))
# update
u=pd.read_csv(m+"update_sim2.csv")
print(u.groupby(["T","mode"]).apply(lambda d: R(d.y,d.pred), include_groups=False).to_string())
# variance decomposition: pred mean by hour and by lead (general RF)
print(o.groupby(pd.to_datetime(o.target_t).dt.hour).pred_general_rf.mean().round(2).to_string())
print(o.groupby(pd.cut(o.lead,[193,205,217,229,241])).pred_general_rf.mean().round(2).to_string())
