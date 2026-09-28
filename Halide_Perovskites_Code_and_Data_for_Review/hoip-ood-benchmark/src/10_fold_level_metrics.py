"""
Stage 10 - per-fold metrics for the extrapolative protocols
(Table S6, Section 3.2).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from features import build_features, assemble
from evaluation import make_splits
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
df=pd.read_csv(DATA / "hoip_master.csv"); df,B=build_features(df); splits=make_splits(df)
def lgbm():
    return Pipeline([("imp",SimpleImputer(strategy="median")),
      ("est",lgb.LGBMRegressor(n_estimators=250,learning_rate=0.07,num_leaves=31,
       min_child_samples=8,subsample=0.85,subsample_freq=1,colsample_bytree=0.8,
       reg_lambda=1.0,n_jobs=1,random_state=0,verbose=-1))])
rows=[]
X0=df[assemble(["COMP","LLM","PHYS","STRUCT"],B)].values
for mode in ["direct","delta"]:
    if mode=="direct": Xu,y=X0,df.Eg_HSE.values
    else: Xu,y=np.column_stack([X0,df.Eg_GGA.values]),df.dEg.values
    for s in ["LOCO","LOBO","LOHO"]:
        col={"LOCO":"A_cation","LOBO":"B","LOHO":"X"}[s]
        for tr,te in splits[s]:
            m=lgbm(); m.fit(Xu[tr],y[tr]); p=m.predict(Xu[te])
            yt=df.Eg_HSE.values[te] if mode=="delta" else y[te]
            if mode=="delta": p=p+df.Eg_GGA.values[te]
            rows.append(dict(mode=mode,split=s,fold=df.iloc[te][col].iloc[0],n=len(te),
                             MAE=mean_absolute_error(yt,p),R2=r2_score(yt,p)))
        print(f"{mode} {s} done",flush=True)
fl=pd.DataFrame(rows); fl.to_csv(TABLES / "fold_level_metrics.csv",index=False)
print(fl.groupby(["mode","split"]).MAE.agg(["mean","std","min","max"]).round(4).to_string(),flush=True)
print("\nLOHO:\n"+fl[fl.split=="LOHO"][["mode","fold","n","MAE","R2"]].round(4).to_string(index=False),flush=True)
print("\nLOBO:\n"+fl[fl.split=="LOBO"][["mode","fold","n","MAE","R2"]].round(4).to_string(index=False),flush=True)
print("DONE_E7",flush=True)
