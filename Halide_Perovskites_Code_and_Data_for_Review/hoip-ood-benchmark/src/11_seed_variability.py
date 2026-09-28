"""
Stage 11 - across-seed variability of the random and grouped
protocols (Table S8, Section 2.6).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings,numpy as np,pandas as pd
warnings.filterwarnings("ignore")
from features import build_features,assemble
from sklearn.model_selection import KFold,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,r2_score
import lightgbm as lgb
df=pd.read_csv(DATA / "hoip_master.csv"); df,B=build_features(df)
def lgbm(seed):
    return Pipeline([("imp",SimpleImputer(strategy="median")),
      ("est",lgb.LGBMRegressor(n_estimators=250,learning_rate=0.07,num_leaves=31,
       min_child_samples=8,subsample=0.85,subsample_freq=1,colsample_bytree=0.8,
       reg_lambda=1.0,n_jobs=1,random_state=seed,verbose=-1))])
X0=df[assemble(["COMP","LLM","PHYS","STRUCT"],B)].values
rows=[]
for mode in ["direct","delta"]:
    if mode=="direct": Xu,y=X0,df.Eg_HSE.values
    else: Xu,y=np.column_stack([X0,df.Eg_GGA.values]),df.dEg.values
    for proto in ["random","group"]:
        for seed in range(5):
            sp=(list(KFold(5,shuffle=True,random_state=seed).split(np.arange(len(df))))
                if proto=="random" else
                list(GroupKFold(5,shuffle=True,random_state=seed).split(np.arange(len(df)),groups=df.composition)))
            yp=np.zeros(len(y))
            for tr,te in sp:
                m=lgbm(seed); m.fit(Xu[tr],y[tr]); yp[te]=m.predict(Xu[te])
            yt=y
            if mode=="delta": yp,yt=yp+df.Eg_GGA.values, df.Eg_HSE.values
            rows.append(dict(mode=mode,protocol=proto,seed=seed,
                             MAE=mean_absolute_error(yt,yp),R2=r2_score(yt,yp)))
            print(f"{mode} {proto} seed{seed} MAE={rows[-1]['MAE']:.4f}",flush=True)
r=pd.DataFrame(rows); r.to_csv(TABLES / "seed_variability.csv",index=False)
print(r.groupby(["mode","protocol"])[["MAE","R2"]].agg(["mean","std"]).round(4).to_string(),flush=True)
print("DONE_E8",flush=True)
