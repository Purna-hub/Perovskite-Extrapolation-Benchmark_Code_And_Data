"""
Stage 09 - permutation control for the LLM descriptor block
(Table 7, Table S7, Section 3.6).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from features import build_features, assemble
from evaluation import make_splits, metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
df = pd.read_csv(DATA / "hoip_master.csv"); df, B = build_features(df)
splits = make_splits(df); SPL=["random","group","LOCO","LOBO","LOHO"]
def lgbm():
    return Pipeline([("imp",SimpleImputer(strategy="median")),
      ("est",lgb.LGBMRegressor(n_estimators=250,learning_rate=0.07,num_leaves=31,
       min_child_samples=8,subsample=0.85,subsample_freq=1,colsample_bytree=0.8,
       reg_lambda=1.0,n_jobs=1,random_state=0,verbose=-1))])

# ---- permutation LLM permutation test (MF-direct, COMP+LLM) ------------------------
cats=sorted(df.A_cation.unique()); llm=B["LLM"]
lut=df.drop_duplicates("A_cation").set_index("A_cation")[llm]
def run_cfg(dfx, splitnames):
    X=np.column_stack([dfx[assemble(["COMP","LLM"],B)].values, df.Eg_GGA.values])
    y=df.Eg_HSE.values; out={}
    for s in splitnames:
        yp=np.zeros(len(y))
        for tr,te in splits[s]:
            m=lgbm(); m.fit(X[tr],y[tr]); yp[te]=m.predict(X[te])
        out[s]=(mean_absolute_error(y,yp), r2_score(y,yp))
    return out
SS=["LOHO","LOBO","LOCO"]
true_=run_cfg(df,SS)
Xc=np.column_stack([df[assemble(["COMP"],B)].values, df.Eg_GGA.values])
comp={}
for s in SS:
    yp=np.zeros(len(df)); y=df.Eg_HSE.values
    for tr,te in splits[s]:
        m=lgbm(); m.fit(Xc[tr],y[tr]); yp[te]=m.predict(Xc[te])
    comp[s]=(mean_absolute_error(y,yp), r2_score(y,yp))
print("COMP only      :", {k:(round(v[0],4),round(v[1],4)) for k,v in comp.items()}, flush=True)
print("COMP+LLM true  :", {k:(round(v[0],4),round(v[1],4)) for k,v in true_.items()}, flush=True)
rows=[]
rng=np.random.default_rng(0)
for rep in range(12):
    perm=rng.permutation(cats); mp=dict(zip(cats,perm))
    dfp=df.copy()
    for c in llm: dfp[c]=dfp.A_cation.map(lambda a,c=c: lut.loc[mp[a],c])
    r=run_cfg(dfp,SS)
    for s in SS: rows.append(dict(rep=rep,split=s,MAE=r[s][0],R2=r[s][1]))
    pd.DataFrame(rows).to_csv(TABLES / "llm_permutation_runs.csv",index=False)
    print(f"[perm {rep}] "+" ".join(f"{s}={r[s][0]:.4f}" for s in SS), flush=True)
pm=pd.DataFrame(rows)
print("\n=== PERMUTATION TEST ===", flush=True)
for s in SS:
    sub=pm[pm.split==s].MAE
    t,c=true_[s][0],comp[s][0]
    p=(sub<=t).mean()
    print(f"{s}: true(COMP+LLM)={t:.4f}  COMP-only={c:.4f}  "
          f"permuted={sub.mean():.4f}+/-{sub.std():.4f} [{sub.min():.4f},{sub.max():.4f}]  "
          f"p(perm<=true)={p:.3f}", flush=True)
pd.DataFrame([dict(split=s, comp_only=comp[s][0], comp_llm=true_[s][0],
                   perm_mean=pm[pm.split==s].MAE.mean(), perm_std=pm[pm.split==s].MAE.std(),
                   p_value=(pm[pm.split==s].MAE<=true_[s][0]).mean()) for s in SS]
            ).to_csv(TABLES / "llm_permutation_summary.csv",index=False)
print("DONE_PERM", flush=True)
