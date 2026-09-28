"""
Stage 06 - split-conformal coverage under every protocol
(Table 6, Table S5, Fig. 4).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from features import build_features, assemble
from evaluation import make_splits, split_conformal
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
df = pd.read_csv(DATA / "hoip_master.csv"); df, B = build_features(df)
splits = make_splits(df)
def lgbm():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
        ("est", lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_child_samples=8, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
         reg_lambda=1.0, n_jobs=1, random_state=0, verbose=-1))])
CONFIGS = {"pre-DFT (C+L+P) direct": (["COMP","LLM","PHYS"],"direct"),
           "pre-DFT (C+L+P) delta": (["COMP","LLM","PHYS"],"delta"),
           "post-relax (C+L+P+S) delta": (["COMP","LLM","PHYS","STRUCT"],"delta")}
rows=[]
for cname,(blocks,mode) in CONFIGS.items():
    X = df[assemble(blocks,B)].values
    if mode=="direct": Xu,yy,bs,yt = X, df.Eg_HSE.values, None, None
    else:
        Xu = np.column_stack([X, df.Eg_GGA.values]); yy=df.dEg.values
        bs=df.Eg_GGA.values; yt=df.Eg_HSE.values
    for sname in ["random","group","LOCO","LOBO","LOHO"]:
        for calib,grp in (("i.i.d.",None),("group",df.composition.values)):
            r,lo,hi = split_conformal(lgbm,Xu,yy,splits[sname],alpha=0.1,
                                      delta_base=bs,y_true=yt,groups=grp)
            r.update(config=cname,split=sname,calibration=calib); rows.append(r)
            pd.DataFrame(rows).to_csv(TABLES / "conformal_coverage.csv",index=False)
            print(f"{cname:27s} {sname:7s} {calib:6s} cov={r['coverage']:.3f} width={r['mean_width']:.3f}",flush=True)
print("DONE_conformal",flush=True)
