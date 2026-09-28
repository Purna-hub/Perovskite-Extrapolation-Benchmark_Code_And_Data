"""
Stage 04 - protocol audit across feature blocks and fidelity
formulations (Table 3, Tables S2-S3, Fig. 2a,b).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd, json
warnings.filterwarnings('ignore')
from features import build_features, assemble
from evaluation import make_splits, model_zoo, metrics, cross_validate
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv(DATA / 'hoip_master.csv')
df, B = build_features(df)
print({k: len(v) for k, v in B.items()})
df.to_csv(DATA / DATA / 'hoip_features.csv', index=False)
splits = make_splits(df)

def lgbm():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
        ('est', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.035, num_leaves=31,
         min_child_samples=8, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
         reg_lambda=1.0, n_jobs=-1, random_state=0, verbose=-1))])

FB = {
 'C'      : ['COMP'],
 'C+L'    : ['COMP','LLM'],
 'C+L+P'  : ['COMP','LLM','PHYS'],
 'C+S'    : ['COMP','STRUCT'],
 'C+L+P+S': ['COMP','LLM','PHYS','STRUCT'],
}
rows=[]
for fb,names in FB.items():
    cols = assemble(names, B)
    X = df[cols].values
    for mode in ['direct','MFdirect','delta']:
        if mode=='direct':
            Xu, y, base, yt = X, df.Eg_HSE.values, None, None
        elif mode=='MFdirect':
            Xu = np.column_stack([X, df.Eg_GGA.values]); y=df.Eg_HSE.values; base=None; yt=None
        else:
            Xu = np.column_stack([X, df.Eg_GGA.values]); y=df.dEg.values
            base=df.Eg_GGA.values; yt=df.Eg_HSE.values
        for sname, sp in splits.items():
            yy, yp = cross_validate(lgbm, Xu, y, sp, base, yt)
            m = metrics(yy, yp); m.update(features=fb, mode=mode, split=sname, nfeat=Xu.shape[1])
            rows.append(m); pd.DataFrame(rows).to_csv(TABLES / 'protocol_audit.csv',index=False); print(f'{fb:9s} {mode:9s} {sname:7s} MAE={m["MAE"]:.4f} R2={m["R2"]:.4f}', flush=True)
res = pd.DataFrame(rows)
res.to_csv(TABLES / 'protocol_audit.csv', index=False)
piv = res.pivot_table(index=['mode','features'], columns='split', values='MAE')
piv = piv[['random','group','LOCO','LOBO','LOHO']]
print('\n=== MAE (eV) ===\n', piv.round(4).to_string())
piv2 = res.pivot_table(index=['mode','features'], columns='split', values='R2')[['random','group','LOCO','LOBO','LOHO']]
print('\n=== R2 ===\n', piv2.round(4).to_string())
