"""
Stage 08 - feature-free lookup baselines (Table 2, Table S1).

Three estimators that use no features and fit no parameters. Each predicts, for a test
structure, the mean HSE06 gap of the training-fold group it belongs to, falling back to
the global training mean when that group is absent from the training partition:

  * composition mean  - the operational form of the 85.6 % variance-decomposition claim
                        of Section 3.1, and the floor any model must beat on a random split
  * halide mean       - four numbers in total
  * halide + metal    - twelve numbers in total
  * A-cation mean     - reported for contrast; the A site carries little gap information

The feature-block ablation is produced by 04_protocol_audit.py, the permutation control
by 09_permutation_control.py, and the fold-level metrics by 10_fold_level_metrics.py.

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

df = pd.read_csv(DATA / "hoip_master.csv")
df, B = build_features(df)
splits = make_splits(df)
SPL = ["random", "group", "LOCO", "LOBO", "LOHO"]

def lgbm():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("est", lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                      num_leaves=31, min_child_samples=8, subsample=0.85,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      n_jobs=1, random_state=0, verbose=-1))])

# ------------------------------------------------------- lookup baselines
def lookup(keycols, name):
    rows = []
    y = df.Eg_HSE.values
    for sname in SPL:
        yp = np.full(len(y), np.nan)
        for tr, te in splits[sname]:
            g = df.iloc[tr].groupby(keycols).Eg_HSE.mean()
            glob = y[tr].mean()
            gd = g.to_dict()
            sub = df.iloc[te][keycols].values
            keys = [k[0] if len(keycols) == 1 else tuple(k) for k in sub]
            yp[te] = np.array([gd.get(k, glob) for k in keys])
        m = metrics(y, yp)
        # fraction of test points with no training-fold key (fallback to global mean)
        m.update(model=name, split=sname)
        rows.append(m)
    return rows

base = []
base += lookup(["A_cation", "B", "X"], "Model 0: composition-mean lookup")
base += lookup(["X"], "Halide-only lookup")
base += lookup(["B", "X"], "Halide + metal lookup")
base += lookup(["A_cation"], "A-cation-only lookup")
bl = pd.DataFrame(base)
bl.to_csv(TABLES / "lookup_baselines.csv", index=False)
print("=== lookup baselines/b lookup baselines ===", flush=True)
print(bl.pivot_table(index="model", columns="split", values="MAE")[SPL].round(4).to_string(), flush=True)
print(bl.pivot_table(index="model", columns="split", values="R2")[SPL].round(4).to_string(), flush=True)

print("\nFeature-block ablation is produced by 04_protocol_audit.py; the permutation\n"
      "control by 09_permutation_control.py; fold-level metrics by 10_fold_level_metrics.py.")
