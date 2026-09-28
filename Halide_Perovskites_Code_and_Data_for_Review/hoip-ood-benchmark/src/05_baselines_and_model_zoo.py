"""
Stage 05 - non-ML physical baselines and the model zoo
(Tables 4 and 5, Table S4, Fig. 2c, Fig. 3a).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd, json
warnings.filterwarnings("ignore")
from sklearn.base import clone
from features import build_features, assemble
from evaluation import make_splits, model_zoo, metrics, cross_validate

df = pd.read_csv(DATA / "hoip_master.csv")
df, B = build_features(df)
splits = make_splits(df)
SPL = ["random", "group", "LOCO", "LOBO", "LOHO"]

# =====================================================================  physical baselines
# Physically-motivated non-ML baselines for the GGA -> HSE06 upgrade.
rows = []
y = df.Eg_HSE.values
g = df.Eg_GGA.values
for sname in SPL:
    sp = splits[sname]
    # (i) constant scissor shift fitted on the training fold
    yp = np.zeros(len(y))
    for tr, te in sp:
        yp[te] = g[te] + df.dEg.values[tr].mean()
    m = metrics(y, yp); m.update(model="Scissor shift (const)", split=sname); rows.append(m)
    # (ii) linear scaling Eg_HSE = a*Eg_GGA + b
    yp = np.zeros(len(y))
    for tr, te in sp:
        a, b = np.polyfit(g[tr], y[tr], 1)
        yp[te] = a * g[te] + b
    m = metrics(y, yp); m.update(model="Linear scaling", split=sname); rows.append(m)
    # (iii) halide-resolved scissor (oracle within training halides only)
    yp = np.zeros(len(y))
    for tr, te in sp:
        med = df.iloc[tr].groupby("X").dEg.mean()
        glob = df.dEg.values[tr].mean()
        yp[te] = g[te] + df.iloc[te].X.map(med).fillna(glob).values
    m = metrics(y, yp); m.update(model="Halide-resolved scissor", split=sname); rows.append(m)
base = pd.DataFrame(rows)
base.to_csv(TABLES / "physical_baselines.csv", index=False)
print("=== physical baselines physical baselines: MAE (eV) ===", flush=True)
print(base.pivot_table(index="model", columns="split", values="MAE")[SPL].round(4).to_string(), flush=True)

# =====================================================================  model zoo
# Model zoo on the two headline configurations.
CONFIGS = {
    "pre-DFT (C+L+P) direct":   (["COMP", "LLM", "PHYS"], "direct"),
    "pre-DFT (C+L+P) delta":    (["COMP", "LLM", "PHYS"], "delta"),
    "post-relax (C+L+P+S) delta": (["COMP", "LLM", "PHYS", "STRUCT"], "delta"),
}
rows = []
for cname, (blocks, mode) in CONFIGS.items():
    cols = assemble(blocks, B)
    X = df[cols].values
    if mode == "direct":
        Xu, yy, bs, yt = X, df.Eg_HSE.values, None, None
    else:
        Xu = np.column_stack([X, df.Eg_GGA.values])
        yy, bs, yt = df.dEg.values, df.Eg_GGA.values, df.Eg_HSE.values
    for mname, mdl in model_zoo().items():
        if mname.startswith("_skip"):
            continue
        for sname in SPL:
            try:
                a, p = cross_validate(lambda m=mdl: clone(m),
                                      Xu, yy, splits[sname], bs, yt)
                m = metrics(a, p)
            except Exception as e:
                m = dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, MaxAE=np.nan)
            m.update(config=cname, model=mname, split=sname)
            rows.append(m)
            pd.DataFrame(rows).to_csv(TABLES / "model_zoo.csv", index=False)
            print(f"{cname:28s} {mname:11s} {sname:7s} MAE={m['MAE']:.4f} R2={m['R2']:.4f}", flush=True)
zoo = pd.DataFrame(rows)
print("\n=== model zoo model zoo: MAE (eV) ===", flush=True)
print(zoo.pivot_table(index=["config", "model"], columns="split", values="MAE")[SPL].round(4).to_string(), flush=True)

print("\nSplit-conformal coverage is produced by 06_conformal_uncertainty.py.")
