"""
Stage 07 - SHAP attribution, learning curves and composition-level
screening (Fig. 3b, Fig. 5, Table S9).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, TABLES
import warnings, numpy as np, pandas as pd, json
warnings.filterwarnings("ignore")
from features import build_features, assemble
from evaluation import make_splits, metrics, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

df = pd.read_csv(DATA / "hoip_master.csv")
df, B = build_features(df)
splits = make_splits(df)

def lgbm():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("est", lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                      num_leaves=31, min_child_samples=8, subsample=0.85,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                      n_jobs=1, random_state=0, verbose=-1))])

# ================================================================= SHAP SHAP
import shap
cols = assemble(["COMP", "LLM", "PHYS", "STRUCT"], B) + ["Eg_GGA"]
X = df[cols].values
SHAP_OUT = {"Eg_HSE": "shap_attribution_bandgap.csv",
            "dEg":    "shap_attribution_residual.csv"}
for tgt, name in ((df.dEg.values, "dEg"), (df.Eg_HSE.values, "Eg_HSE")):
    m = lgbm(); m.fit(X, tgt)
    ex = shap.TreeExplainer(m.named_steps["est"])
    sv = ex.shap_values(m.named_steps["imp"].transform(X))
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0)})
    imp["block"] = ["GGA" if f == "Eg_GGA" else
                    next(b for b in ["LLM", "PHYS", "STRUCT", "COMP"] if f in B[b])
                    for f in cols]
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(TABLES / SHAP_OUT[name], index=False)
    print(f"\n=== SHAP top-15 for {name} ===", flush=True)
    print(imp.head(15).to_string(index=False), flush=True)
    print("block totals:", imp.groupby("block").mean_abs_shap.sum().round(4).to_dict(), flush=True)

# ======================================================== learning curves learning curves
rows = []
fracs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
for cname, blocks, mode in (("direct", ["COMP", "LLM", "PHYS", "STRUCT"], "direct"),
                            ("delta", ["COMP", "LLM", "PHYS", "STRUCT"], "delta")):
    cc = assemble(blocks, B)
    Xb = df[cc].values
    if mode == "direct":
        Xu, y = Xb, df.Eg_HSE.values
    else:
        Xu, y = np.column_stack([Xb, df.Eg_GGA.values]), df.dEg.values
    for f in fracs:
        maes = []
        for rep in range(5):
            gss = GroupShuffleSplit(n_splits=1, train_size=f, random_state=rep)
            tr, te = next(gss.split(Xu, y, groups=df.composition))
            m = lgbm(); m.fit(Xu[tr], y[tr]); p = m.predict(Xu[te])
            if mode == "delta":
                p = p + df.Eg_GGA.values[te]; yt = df.Eg_HSE.values[te]
            else:
                yt = y[te]
            maes.append(np.abs(yt - p).mean())
        rows.append(dict(mode=cname, frac=f, n_train=int(f * len(df)),
                         MAE=np.mean(maes), MAE_std=np.std(maes)))
        print(f"[LC] {cname:7s} frac={f:.2f} MAE={np.mean(maes):.4f}+-{np.std(maes):.4f}", flush=True)
lc = pd.DataFrame(rows); lc.to_csv(TABLES / "learning_curves.csv", index=False)

# ============================================== screening composition-level screening
# Train the delta model on all data, then report composition-averaged predicted
# HSE06 gaps and geometric factors for every A-B-X combination, flagging those
# inside the Shockley-Queisser optimum for single-junction PV (1.1-1.5 eV) and
# for the wide-gap top cell of a tandem (1.7-1.9 eV).
from llm_descriptors import geometric_factors
cc = assemble(["COMP", "LLM", "PHYS", "STRUCT"], B)
Xu = np.column_stack([df[cc].values, df.Eg_GGA.values])
m = lgbm(); m.fit(Xu, df.dEg.values)
df["Eg_HSE_pred"] = m.predict(Xu) + df.Eg_GGA.values

agg = (df.groupby(["A_cation", "B", "X"])
         .agg(n_poly=("entry", "size"),
              Eg_GGA=("Eg_GGA", "mean"),
              Eg_HSE=("Eg_HSE", "mean"),
              Eg_HSE_min=("Eg_HSE", "min"),
              Erel1=("Erel1", "mean"),
              eps_tot=("eps_tot", "mean"))
         .reset_index())
gf = [geometric_factors(a, b, x) for a, b, x in zip(agg.A_cation, agg.B, agg.X)]
agg["tol_t"] = [g[0] for g in gf]; agg["oct_mu"] = [g[1] for g in gf]
agg["perovskite_formable"] = ((agg.tol_t.between(0.8, 1.1)) & (agg.oct_mu > 0.414))
agg["SQ_single"] = agg.Eg_HSE_min.between(1.1, 1.5)
agg["tandem_top"] = agg.Eg_HSE_min.between(1.7, 1.9)
agg["Pb_free"] = agg.B != "Pb"
agg = agg.sort_values("Eg_HSE_min")
agg.to_csv(TABLES / "composition_screening.csv", index=False)
hits = agg[agg.perovskite_formable & (agg.SQ_single | agg.tandem_top)]
print("\n=== screening screening: formable AND in a PV gap window ===", flush=True)
print(hits[["A_cation", "B", "X", "n_poly", "Eg_HSE_min", "Eg_HSE", "tol_t",
            "oct_mu", "Erel1", "SQ_single", "tandem_top", "Pb_free"]]
      .round(3).to_string(index=False), flush=True)
print(f"\nformable compositions: {int(agg.perovskite_formable.sum())}/{len(agg)}", flush=True)
print(f"Pb-free hits: {int((hits.Pb_free).sum())}", flush=True)
print("\nDONE_E4", flush=True)
