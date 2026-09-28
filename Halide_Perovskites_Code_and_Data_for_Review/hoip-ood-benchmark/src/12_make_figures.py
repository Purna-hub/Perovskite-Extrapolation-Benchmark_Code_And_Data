"""
Stage 12 - generate Figures 1-5 at publication resolution.


Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, FIGURES, TABLES
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "figure.dpi": 300,
    "savefig.dpi": 600, "savefig.bbox": "tight", "axes.grid": False,
})
SPL = ["random", "group", "LOCO", "LOBO", "LOHO"]
LBL = {"random": "Random", "group": "Grouped", "LOCO": "LOCO", "LOBO": "LOBO", "LOHO": "LOHO"}
C = {"direct": "#C1443C", "MFdirect": "#E8A33D", "delta": "#2A6FA8"}

e1 = pd.read_csv(TABLES / "protocol_audit.csv")
e2a = pd.read_csv(TABLES / "physical_baselines.csv")
e2b = pd.read_csv(TABLES / "model_zoo.csv")
e3 = pd.read_csv(TABLES / "conformal_coverage.csv")
e4b = pd.read_csv(TABLES / "learning_curves.csv")
shap_hse = pd.read_csv(TABLES / "shap_attribution_bandgap.csv")
val = pd.read_csv(DATA / "llm_radii_validation.csv")
df = pd.read_csv(DATA / "hoip_features.csv")

# ---------------------------------------------------------------- Figure 1 --
# Dataset anatomy + variance decomposition
fig = plt.figure(figsize=(7.2, 4.6))
gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
for x, c in zip(["F", "Cl", "Br", "I"], ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
    s = df[df.X == x].Eg_HSE
    ax.hist(s, bins=22, alpha=0.65, label=x, color=c)
ax.set_xlabel(r"$E_g^{\rm HSE06}$ (eV)"); ax.set_ylabel("count")
ax.legend(frameon=False, fontsize=7, title="X", title_fontsize=7)
ax.set_title("(a) gap by halide", fontsize=8, loc="left")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(df.Eg_GGA, df.Eg_HSE, s=4, alpha=0.35, c="#2A6FA8", lw=0)
lim = [0, max(df.Eg_HSE.max(), df.Eg_GGA.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.7)
a, b = np.polyfit(df.Eg_GGA, df.Eg_HSE, 1)
ax.plot(lim, [a * v + b for v in lim], "-", c="#C1443C", lw=1.1,
        label=f"$y$={a:.2f}$x$+{b:.2f}")
ax.set_xlabel(r"$E_g^{\rm GGA}$ (eV)"); ax.set_ylabel(r"$E_g^{\rm HSE06}$ (eV)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("(b) fidelity correlation", fontsize=8, loc="left")

ax = fig.add_subplot(gs[0, 2])
ax.hist(df.dEg, bins=40, color="#7A9E3F", alpha=0.85)
ax.axvline(df.dEg.mean(), c="k", ls="--", lw=0.8)
ax.set_xlabel(r"$\Delta = E_g^{\rm HSE06}-E_g^{\rm GGA}$ (eV)"); ax.set_ylabel("count")
ax.text(0.97, 0.9, f"$\\mu$={df.dEg.mean():.3f}\n$\\sigma$={df.dEg.std():.3f}",
        transform=ax.transAxes, ha="right", fontsize=7)
ax.set_title("(c) the learnable residual", fontsize=8, loc="left")

ax = fig.add_subplot(gs[1, :2])
order = df.groupby("A_cation").Eg_HSE.median().sort_values().index
data = [df[df.A_cation == c].Eg_HSE.values for c in order]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
for p in bp["boxes"]:
    p.set(facecolor="#9EC1DE", lw=0.6)
for p in bp["medians"]:
    p.set(color="#17324B", lw=1.0)
ax.set_xticklabels([c[:9] for c in order], rotation=55, ha="right", fontsize=6.5)
ax.set_ylabel(r"$E_g^{\rm HSE06}$ (eV)")
ax.set_title("(d) spread across the 16 A-site cations", fontsize=8, loc="left")

ax = fig.add_subplot(gs[1, 2])
wg = df.groupby("composition").Eg_HSE.transform("mean")
frac = 1 - (df.Eg_HSE - wg).var() / df.Eg_HSE.var()
ax.bar([0, 1], [frac, 1 - frac], color=["#C1443C", "#BFC7CE"], width=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["between\ncomposition", "within\ncomposition"], fontsize=7)
ax.set_ylabel(r"fraction of var($E_g^{\rm HSE06}$)")
ax.text(0, frac + 0.03, f"{frac:.3f}", ha="center", fontsize=8, weight="bold")
ax.set_ylim(0, 1.12)
ax.set_title("(e) variance decomposition", fontsize=8, loc="left")
fig.savefig(FIGURES / "figure_1_dataset.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2 --
# The protocol audit: performance vs split severity
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
for ax, metric, ylab in zip(axes, ["MAE", "R2"], [r"MAE (eV)", r"$R^2$"]):
    for mode, lab in (("direct", "Direct"), ("MFdirect", "MF-direct"), ("delta", r"$\Delta$-ML")):
        sub = e1[(e1["mode"] == mode) & (e1.features == "C+L+P+S")]
        v = [sub[sub.split == s][metric].iloc[0] for s in SPL]
        ax.plot(range(5), v, "o-", color=C[mode], label=lab, ms=4, lw=1.3)
    ax.set_xticks(range(5)); ax.set_xticklabels([LBL[s] for s in SPL], rotation=40, ha="right")
    ax.set_ylabel(ylab)
axes[0].set_yscale("log"); axes[0].legend(frameon=False, fontsize=7)
axes[0].set_title("(a) error vs protocol", fontsize=8, loc="left")
axes[1].set_ylim(0, 1.05); axes[1].set_title("(b) variance explained", fontsize=8, loc="left")

ax = axes[2]
zoo = e2b[e2b.config == "pre-DFT (C+L+P) direct"]
models = ["Ridge", "KRR-RBF", "MLP", "RF", "XGBoost", "LightGBM"]
w = 0.35
xs = np.arange(len(models))
for i, (s, c) in enumerate(zip(["random", "LOHO"], ["#8FB8DC", "#C1443C"])):
    v = [zoo[(zoo.model == m) & (zoo.split == s)].R2.mean() for m in models]
    ax.bar(xs + i * w - w / 2, v, w, color=c, label=LBL[s])
ax.axhline(0, c="k", lw=0.6); ax.set_ylim(-0.6, 1.0)
ax.set_xticks(xs); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
ax.set_ylabel(r"$R^2$"); ax.legend(frameon=False, fontsize=7)
ax.set_title("(c) rank inversion under OOD", fontsize=8, loc="left")
fig.savefig(FIGURES / "figure_2_protocol_audit.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3 --
# ML vs physical baselines + learning curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axes[0]
base_models = ["Scissor shift (const)", "Halide-resolved scissor", "Linear scaling"]
cols = ["#B0B7BD", "#7E8D99", "#4E6472"]
xs = np.arange(5); w = 0.19
for i, (bm, c) in enumerate(zip(base_models, cols)):
    v = [e2a[(e2a.model == bm) & (e2a.split == s)].MAE.iloc[0] for s in SPL]
    ax.bar(xs + (i - 1.5) * w, v, w, color=c, label=bm)
dl = e1[(e1["mode"] == "delta") & (e1.features == "C+L+P+S")]
v = [dl[dl.split == s].MAE.iloc[0] for s in SPL]
ax.bar(xs + 1.5 * w, v, w, color="#2A6FA8", label=r"$\Delta$-ML (LightGBM)")
ax.set_xticks(xs); ax.set_xticklabels([LBL[s] for s in SPL], rotation=40, ha="right")
ax.set_ylabel("MAE (eV)"); ax.legend(frameon=False, fontsize=6.5, ncol=1)
ax.set_title("(a) ML vs two-parameter physics", fontsize=8, loc="left")

ax = axes[1]
for mode, lab, c in (("direct", "Direct", "#C1443C"), ("delta", r"$\Delta$-ML", "#2A6FA8")):
    s = e4b[e4b["mode"] == mode]
    ax.errorbar(s.n_train, s.MAE, yerr=s.MAE_std, fmt="o-", color=c, label=lab, ms=4, lw=1.2, capsize=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("composition-grouped training set size")
ax.set_ylabel("MAE (eV)"); ax.legend(frameon=False, fontsize=7)
ax.set_title("(b) data efficiency", fontsize=8, loc="left")
fig.savefig(FIGURES / "figure_3_baselines_learning.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4 --
# Conformal coverage collapse
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axes[0]
cfgs = e3.config.unique()
mk = {"i.i.d.": "o", "group": "s"}
cc = {cfgs[0]: "#C1443C", cfgs[1]: "#E8A33D", cfgs[2]: "#2A6FA8"}
for cfg in cfgs:
    for cal in ["i.i.d.", "group"]:
        s = e3[(e3.config == cfg) & (e3.calibration == cal)]
        v = [s[s.split == sp].coverage.iloc[0] for sp in SPL]
        ax.plot(range(5), v, mk[cal] + "-", color=cc[cfg], ms=4, lw=1.1,
                alpha=1.0 if cal == "i.i.d." else 0.5,
                label=f"{cfg.split(' ')[0]} / {cal}")
ax.axhline(0.9, c="k", ls="--", lw=0.9)
ax.text(0.05, 0.915, "nominal 90%", fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels([LBL[s] for s in SPL], rotation=40, ha="right")
ax.set_ylabel("empirical coverage"); ax.set_ylim(0.2, 1.0)
ax.legend(frameon=False, fontsize=6, ncol=2)
ax.set_title("(a) conformal validity collapses OOD", fontsize=8, loc="left")

ax = axes[1]
s = e3[(e3.calibration == "i.i.d.")]
for cfg in cfgs:
    ss = s[s.config == cfg]
    ax.scatter([ss[ss.split == sp].mean_width.iloc[0] for sp in SPL],
               [ss[ss.split == sp].coverage.iloc[0] for sp in SPL],
               c=cc[cfg], s=32, label=cfg, edgecolor="k", lw=0.3)
ax.axhline(0.9, c="k", ls="--", lw=0.9)
ax.set_xscale("log"); ax.set_xlabel("mean interval width (eV)")
ax.set_ylabel("empirical coverage")
ax.legend(frameon=False, fontsize=6)
ax.set_title("(b) narrow and wrong", fontsize=8, loc="left")
fig.savefig(FIGURES / "figure_4_conformal.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5 --
# SHAP attribution + LLM descriptor validation
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
ax = axes[0]
top = shap_hse.head(12).iloc[::-1]
bc = {"GGA": "#2A6FA8", "COMP": "#7A9E3F", "STRUCT": "#E8A33D", "PHYS": "#B07AA1", "LLM": "#C1443C"}
ax.barh(range(len(top)), top.mean_abs_shap, color=[bc[b] for b in top.block])
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.feature, fontsize=6.5)
ax.set_xlabel("mean |SHAP| (eV)"); ax.set_xscale("log")
ax.set_title("(a) attribution", fontsize=8, loc="left")

ax = axes[1]
bt = shap_hse.groupby("block").mean_abs_shap.sum().sort_values(ascending=False)
ax.bar(range(len(bt)), bt.values, color=[bc[b] for b in bt.index])
ax.set_xticks(range(len(bt))); ax.set_xticklabels(bt.index, rotation=40, ha="right", fontsize=7)
ax.set_ylabel(r"$\Sigma$ mean |SHAP| (eV)"); ax.set_yscale("log")
ax.set_title("(b) block totals", fontsize=8, loc="left")

ax = axes[2]
ax.scatter(val.kies, val.llm, s=30, c="#2A6FA8", label="Kieslich", edgecolor="k", lw=0.3)
ax.scatter(val.dft, val.llm, s=30, c="#C1443C", marker="^", label="DFT isocharge", edgecolor="k", lw=0.3)
lim = [130, 370]; ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("literature $r_A$ (pm)"); ax.set_ylabel("LLM-elicited $r_A$ (pm)")
ax.legend(frameon=False, fontsize=6.5)
ax.set_title("(c) LLM descriptor fidelity", fontsize=8, loc="left")
fig.savefig(FIGURES / "figure_5_shap_llm.png"); plt.close(fig)
print("figures written")
