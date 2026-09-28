"""
Feature construction: the five switchable descriptor blocks
(COMP, LLM, PHYS, STRUCT, GGA).


Four feature blocks, designed so that each can be switched on/off for ablation:

  COMP   : composition-derived elemental descriptors of the B and X sites
           (pymatgen Element properties) + stoichiometric heavy-atom counts.
           Available *before* any DFT calculation.
  LLM    : zero-shot LLM-elicited physico-chemical descriptors of the A-site
           molecular cation (see llm_descriptors.py). Also pre-DFT.
  PHYS   : physically-motivated composite descriptors built from LLM radii and
           Shannon radii -- Goldschmidt t, octahedral mu, Bartel tau, plus
           electronegativity differences and a hydrogen-bond-density term.
           Also pre-DFT.
  STRUCT : relaxed-cell geometric descriptors (lattice parameters, cell volume,
           density, angular distortion). Requires the GGA relaxation, so this
           block is only available in the "post-relaxation" regime.

  GGA    : the PBE/GGA band gap -- the low-fidelity rung of the multi-fidelity
           ladder. Used as an input feature for MF-direct and Delta models.

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
import numpy as np
import pandas as pd
from pymatgen.core import Element

from llm_descriptors import LLM_DESCRIPTORS, geometric_factors, R_B, R_X

_ELEMENT_CACHE = {}


def _element_props(sym):
    if sym in _ELEMENT_CACHE:
        return _ELEMENT_CACHE[sym]
    e = Element(sym)
    p = {
        "EN": float(e.X),
        "Z": float(e.Z),
        "mass": float(e.atomic_mass),
        "row": float(e.row),
        "group": float(e.group),
        "r_ion": float(e.average_ionic_radius),
        "r_atom": float(e.atomic_radius) if e.atomic_radius else np.nan,
        "IE1": float(e.ionization_energy) if e.ionization_energy else np.nan,
        "EA": float(e.electron_affinity) if e.electron_affinity else np.nan,
        "n_val": float(sum(e.full_electronic_structure[-1][2:])) if e.full_electronic_structure else np.nan,
    }
    _ELEMENT_CACHE[sym] = p
    return p


def build_features(df):
    """Attach all feature blocks to the master dataframe. Returns (df, blocks)."""
    df = df.copy()

    # ---------------- COMP ----------------
    comp_cols = []
    for site in ("B", "X"):
        props = df[site].map(_element_props)
        for key in _element_props("Pb").keys():
            col = f"{site}_{key}"
            df[col] = props.map(lambda d, k=key: d[k])
            comp_cols.append(col)
    for c in ("nC", "nN", "nH", "nO", "nS", "natoms"):
        comp_cols.append(c)
    df["heavy_frac"] = (df.nC + df.nN + df.nO + df.nS) / df.natoms
    df["CN_ratio"] = df.nC / df.nN.replace(0, np.nan)
    df["CN_ratio"] = df["CN_ratio"].fillna(0.0)
    comp_cols += ["heavy_frac", "CN_ratio"]

    # ---------------- LLM ----------------
    llm_keys = ["r_eff_pm", "mu_D", "n_hbond_don", "n_hbond_acc", "rot_sym",
                "globularity", "rigidity", "pi_system", "charge_deloc", "polariz_A3"]
    llm_cols = []
    for k in llm_keys:
        col = f"A_{k}"
        df[col] = df.A_cation.map(lambda c, k=k: LLM_DESCRIPTORS[c][k])
        llm_cols.append(col)

    # ---------------- PHYS ----------------
    gf = [geometric_factors(a, b, x) for a, b, x in zip(df.A_cation, df.B, df.X)]
    df["tol_t"] = [g[0] for g in gf]
    df["oct_mu"] = [g[1] for g in gf]
    df["bartel_tau"] = [g[2] for g in gf]
    df["r_A"] = df.A_cation.map(lambda c: LLM_DESCRIPTORS[c]["r_eff_pm"])
    df["r_Bs"] = df.B.map(R_B)
    df["r_Xs"] = df.X.map(R_X)
    df["dEN_BX"] = df.X_EN - df.B_EN                      # B-X bond ionicity proxy
    df["BX_ionicity"] = 1.0 - np.exp(-0.25 * df.dEN_BX ** 2)   # Pauling ionicity
    df["hbond_density"] = df.A_n_hbond_don / (df.r_A ** 2) * 1e4
    df["A_vol_frac"] = (df.r_A ** 3) / ((df.r_A + df.r_Xs) ** 3)
    df["polariz_ratio"] = df.A_polariz_A3 / df.X_r_ion ** 3
    df["dipole_over_r"] = df.A_mu_D / (df.r_A / 100.0) ** 2
    phys_cols = ["tol_t", "oct_mu", "bartel_tau", "r_A", "r_Bs", "r_Xs", "dEN_BX",
                 "BX_ionicity", "hbond_density", "A_vol_frac", "polariz_ratio",
                 "dipole_over_r"]

    # ---------------- STRUCT ----------------
    df["V_per_atom"] = df.cell_vol / df.natoms
    df["aspect"] = df[["a", "b", "c"]].max(axis=1) / df[["a", "b", "c"]].min(axis=1)
    df["ang_dev"] = (df[["alpha", "beta", "gamma"]] - 90.0).abs().mean(axis=1)
    df["ang_max_dev"] = (df[["alpha", "beta", "gamma"]] - 90.0).abs().max(axis=1)
    df["pack_frac"] = ((4.0 / 3.0) * np.pi *
                       (df.r_A ** 3 + df.r_Bs ** 3 + 3 * df.r_Xs ** 3) * 1e-6) / df.cell_vol
    struct_cols = ["a", "b", "c", "alpha", "beta", "gamma", "cell_vol", "V_per_atom",
                   "aspect", "ang_dev", "ang_max_dev", "rho", "pack_frac"]

    blocks = {
        "COMP": comp_cols,
        "LLM": llm_cols,
        "PHYS": phys_cols,
        "STRUCT": struct_cols,
        "GGA": ["Eg_GGA"],
    }
    # drop all-NaN / constant columns from blocks
    for name, cols in blocks.items():
        keep = [c for c in cols if df[c].notna().any() and df[c].nunique() > 1]
        blocks[name] = keep
    return df, blocks


def assemble(block_names, blocks):
    cols = []
    for b in block_names:
        cols += blocks[b]
    # preserve order, drop duplicates
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
