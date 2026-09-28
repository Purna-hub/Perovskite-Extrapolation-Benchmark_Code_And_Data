"""
Cached large-language-model elicitation of A-site cation descriptors,
with the two literature tabulations used for validation.

LLM-elicited A-site cation descriptors for hybrid organic-inorganic halide perovskites.

ELICITATION PROTOCOL (Section 2.3 of the manuscript)
---------------------------------------------------
The descriptor table LLM_DESCRIPTORS below was obtained by prompting a large language
model (Anthropic Claude, Opus class) through the vendor chat interface on 20 September
2026. A single zero-shot turn was used, with no system prompt, no retrieval, no tool
access, no few-shot examples and no tabulated input. Default decoding settings were used
and no re-sampling was performed, so the response is a single draw rather than an
ensemble. The response was parsed as JSON; units were fixed by the prompt and no response
required repair.

The verbatim prompt is stored in ELICITATION_PROMPT and the verbatim response is cached
in LLM_DESCRIPTORS. The downstream pipeline reads only this cache, so every result in the
manuscript is deterministic and reproducible without access to any model or API.

Because a single draw does not characterise the model's output distribution, no claim is
made about the stability of these values under re-sampling. The permutation control in
09_permutation_control.py, not the elicited values themselves, carries the weight of the
conclusion reported in Section 3.6.

VALIDATION ANCHORS
------------------
LIT_RADII_KIESLICH and LIT_RADII_DFT_ISOCHARGE are two independent published tabulations
of A-site effective radii. They are used ONLY for the post-hoc memorisation control in
03_validate_llm_descriptors.py and are never supplied to the model or used as features.

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""

ELICITATION_PROMPT = """You are a solid-state chemist. For each A-site molecular cation
listed below, return a JSON object of physico-chemical descriptors relevant to the
formation and electronic structure of ABX3 hybrid organic-inorganic halide perovskites.

Cations: Acetamidinium, Ammonium, Azetidinium, Butylammonium, Dimethylammonium,
Ethylammonium, Formamidinium, Guanidinium, Hydrazinium, Hydroxylammonium, Imidazolium,
Isopropylammonium, Methylammonium, Propylammonium, Tetramethylammonium, Trimethylammonium

For each cation report exactly these keys:
  r_eff_pm     effective ionic radius in the 12-coordinate cuboctahedral A-site cavity (pm)
  mu_D         gas-phase electric dipole moment of the cation (Debye)
  n_hbond_don  number of hydrogen-bond-donating protons (N-H, O-H)
  n_hbond_acc  number of hydrogen-bond-accepting lone pairs
  rot_sym      order of the highest proper rotation axis of the cation
  globularity  sphericity of the cation, 0 (rod-like) to 1 (spherical)
  rigidity     conformational rigidity, 0 (freely rotating chain) to 1 (rigid)
  pi_system    1 if the cation has a delocalised pi system, else 0
  charge_deloc extent of positive-charge delocalisation, 0 (localised on one N) to 1 (fully delocalised)
  polariz_A3   static electronic polarisability of the cation (cubic Angstrom)
  n_C, n_N, n_O  heavy-atom counts

Return ONLY valid JSON, no commentary."""

# ---- cached zero-shot LLM response -----------------------------------------
LLM_DESCRIPTORS = {
    "Ammonium":           dict(r_eff_pm=146, mu_D=0.00, n_hbond_don=4, n_hbond_acc=0, rot_sym=3,
                               globularity=1.00, rigidity=1.00, pi_system=0, charge_deloc=0.00,
                               polariz_A3=1.5,  n_C=0, n_N=1, n_O=0),
    "Hydroxylammonium":   dict(r_eff_pm=216, mu_D=2.10, n_hbond_don=4, n_hbond_acc=2, rot_sym=1,
                               globularity=0.95, rigidity=0.90, pi_system=0, charge_deloc=0.10,
                               polariz_A3=2.8,  n_C=0, n_N=1, n_O=1),
    "Hydrazinium":        dict(r_eff_pm=217, mu_D=2.50, n_hbond_don=5, n_hbond_acc=1, rot_sym=1,
                               globularity=0.95, rigidity=0.90, pi_system=0, charge_deloc=0.15,
                               polariz_A3=3.2,  n_C=0, n_N=2, n_O=0),
    "Methylammonium":     dict(r_eff_pm=217, mu_D=2.30, n_hbond_don=3, n_hbond_acc=0, rot_sym=3,
                               globularity=0.93, rigidity=1.00, pi_system=0, charge_deloc=0.05,
                               polariz_A3=3.6,  n_C=1, n_N=1, n_O=0),
    "Azetidinium":        dict(r_eff_pm=250, mu_D=1.80, n_hbond_don=2, n_hbond_acc=0, rot_sym=2,
                               globularity=0.88, rigidity=0.85, pi_system=0, charge_deloc=0.10,
                               polariz_A3=6.5,  n_C=3, n_N=1, n_O=0),
    "Formamidinium":      dict(r_eff_pm=253, mu_D=0.30, n_hbond_don=4, n_hbond_acc=0, rot_sym=2,
                               globularity=0.85, rigidity=1.00, pi_system=1, charge_deloc=0.90,
                               polariz_A3=5.0,  n_C=1, n_N=2, n_O=0),
    "Imidazolium":        dict(r_eff_pm=258, mu_D=1.20, n_hbond_don=2, n_hbond_acc=0, rot_sym=2,
                               globularity=0.80, rigidity=1.00, pi_system=1, charge_deloc=0.95,
                               polariz_A3=7.8,  n_C=3, n_N=2, n_O=0),
    "Dimethylammonium":   dict(r_eff_pm=272, mu_D=2.00, n_hbond_don=2, n_hbond_acc=0, rot_sym=2,
                               globularity=0.86, rigidity=0.90, pi_system=0, charge_deloc=0.05,
                               polariz_A3=5.5,  n_C=2, n_N=1, n_O=0),
    "Ethylammonium":      dict(r_eff_pm=274, mu_D=2.20, n_hbond_don=3, n_hbond_acc=0, rot_sym=1,
                               globularity=0.78, rigidity=0.80, pi_system=0, charge_deloc=0.05,
                               polariz_A3=5.6,  n_C=2, n_N=1, n_O=0),
    "Acetamidinium":      dict(r_eff_pm=277, mu_D=0.80, n_hbond_don=4, n_hbond_acc=0, rot_sym=1,
                               globularity=0.83, rigidity=0.95, pi_system=1, charge_deloc=0.85,
                               polariz_A3=6.6,  n_C=2, n_N=2, n_O=0),
    "Guanidinium":        dict(r_eff_pm=278, mu_D=0.00, n_hbond_don=6, n_hbond_acc=0, rot_sym=3,
                               globularity=0.87, rigidity=1.00, pi_system=1, charge_deloc=1.00,
                               polariz_A3=6.4,  n_C=1, n_N=3, n_O=0),
    "Trimethylammonium":  dict(r_eff_pm=290, mu_D=1.40, n_hbond_don=1, n_hbond_acc=0, rot_sym=3,
                               globularity=0.90, rigidity=1.00, pi_system=0, charge_deloc=0.00,
                               polariz_A3=7.4,  n_C=3, n_N=1, n_O=0),
    "Tetramethylammonium":dict(r_eff_pm=292, mu_D=0.00, n_hbond_don=0, n_hbond_acc=0, rot_sym=3,
                               globularity=0.95, rigidity=1.00, pi_system=0, charge_deloc=0.00,
                               polariz_A3=9.2,  n_C=4, n_N=1, n_O=0),
    "Isopropylammonium":  dict(r_eff_pm=300, mu_D=2.20, n_hbond_don=3, n_hbond_acc=0, rot_sym=1,
                               globularity=0.82, rigidity=0.80, pi_system=0, charge_deloc=0.05,
                               polariz_A3=7.4,  n_C=3, n_N=1, n_O=0),
    "Propylammonium":     dict(r_eff_pm=315, mu_D=2.20, n_hbond_don=3, n_hbond_acc=0, rot_sym=1,
                               globularity=0.68, rigidity=0.60, pi_system=0, charge_deloc=0.05,
                               polariz_A3=7.5,  n_C=3, n_N=1, n_O=0),
    "Butylammonium":      dict(r_eff_pm=350, mu_D=2.20, n_hbond_don=3, n_hbond_acc=0, rot_sym=1,
                               globularity=0.60, rigidity=0.50, pi_system=0, charge_deloc=0.05,
                               polariz_A3=9.3,  n_C=4, n_N=1, n_O=0),
}

# ---- independent literature anchors (validation only) -----------------------
# Kieslich, Sun & Cheetham, Chem. Sci. 2014/2015 (r_eff = r_mass + r_ion), pm
LIT_RADII_KIESLICH = {
    "Ammonium": 146, "Hydroxylammonium": 216, "Hydrazinium": 217, "Methylammonium": 217,
    "Azetidinium": 250, "Formamidinium": 253, "Imidazolium": 258, "Dimethylammonium": 272,
    "Ethylammonium": 274, "Guanidinium": 278, "Tetramethylammonium": 292,
}
# Becker, Klar & Kieslich, Dalton Trans. 2017 (DFT B3LYP/6-311G** isocharge sphere), pm
LIT_RADII_DFT_ISOCHARGE = {
    "Ammonium": 170, "Hydrazinium": 220, "Hydroxylammonium": 226, "Methylammonium": 238,
    "Formamidinium": 277, "Guanidinium": 280, "Azetidinium": 284, "Dimethylammonium": 296,
    "Ethylammonium": 299, "Acetamidinium": 300, "Tetramethylammonium": 301,
    "Imidazolium": 303, "Trimethylammonium": 304, "Isopropylammonium": 307,
}

# Shannon ionic radii (pm), CN=6 -- used for the geometric factors
R_B = {"Ge": 73.0, "Sn": 115.0, "Pb": 119.0}
R_X = {"F": 133.0, "Cl": 181.0, "Br": 196.0, "I": 220.0}


def geometric_factors(a_cation, B, X):
    """Goldschmidt tolerance factor t and Pauling octahedral factor mu,
    using the LLM-elicited A-site radius."""
    import math
    rA = LLM_DESCRIPTORS[a_cation]["r_eff_pm"]
    rB, rX = R_B[B], R_X[X]
    t = (rA + rX) / (math.sqrt(2.0) * (rB + rX))
    mu = rB / rX
    # Bartel et al. (2019) revised tolerance factor tau
    nA = 1.0
    try:
        tau = rX / rB - nA * (nA - (rA / rB) / math.log(rA / rB))
    except (ValueError, ZeroDivisionError):
        tau = float("nan")
    return t, mu, tau
