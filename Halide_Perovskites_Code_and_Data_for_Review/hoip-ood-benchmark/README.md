# Reproducibility package

**Manuscript**
Large language model descriptors and multi-fidelity learning for HSE06 band gaps of hybrid halide
perovskites under chemical extrapolation

**Authors**
Purnachary Munigadapa, B. Indera, Avula Edukondalu, Shyam Sunder Pabboju, J. Sampurna,
B. Srinivasa S. P. Kumar

**Submitted to** Computational Materials Science

> **Release status.** This archive accompanies a manuscript under review. The corresponding Zenodo
> record is embargoed until the date of publication, at which point it opens to unrestricted public
> access. Nothing is withheld from referees: the pipeline is complete and regenerates every number,
> table and figure reported in the manuscript.

---

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh
```

Runs end to end on **one CPU core in under one hour**. No GPU, no API key, and — after the single
download in `src/00_fetch_source_data.sh` — no network access. Every stochastic procedure is seeded.

Scripts resolve all paths through `src/paths.py`, so they can be run individually from any working
directory:

```bash
python src/04_protocol_audit.py
```

---

## Layout

```
├── run_all.sh                  full pipeline, stages 00-12
├── requirements.txt            pinned dependency versions
├── LICENSE
├── data/
│   ├── README.md               data dictionary
│   ├── raw/                    source dataset lands here (fetched, not redistributed)
│   ├── hoip_master.csv         merged structures + GGA/HSE06 properties  (1,346 rows)
│   ├── hoip_structures.csv     parsed CIF geometry and composition
│   ├── hoip_features.csv       full feature matrix used by the models
│   └── llm_radii_validation.csv   elicited vs published A-site radii
├── src/
│   ├── paths.py                single source of truth for all paths
│   ├── llm_descriptors.py      verbatim elicitation prompt + cached response
│   ├── features.py             the five switchable descriptor blocks
│   ├── evaluation.py           split protocols, model zoo, split-conformal UQ
│   ├── 00_fetch_source_data.sh
│   ├── 01_parse_structures.py        … 12_make_figures.py
├── results/
│   ├── tables/                 machine-readable results (CSV)
│   └── logs/                   unedited stdout of the original runs
└── figures/                    Figures 1-5 at publication resolution
```

---

## Every claim mapped to the file that produces it

| Manuscript item | Claim | Produced by | Released file |
|---|---|---|---|
| §3.1, Fig. 1e | Composition identity explains **85.6 %** of HSE06 gap variance | `02_assemble_dataset.py` | stdout; `data/hoip_master.csv` |
| Table 2, §3.1 | Composition-mean lookup R² = **0.784**; halide-only lookup **0.771** | `08_lookup_baselines.py` | `lookup_baselines.csv` |
| Table 3, Fig. 2a,b | Direct ML R² 0.967 (random) → **0.432** (leave-one-halide-out) | `04_protocol_audit.py` | `protocol_audit.csv` |
| Table 4, Fig. 2c | Relative performance reverses: ridge 0.730 vs LightGBM 0.307 | `05_baselines_and_model_zoo.py` | `model_zoo.csv` |
| Table 5, Fig. 3a | Two-parameter linear scaling **0.084 eV** beats Δ-ML **0.095 eV** under LOHO | `05_baselines_and_model_zoo.py` | `physical_baselines.csv` |
| Table 6, Fig. 4 | Conformal coverage 0.90 → **0.30** under leave-one-metal-out | `06_conformal_uncertainty.py` | `conformal_coverage.csv` |
| Table 7, §3.6 | **Permutation control**: permuted ≈ true, p = 0.17–0.67 | `09_permutation_control.py` | `llm_permutation_summary.csv` |
| §3.6, Fig. 5c | Elicited radii reproduce the Kieslich table exactly (MAE 0.0 pm) | `03_validate_llm_descriptors.py` | `llm_radii_validation.csv` |
| §3.2, Table S6 | Per-fold R² negative for 3 of 4 halides (direct model) | `10_fold_level_metrics.py` | `fold_level_metrics.csv` |
| Fig. 3b | Δ-learning reaches 0.073 eV with 10 % of the data | `07_attribution_and_screening.py` | `learning_curves.csv` |
| Fig. 5a,b | LLM block contributes 0.8 % of total SHAP attribution | `07_attribution_and_screening.py` | `shap_attribution_bandgap.csv` |
| §2.6, Table S8 | Across-seed dispersion ±0.002 eV, far below protocol effects | `11_seed_variability.py` | `seed_variability.csv` |
| §3.7 | 79 of 192 compositions formable; four lead-free candidates | `07_attribution_and_screening.py` | `composition_screening.csv` |
| Tables S2–S3 | Feature-block ablation across all protocols | `04_protocol_audit.py` | `protocol_audit.csv` |

`results/logs/` holds the unedited stdout of the original runs, so a referee can diff their output
against ours line by line.

---

## Notes for referees

**Determinism.** Every estimator is seeded and the seeds are in the source. Reported metrics are
computed on *pooled out-of-fold predictions*, so each structure is predicted exactly once per
protocol; the numbers are not averages over folds and carry no fold-selection variance.

**The language-model component needs no model access.** `src/llm_descriptors.py` contains the
verbatim elicitation prompt and the cached response. Nothing is fetched at runtime. The two
published tabulations used for the memorisation control are in the same file and are used only in
`03_validate_llm_descriptors.py`, never as model input.

**Source data.** The primary data are the published HOIP dataset of Kim, Huan, Krishnan and
Ramprasad, *Sci. Data* **4** (2017) 170057 (reference [47] in the manuscript). It is not
redistributed here; `src/00_fetch_source_data.sh` downloads it into `data/raw/`. Please cite the
original paper. No new density functional theory calculations were performed for this study.

**Single-core note.** `RandomForestRegressor` and `ExtraTreesRegressor` oversubscribe and stall when
`n_jobs=-1` on a single core, so all estimators are pinned to `n_jobs=1` in `evaluation.py`. Raise
this if more cores are available.
