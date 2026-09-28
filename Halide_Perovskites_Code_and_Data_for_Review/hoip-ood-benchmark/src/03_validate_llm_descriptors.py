"""
Stage 03 - memorisation control: compare the elicited A-site radii
with two independent literature tabulations (Section 3.6).

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from llm_descriptors import LLM_DESCRIPTORS, LIT_RADII_KIESLICH, LIT_RADII_DFT_ISOCHARGE

rows=[]
for c,d in LLM_DESCRIPTORS.items():
    rows.append(dict(cation=c, llm=d['r_eff_pm'],
                     kies=LIT_RADII_KIESLICH.get(c,np.nan),
                     dft=LIT_RADII_DFT_ISOCHARGE.get(c,np.nan)))
v=pd.DataFrame(rows).sort_values('llm')
print(v.to_string(index=False))
for ref in ['kies','dft']:
    m=v.dropna(subset=[ref])
    mae=np.abs(m.llm-m[ref]).mean(); mape=100*np.abs(m.llm-m[ref]).div(m[ref]).mean()
    r=pearsonr(m.llm,m[ref]); rho=spearmanr(m.llm,m[ref])
    print(f'\n[{ref}] n={len(m)}  MAE={mae:.1f} pm  MAPE={mape:.2f}%  Pearson r={r[0]:.4f} (p={r[1]:.2e})  Spearman rho={rho[0]:.4f} (p={rho[1]:.2e})')
# cross-check the two literature sets against each other (systematic offset)
m=v.dropna(subset=['kies','dft'])
print(f'\n[lit-vs-lit] n={len(m)} MAE={np.abs(m.kies-m.dft).mean():.1f} pm  Spearman rho={spearmanr(m.kies,m.dft)[0]:.4f}')
v.to_csv(DATA / DATA / 'llm_radii_validation.csv',index=False)
