"""
Stage 02 - merge parsed structures with the tabulated GGA/HSE06
properties into the master dataset.

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, RAW
import pandas as pd, numpy as np
s=pd.read_csv(DATA / 'hoip_structures.csv')
p=pd.read_excel(RAW / 'HOIPs-Bandgap-Data-set-main' / 'HOIP_Extracted_dataset.xlsx')
p['entry']=p.Filename.str.replace('.cif','',regex=False).astype(str).str.zfill(4)
s['entry']=s.entry.astype(str).str.zfill(4)
df=s.merge(p.drop(columns=['Filename']),on='entry',how='inner')
df=df.rename(columns={'Bandgap GGA':'Eg_GGA','Bandgap HSE06':'Eg_HSE','Volume of Unit Cell':'V_cell',
  'Density':'rho','Atomization Energy':'E_atom','Relative Energy1':'Erel1','Relative Energy2':'Erel2',
  'Dielectric Constant Electronic':'eps_el','Dielectric Constant Ionic':'eps_ion',
  'Dielectric Constant Total':'eps_tot','Refractive Index':'n_ref'})
df['composition']=df.A_cation+'_'+df.B+'_'+df.X
df['dEg']=df.Eg_HSE-df.Eg_GGA
print(df.shape, df.composition.nunique())
print(df[['Eg_GGA','Eg_HSE','dEg','eps_el','E_atom','Erel1','V_cell','rho']].describe().round(3).to_string())
print('\nHSE gap by halide:'); print(df.groupby('X').Eg_HSE.agg(['mean','std','count']).round(3))
print('\nHSE gap by B:'); print(df.groupby('B').Eg_HSE.agg(['mean','std','count']).round(3))
print('\npolymorphs per composition:', df.groupby('composition').size().describe().round(2).to_dict())
# within-composition vs total variance of target
wg=df.groupby('composition').Eg_HSE.transform('mean')
print('\nTotal var Eg_HSE=%.4f  within-composition var=%.4f  -> R2 ceiling from composition alone=%.4f'%(
    df.Eg_HSE.var(), (df.Eg_HSE-wg).var(), 1-(df.Eg_HSE-wg).var()/df.Eg_HSE.var()))
df.to_csv(DATA / DATA / 'hoip_master.csv',index=False)
