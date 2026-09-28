"""
Stage 01 - parse the 1,346 relaxed CIF files into composition,
named A-site cation and lattice parameters.

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
from paths import DATA, RAW
import re, glob, os, numpy as np, pandas as pd
from collections import Counter

rows=[]
for f in sorted(glob.glob(str(RAW / 'HOIPs-Bandgap-Data-set-main' / 'cifs' / 'cif_merge' / '*.cif'))):
    txt=open(f).read()
    eid=os.path.basename(f).replace('.cif','')
    g=lambda k: float(re.search(rf'_cell_{k}\s+([-\d\.Ee+]+)',txt).group(1))
    a,b,c=g('length_a'),g('length_b'),g('length_c')
    al,be,ga=g('angle_alpha'),g('angle_beta'),g('angle_gamma')
    # atoms
    els=re.findall(r'^\s*[A-Za-z]+\d+\s+([A-Z][a-z]?)\s+[-\d\.]',txt,re.M)
    cnt=Counter(els)
    lab=re.search(r'# Label:\s+(.+)',txt)
    lab=lab.group(1).strip() if lab else ''
    src=re.search(r'# Organic cation source:\s+(.+)',txt)
    rows.append(dict(entry=eid, a=a,b=b,c=c,alpha=al,beta=be,gamma=ga,
                     label=lab, natoms=len(els), comp=dict(cnt),
                     src=(src.group(1).strip() if src else '')))
df=pd.DataFrame(rows)
# volume from cell params
r=np.radians
ca,cb,cg=np.cos(r(df.alpha)),np.cos(r(df.beta)),np.cos(r(df.gamma))
df['cell_vol']=df.a*df.b*df.c*np.sqrt(1-ca**2-cb**2-cg**2+2*ca*cb*cg)
# B and X
for B in ['Ge','Sn','Pb']:
    pass
df['B']=df.comp.apply(lambda d: next((e for e in ['Ge','Sn','Pb'] if e in d), None))
df['X']=df.comp.apply(lambda d: next((e for e in ['F','Cl','Br','I'] if e in d), None))
df['nB']=[d.get(B,0) for d,B in zip(df.comp,df.B)]
df['nX']=[d.get(X,0) for d,X in zip(df.comp,df.X)]
df['nC']=df.comp.apply(lambda d:d.get('C',0)); df['nN']=df.comp.apply(lambda d:d.get('N',0))
df['nH']=df.comp.apply(lambda d:d.get('H',0)); df['nS']=df.comp.apply(lambda d:d.get('S',0))
df['nO']=df.comp.apply(lambda d:d.get('O',0))
# A-cation = label minus " <Metal> <Halide>"
def acat(l):
    return re.sub(r'\s+(Germanium|Tin|Lead)\s+(Fluoride|Chloride|Bromide|Iodide)\s*$','',l).strip()
df['A_cation']=df.label.apply(acat)
print(df.shape)
print('B:',df.B.value_counts().to_dict())
print('X:',df.X.value_counts().to_dict())
print('n A-cations:',df.A_cation.nunique())
print(sorted(df.A_cation.unique()))
print('compositions:',df.groupby(['A_cation','B','X']).ngroups)
print(df.head(3).to_string()[:900])
df.drop(columns=['comp']).to_csv(DATA / 'hoip_structures.csv',index=False)
