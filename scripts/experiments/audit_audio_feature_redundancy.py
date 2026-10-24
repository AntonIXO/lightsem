"""Pure-numpy authenticated MuQ/Base92 redundancy audit."""
from pathlib import Path
import json, hashlib, numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013'); VIS=Path('/mnt/lightsem-data/cache/beatsaver_visual'); LIGHT=Path('/mnt/lightsem-data/cache/beatsaver_light')
IDS=('1023','10dcc','1181f','11b01','1672d','210e3','21c3a','3b4ab')
def one(root,s):
 p=sorted(root.glob(f'*={s}.npz')); assert len(p)==1; return p[0]
def near(v,t,q):
 r=np.searchsorted(t,q,'left').clip(0,len(t)-1); l=np.maximum(r-1,0); return v[np.where(np.abs(t[l]-q)<=np.abs(t[r]-q),l,r)]
Xs=[]
for s in IDS:
 meta=json.loads((ROOT/f'body_{s}'/'body.json').read_text())['metadata']; tp=Path(meta['audio']['timeline']); bp=Path(meta['audio']['base92'])
 with np.load(tp) as z: muq=z['muq'].astype('f4'); mt=z['muq_time'].astype('f4')
 with np.load(bp) as z: base=z['x30'].astype('f4'); bt=z['time30'].astype('f4')
 with np.load(ROOT/f'body_{s}'/'body.npz') as z: n=len(z['world'])
 n=min(n,int(np.floor(float(mt[-1])*30))+1)
 q=np.arange(n,dtype='f4')/30; Xs.append(np.concatenate([near(muq,mt,q),near(base,bt,q)],1))
X=np.concatenate(Xs); rng=np.random.default_rng(0); X=X[rng.choice(len(X),min(50000,len(X)),False)].astype('f8')
def cka(A,B):
 A=A-A.mean(0); B=B-B.mean(0); A=A/(A.std(0)+1e-12); B=B/(B.std(0)+1e-12); K=A@A.T; L=B@B.T; return float((K*L).sum()/np.sqrt((K*K).sum()*(L*L).sum()))
def corr(A,B):
 A=(A-A.mean(0))/(A.std(0)+1e-12); B=(B-B.mean(0))/(B.std(0)+1e-12); C=np.abs(A.T@B/len(A)); return {'mean_abs':float(C.mean()),'p50':float(np.quantile(C,.5)),'p95':float(np.quantile(C,.95)),'max':float(C.max())}
def stats(name,A,B): return {'group':name,'corr':corr(A,B),'cka':cka(A[rng.choice(len(A),min(2000,len(A)),False)],B[rng.choice(len(B),min(2000,len(B)),False)])}
# matched sample indices for CKA are redrawn independently above only same lengths; use correlation as primary.
out={'rows':list(IDS),'frames':len(X),'groups':{
 'muq_internal':corr(X[:,:1024],X[:,:1024]),'base92_internal':corr(X[:,1024:],X[:,1024:]),'muq_base92':corr(X[:,:1024],X[:,1024:]),
 'muq_timing':corr(X[:,:1024],X[:,1111:1116]),'muq_energy':corr(X[:,:1024],X[:,1032:1037]),'muq_onset':corr(X[:,:1024],X[:,1027:1032]),
 'base_energy_timing':corr(X[:,1032:1037],X[:,1111:1116]),'base_onset_timing':corr(X[:,1027:1032],X[:,1111:1116])}}
Path('/mnt/lightsem-data/experiments/lightsem/audio_light_feature_redundancy_20261015.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
