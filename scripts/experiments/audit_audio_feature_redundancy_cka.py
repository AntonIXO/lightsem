from pathlib import Path
import json,numpy as np
j=json.loads(Path('/mnt/lightsem-data/experiments/lightsem/audio_light_feature_redundancy_20261015.json').read_text())
# Rebuild arrays by executing the source's setup without its output section
exec(Path('scripts/experiments/audit_audio_feature_redundancy.py').read_text().split("out={'rows'")[0])
rng=np.random.default_rng(7); idx=rng.choice(len(X),2000,False); Z=X[idx]
def cka(A,B):
 A=A-A.mean(0,keepdims=True); B=B-B.mean(0,keepdims=True); K=A@A.T; L=B@B.T; K=K-np.mean(K,0)-np.mean(K,1)[:,None]+K.mean(); L=L-np.mean(L,0)-np.mean(L,1)[:,None]+L.mean(); return float((K*L).sum()/np.sqrt((K*K).sum()*(L*L).sum()))
def offdiag(A):
 A=A-A.mean(0); A=A/(A.std(0)+1e-12); C=np.abs(A.T@A/len(A)); C=C[~np.eye(C.shape[0],dtype=bool)]; return {'mean_abs':float(C.mean()),'p50':float(np.quantile(C,.5)),'p95':float(np.quantile(C,.95)),'max':float(C.max())}
out={'sample_frames':len(X),'cka':{'muq_internal':cka(Z[:,:1024],Z[:,:1024]),'base92_internal':cka(Z[:,1024:],Z[:,1024:]),'muq_base92':cka(Z[:,:1024],Z[:,1024:]),'muq_timing':cka(Z[:,:1024],Z[:,1111:1116]),'muq_energy':cka(Z[:,:1024],Z[:,1032:1037]),'muq_onset':cka(Z[:,:1024],Z[:,1027:1032])},'within_group_offdiag':{'muq':offdiag(Z[:,:1024]),'base92':offdiag(Z[:,1024:])}}
Path('/mnt/lightsem-data/experiments/lightsem/audio_light_feature_redundancy_20261015_CKA.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
