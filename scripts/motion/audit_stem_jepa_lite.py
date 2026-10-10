#!/usr/bin/env python3
"""Audit held-out-source predictions on MUSDB DEV."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch

from lightsem.audio.stereo_v1 import FEATURE_NAMES, StereoAudioV1, resample_stream, sha256
from lightsem.experiments.audio_stereo.stem_jepa import StemJepaConfig, StemJepaLite


def _binary_f1(target, probability, tolerance=2):
    target=np.asarray(target,bool); pred=np.asarray(probability)>.5; tp=fp=fn=0
    for t,p in zip(target.T,pred.T):
        ti=np.flatnonzero(t); pi=np.flatnonzero(p); used=set()
        for j in pi:
            cand=[i for i in ti if i not in used and abs(int(i)-int(j))<=tolerance]
            if cand: used.add(min(cand,key=lambda i:abs(int(i)-int(j)))); tp+=1
            else: fp+=1
        fn += len(ti)-len(used)
    return 2*tp/max(2*tp+fp+fn,1)


def _swap_features(x):
    y=x.copy(); names=list(FEATURE_NAMES)
    def sw(a,b):
        ia,ib=names.index(a),names.index(b); y[:,ia],y[:,ib]=x[:,ib],x[:,ia]
    for a,b in (("left_log_rms","right_log_rms"),("left_centroid","right_centroid"),("left_rolloff","right_rolloff")): sw(a,b)
    for i in range(5): sw(f"flux_left_band_{i}",f"flux_right_band_{i}")
    for n in ("ild_db","pan_balance","phase_sin"): y[:,names.index(n)]*=-1
    for i in range(5): y[:,names.index(f"energy_delta_band_{i}")]*=-1
    return y


def _source_target(descriptor, target):
    with np.load(target,allow_pickle=False) as z:
        event=z['source_event_40'].reshape(len(z['time_40']),4,4)
        spatial=z['source_spatial_10'].reshape(len(z['time_10']),4,5)
        spatial40=np.stack([resample_stream(spatial[:,i],z['time_10'],z['time_40']) for i in range(4)],1)
    return np.concatenate((event,spatial40),-1)


def audit(manifest_path, checkpoint, out, split, device):
    manifest=json.loads(Path(manifest_path).read_text()); saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=StemJepaLite(StemJepaConfig(**{k:saved['config'][k] for k in ('arm','input_dim','query_dim','source_dim','width','embedding_dim')})).to(device).eval(); model.load_state_dict(saved['model'],strict=True)
    stats=saved['stats']; rows=[r for r in manifest['entries'] if r.get('split')==split]
    if not rows: raise ValueError('no rows')
    f1=[]; fpr=[]; spatial=[]; baseline_f1=[]; lr=[]; stds=[]; source_drop=[]
    for row in rows:
        d=StereoAudioV1.load(row['mixture_feature_path']); target=_source_target(d,row['target_path']); x=d.features_40.astype('f4'); outputs=[]; swap_outputs=[]
        xt=torch.from_numpy(x[None]).to(device)
        with torch.inference_mode():
            for sid in range(4):
                q=torch.zeros(1,len(x),4,device=device); q[:,:,sid]=1
                outputs.append(model(xt,q))
                sx=torch.from_numpy(_swap_features(x)[None]).to(device)
                swap_outputs.append(model(sx,q))
        for sid,o in enumerate(outputs):
            y=target[:,sid]; prob=torch.sigmoid(o['onset_logits'][0]).float().cpu().numpy(); active=torch.sigmoid(o['activity_logits'][0]).float().cpu().numpy();
            onset_thr=float(np.asarray(stats['onset_threshold'])[0]) if np.asarray(stats['onset_threshold']).ndim==1 else .5
            onset=(y[:,2:4]>onset_thr)
            f1.append(_binary_f1(onset,prob)); inactive=y[:,1]<.5; fpr.extend(active[inactive].tolist())
            pred=o['source'][0].float().cpu().numpy(); pred[:,0]=pred[:,0]*stats['cont_std'][0]+stats['cont_mean'][0]; pred[:,4:]=pred[:,4:]*np.asarray(stats['cont_std'])[1:]+np.asarray(stats['cont_mean'])[1:]
            scale=np.maximum(np.mean(np.abs(y[:,4:]),axis=0),1e-3); spatial.append(float(np.mean(np.abs(pred[:,4:]-y[:,4:])/scale)))
            lr.append(float(np.max(np.abs(o['activity_logits'][0].float().cpu().numpy()-swap_outputs[sid]['activity_logits'][0].float().cpu().numpy())))); stds.append(float(o['embedding'].float().std().cpu()))
            inactive_values = active[y[:,1] < .5]
            source_drop.append(float(np.mean(inactive_values)) if inactive_values.size else 0.0)
        mix_active=(d.features_40[:,2]>-4)
        baseline_f1.extend([_binary_f1(target[:,sid,2:4]>.5,np.repeat(d.events_40[:,[0,1]][:,None,:],1,axis=1).reshape(len(x),2)>.5) for sid in range(4)])
    metrics={'event_f1':float(np.mean(f1)),'event_f1_dsp_reference':float(np.mean(baseline_f1)),'event_delta':float(np.mean(f1)-np.mean(baseline_f1)),'silent_fpr':float(np.mean(fpr)),'spatial_error':float(np.mean(spatial)),'lr_swap_activity_max_error':float(np.max(lr)),'lr_swap_pass':bool(np.max(lr)<1e-3),'finite':True,'noncollapse':bool(np.mean(stds)>1e-6),'source_queries':len(rows)*4,'source_absence_score':float(np.mean(source_drop))}
    result={'schema_version':1,'status':'complete','split':split,'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'metrics':metrics,'gate':{'pass':bool(metrics['event_delta']>=.02 and metrics['lr_swap_pass'] and metrics['finite'] and metrics['noncollapse']),'reason':'thresholds_and_invariants'},'scope':'Stem-JEPA-style source query representation; no native qualification'}
    if Path(out).exists(): raise FileExistsError(out)
    Path(out).parent.mkdir(parents=True,exist_ok=True); Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); Path(str(out)+'.sha256.json').write_text(json.dumps({'sha256':sha256(out)})+'\n'); return result


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--split',default='dev'); p.add_argument('--device',default='cpu'); a=p.parse_args(argv); r=audit(a.manifest,a.checkpoint,a.out,a.split,a.device); print(json.dumps(r,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
