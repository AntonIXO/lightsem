#!/usr/bin/env python3
"""Fit the seven fixed TRAIN projections and apply them with common support."""
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from lightsem.experiments.visual_next.features import ARMS
from lightsem.experiments.visual_next.data import DIMENSIONS,arm_features,train_factory
from lightsem.experiments.visual_next.projection import fit_pca,project
from lightsem.model.light_layers.visual import sha256,write_json,write_arrays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    mode=freeze.get('mode','fit_apply')
    if mode not in ('fit','apply','fit_apply'):raise ValueError('Unknown projection phase')
    root=Path(freeze['output'])
    if mode=='apply':
        if not root.is_dir() or not (root/'FROZEN_TRAIN_TRANSFORMS.json').exists():raise RuntimeError('Missing frozen TRAIN transforms')
    else:root.mkdir(exist_ok=False)
    shared=freeze['shared_root'];rows=freeze['records'];train=[r for r in rows if r['partition']=='train']
    expected=91 if mode=='fit' else 122
    if len(train)!=91 or len(rows)!=expected:raise RuntimeError('Original phase-specific TRAIN/DEV cohort required')
    if len({r['canonical_id'] for r in rows})!=expected:raise RuntimeError('Canonical identity duplication')
    states={};summaries={};started=time.monotonic()
    with threadpool_limits(limits=2):
        for arm in ARMS:
            if mode=='apply':
                metadata=json.loads((root/(arm+'_PCA.json')).read_text())
                if metadata['fit_partition']!='train' or metadata['fit_families']!=[r['canonical_id'] for r in train]:
                    raise RuntimeError('PCA TRAIN identity drift')
                if metadata['arrays_sha256']!=sha256(root/(arm+'_PCA.npz')):raise RuntimeError('PCA parameter drift')
                with np.load(root/(arm+'_PCA.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
                states[arm]=state;summaries[arm]=metadata
                continue
            begin=time.monotonic()
            state=fit_pca(train_factory(train,shared,arm),DIMENSIONS[arm],components=96,
                          oversample=32,power_iterations=2,seed=20260929)
            arrays={k:v for k,v in state.items() if isinstance(v,np.ndarray)}
            metadata={k:v for k,v in state.items() if not isinstance(v,np.ndarray)}
            metadata.update(arm=arm,fit_partition='train',fit_families=[r['canonical_id'] for r in train],
                            seconds=time.monotonic()-begin)
            write_arrays(root/(arm+'_PCA.npz'),arrays)
            metadata['arrays_sha256']=sha256(root/(arm+'_PCA.npz'))
            write_json(root/(arm+'_PCA.json'),metadata)
            states[arm]=state;summaries[arm]=metadata
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('RAM cap')
            if time.monotonic()-started>freeze['max_cpu_seconds']:raise RuntimeError('CPU projection allowance exceeded')
            print('PCA_FIT',arm,state['count'],state['retained_variance_fraction'],metadata['seconds'],flush=True)
        # All TRAIN-fitted states are immutable before any DEV application.
        if mode!='apply':
            write_json(root/'FROZEN_TRAIN_TRANSFORMS.json',{'arms':summaries,
                'transform_files':{str((root/(arm+'_PCA.npz')).resolve()):sha256(root/(arm+'_PCA.npz')) for arm in ARMS},
                'dev_fitted':False,'scope':'unsupervised TRAIN projections; no quality/retention verdict yet'})
        if mode=='fit':
            for p,d in freeze['files'].items():assert sha256(p)==d,p
            write_json(root/'FIT_REPORT.json',{'status':'all7_TRAIN_projections_frozen','states':summaries,
                'freeze_sha256':sha256(args.freeze),'elapsed_s':time.monotonic()-started,'dev_fitted':False,
                'DEV_read':False,'test_access':False,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
            return
        artifacts=[]
        for row in rows:
            output={};support=None;errors={}
            for arm in ARMS:
                x,valid=arm_features(row,shared,arm)
                if support is None:support=valid
                elif not np.array_equal(valid,support):raise RuntimeError('Arm support mismatch')
                y=np.zeros((len(x),96),np.float32);y[valid]=project(x[valid],states[arm])
                output[arm]=y.astype(np.float16)
                errors[arm]=float(np.max(np.abs(output[arm].astype(np.float32)-y)))
                if not np.isfinite(output[arm]).all():raise RuntimeError('Invalid float16 projected target')
            output['valid']=support
            path=root/(row['canonical_id']+'.npz');write_arrays(path,output)
            artifacts.append({'canonical_id':row['canonical_id'],'partition':row['partition'],
                              'path':str(path),'sha256':sha256(path),'common_valid_frames':int(support.sum()),
                              'max_float16_roundoff_by_arm':errors})
            print('PCA_APPLIED',row['canonical_id'],flush=True)
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(root/'REPORT.json',{'status':'train_projections_frozen_and_applied','freeze_sha256':sha256(args.freeze),
        'states':summaries,'artifacts':artifacts,'elapsed_s':time.monotonic()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'dev_fitted':False,'test_access':False,'representation_retention_gate':'pending separate controlled probes'})


if __name__=='__main__':main()
