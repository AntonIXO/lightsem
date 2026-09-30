#!/usr/bin/env python3
"""One fixed checkpoint on31historicalDEV families with identical audio controls."""
from __future__ import annotations
import argparse,json,os,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.model.light_layers.training import SourceStore,load_checkpoint,RHYTHM_COLUMNS
from lightsem.model.light_layers.evaluation import (audio_window,freeze_labels,predict_window,nonrhythm_donor,verify_donor_tempos)
from lightsem.experiments.visual_next.scoring import score_color_window
from lightsem.model.light_layers.visual import sha256,write_json,write_arrays


def check_storage(freeze):
    root=Path(freeze['study_root'])
    if any(shutil.disk_usage(path).free<2**30 for path in (root,'/mnt')):raise RuntimeError('Free-space floor')
    total=sum(p.stat().st_size for d in root.iterdir() if d.is_dir() and d.name!='papers' for p in d.rglob('*') if p.is_file())
    if total>freeze['max_study_bytes']:raise RuntimeError('Study artifact allowance')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path);ap.add_argument('--checkpoint-id',required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    reference=freeze['reference_preparation']
    if freeze['files'].get(reference['path'])!=reference['sha256']:
        raise RuntimeError('Source-reference preparation is not bound to this evaluation')
    prepared=json.loads(Path(reference['path']).read_text())
    qualification_ref=freeze['representation_qualification']
    if freeze['files'].get(qualification_ref['path'])!=qualification_ref['sha256']:
        raise RuntimeError('Representation qualification is not bound to this evaluation')
    qualification=json.loads(Path(qualification_ref['path']).read_text())
    if prepared['status']!='source_references_and_audio_controls_frozen_before_predictions':
        raise RuntimeError('Source-reference preparation is incomplete')
    for key in ('training_manifest','transforms','donors','donor_tempos_bpm','TRAIN_color_prevalence'):
        if freeze[key]!=prepared[key]:raise RuntimeError('Frozen evaluation input drift: '+key)
    matches=[r for r in freeze['checkpoints'] if r['id']==args.checkpoint_id]
    if len(matches)!=1:raise RuntimeError('Checkpoint is outside frozen matrix')
    item=matches[0]
    if freeze['files'].get(item['path'])!=item['sha256']:raise RuntimeError('Checkpoint is not bound to evaluation')
    check_storage(freeze)
    out=Path(freeze['output'])/item['id'];out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(freeze['training_manifest']['path']).read_text());store=SourceStore(manifest)
    records=[store.visual(row['recording_id']) for row in manifest['records'] if row['partition']=='val']
    if len(records)!=31 or len({r.canonical_id for r in records})!=31:raise RuntimeError('Fixed31historicalDEV families required')
    transforms=freeze['transforms'];verify_donor_tempos(store,[r.recording_id for r in records],freeze['donor_tempos_bpm'])
    windows={r.recording_id:audio_window(store,r,transforms) for r in records}
    frozen_labels={r.recording_id:freeze_labels(r,transforms) for r in records}
    expected={row['recording_id']:row['reference_hash'] for row in prepared['DEV_references']}
    if {rid:labels.content_hash() for rid,labels in frozen_labels.items()}!=expected:
        raise RuntimeError('Reference labels or source support changed after preparation')
    rows=[];started=time.monotonic();torch.set_num_threads(2)
    with gpu_lock():
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        others=[x.strip() for x in active.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
        if others:raise RuntimeError('GPU is in use: '+','.join(others))
        model,metadata=load_checkpoint(item['path'],device='cuda');torch.cuda.reset_peak_memory_stats()
        if item['role'] in ('final','demonstration'):
            if metadata['step']!=6000:raise RuntimeError('Not a final6000-step checkpoint')
            if (metadata['arm'],metadata['seed'])!=(item['arm'],item['seed']):raise RuntimeError('Checkpoint arm/seed mislabeled')
            if metadata['main_targets_sha256']!=item['main_targets_sha256']:raise RuntimeError('Checkpoint target recipe mismatch')
        if item['role']=='untrained' and metadata['step']!=0:raise RuntimeError('Untrained control was trained')
        for r in records:
            rid=r.recording_id;x=windows[rid];donor=freeze['donors'][rid]
            if donor==rid or donor not in windows:raise RuntimeError('Invalid frozen donor')
            beat=np.zeros_like(x);beat[:,RHYTHM_COLUMNS]=x[:,RHYTHM_COLUMNS]
            inputs={'model':x,'donor':nonrhythm_donor(x,windows[donor]),'beat_only':beat}
            predictions={};arrays={}
            for condition,features in inputs.items():
                result=predict_window(model,features,requested_times10_s=r.times,source_times10_s=r.source_times)
                predictions[condition]=result['color']
                for key,value in result.items():arrays[condition+'_'+key]=value
            predictions['constant']=np.full(640,freeze['TRAIN_color_prevalence'])
            labels=frozen_labels[rid];score=score_color_window(labels,predictions)
            score.update(checkpoint_id=item['id'],arm=item['arm'],seed=item['seed'],checkpoint_role=item['role'],donor_recording_id=donor)
            path=out/(r.canonical_id+'.npz');write_arrays(path,arrays);score['predictions_sha256']=sha256(path)
            write_json(out/(r.canonical_id+'.json'),score);rows.append(score)
            check_storage(freeze)
            if time.monotonic()-started>freeze['per_checkpoint_max_seconds']:raise RuntimeError('Evaluation runtime allowance')
            if torch.cuda.max_memory_reserved()>10*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:
                raise RuntimeError('Evaluation memory allowance')
            print('EVAL',item['id'],len(rows),31,flush=True)
        peak=torch.cuda.max_memory_reserved()
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(out/'REPORT.json',{'status':'all31historicalDEV_conditions_scored','checkpoint':item,
        'freeze_sha256':sha256(args.freeze),'rows':rows,'elapsed_s':time.monotonic()-started,
        'reference_preparation':reference,
        'representation_qualification':qualification_ref,
        'original_representation_fit_gate_pass':qualification['original_fit_gate_pass'],
        'peak_reserved_bytes':peak,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scope':'Full64s automatic-observer proxy; no independent content labels, TEST, hyperparameter selection or baseline promotion.',
        'beat_control':'Keep8standardized rhythm columns; all remaining columns fixed to their TRAIN mean (zero after standardization).',
        'donor_control':'Frozen nearest-tempo donor nonrhythm columns plus recipient rhythm; limited intervention, not complete causal isolation.'})


if __name__=='__main__':main()
