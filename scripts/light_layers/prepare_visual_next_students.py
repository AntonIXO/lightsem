#!/usr/bin/env python3
"""Freeze paired untrained states and TRAIN main-target/data-draw identities."""
from __future__ import annotations
from dataclasses import asdict
import argparse,hashlib,json
from pathlib import Path
import torch
from lightsem.model.light_layers.model import LightLayersConfig,build_light_layers,count_trainable_parameters
from lightsem.model.light_layers.training import SourceStore,TrainingData,BatchSampler,seed_all
from lightsem.model.light_layers.visual import sha256,write_json
from scripts.light_layers.train_visual_next import state_digest,main_target_digest,checkpoint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    out=Path(freeze['output']);out.mkdir(exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads(Path(freeze['training_manifest']['path']).read_text())
    store=SourceStore(manifest)
    records=[store.visual(r['recording_id']) for r in manifest['records'] if r['partition']=='train']
    if len(records)!=91 or len({r.canonical_id for r in records})!=91:raise RuntimeError('Original91TRAIN families required')
    transforms=json.loads(Path(freeze['baseline_fit_report']).read_text())['frozen_transforms']
    data=TrainingData(store,records,transforms,'I_VR_P0');main_digest=main_target_digest(data)
    initialization={};samples={};weights={};reference=LightLayersConfig(visual_aux_dim=96,pose_aux_dim=0)
    for seed in (0,1,2):
        seed_all(seed,cuda=False);model=build_light_layers(reference,architecture='independent')
        assert count_trainable_parameters(model)==1284410
        initialization[str(seed)]=state_digest(model)
        metadata={'step':0,'seed':seed,'architecture':'independent','reference_model_config':asdict(reference),
            'model_config':asdict(model.cfg),'parameter_match':asdict(model.parameter_match),
            'scope':'Paired untrained control; no inference/fitting performed','preparation_freeze_sha256':sha256(args.freeze),
            'main_targets_sha256':main_digest,'initial_state_sha256':initialization[str(seed)]}
        path=out/f'seed{seed}_untrained.pt';checkpoint(path,model,metadata);weights[str(seed)]={'path':str(path),'sha256':sha256(path)}
        sampler=BatchSampler(records,seed);h=hashlib.sha256()
        for _ in range(6000):h.update(json.dumps(sampler.sample(4),separators=(',',':')).encode())
        samples[str(seed)]=h.hexdigest();del model
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(out/'REPORT.json',{'status':'paired_untrained_controls_and_TRAIN_invariants_frozen',
        'freeze_sha256':sha256(args.freeze),'initialization_sha256':initialization,'sample_sequence_sha256':samples,
        'untrained_checkpoints':weights,'main_targets_sha256':main_digest,'transforms':transforms,
        'train_families':[r.canonical_id for r in records],'train_records':91,'DEV_read':False,'TEST_read':False,
        'inference_performed':False,'optimizer_steps':0})
    print('PAIRED_STUDENTS_READY',main_digest,initialization,flush=True)


if __name__=='__main__':main()
