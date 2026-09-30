#!/usr/bin/env python3
"""Controlled TRAIN-only hue/layout retention and fixed null-image diagnostics."""
from __future__ import annotations
import argparse,json,os,resource,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.experiments.visual_next import models
from lightsem.experiments.visual_next.features import ARMS
from lightsem.experiments.visual_next.observation import FrameObservation
from lightsem.experiments.visual_next.probes import HUE_TURNS,controlled_background,explicit_descriptors,family_probe_skill
from lightsem.experiments.visual_next.projection import project
from lightsem.model.light_layers.visual import sha256,write_arrays,write_json,ui_mask,retained_mask


def encoded_features(rgb,keep,pretrained,random):
    obs=FrameObservation(rgb);r1=obs.grid(keep)
    r0=obs.target(keep).rich.astype(np.float16).astype(np.float32)
    if r0[:72].sum()>0:r0[:72]/=r0[:72].sum()
    p=next(models.dino_grid(pretrained,[rgb]));r=next(models.dino_grid(random,[rgb]))
    # Same float16 storage boundary as the real teacher extractor.
    grid=r1.astype(np.float16).astype(np.float32)
    p=p.astype(np.float16).astype(np.float32).ravel();r=r.astype(np.float16).astype(np.float32).ravel()
    r2=np.r_[grid,p];temporal=np.r_[r2,np.zeros_like(r2),np.zeros_like(r2)]
    arms={'R0':r0,'R1':grid,'Rrand':np.r_[grid,r],'R2':r2,
          'R3':temporal,'R3-shuffle':temporal,'R3-reverse':temporal}
    return arms,r1,p,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    out=Path(freeze['output']);out.mkdir(exist_ok=False)
    if len(freeze['families'])!=12 or any(r['partition']!='train' for r in freeze['families']):raise RuntimeError('Original12 TRAIN calibration families required')
    states={};pca_meta={}
    for arm in ARMS:
        root=Path(freeze['projection_root'])
        with np.load(root/(arm+'_PCA.npz'),allow_pickle=False) as z:states[arm]={k:z[k].copy() for k in z.files}
        pca_meta[arm]=json.loads((root/(arm+'_PCA.json')).read_text())
        if pca_meta[arm]['fit_partition']!='train' or len(pca_meta[arm]['fit_families'])!=91:raise RuntimeError('PCA fitting identity')
    torch.set_num_threads(2);rows=[];excluded=[];vectors={arm:[] for arm in ARMS};grids=[];raw_pretrained=[];raw_random=[]
    started=time.monotonic()
    with gpu_lock():
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        others=[x.strip() for x in active.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
        if others:raise RuntimeError('GPU is in use: '+','.join(others))
        torch.cuda.init();torch.cuda.reset_peak_memory_stats();pretrained=models.load_dino();random=models.load_dino(pretrained=False)
        for family in freeze['families']:
            cid=family['canonical_id'];mask_file=Path(freeze['shared_root'])/cid/'mask_controls.npz'
            with np.load(mask_file,allow_pickle=False) as z:
                indices=z['sample_indices'];packed=z['sam2_packed'];present=z['sam2_present'];width,height=z['geometry_wh']
                source_times=z['source_times_s']
            for j,index in enumerate(indices):
                if not present[j]:excluded.append({'family':cid,'index':int(index),'reason':'original_source_SAM_observation_missing'});continue
                still=family['stills'][j]
                if still['source_time_s']!=source_times[j] or int(still['offset_s']*10)!=index:raise RuntimeError('Probe source timestamp mismatch')
                rgb=np.asarray(Image.open(still['path']).convert('RGB'))
                if rgb.shape!=(height,width,3):raise RuntimeError('Probe still/mask geometry mismatch')
                foreground=np.unpackbits(packed[j],axis=-1)[:,:width].astype(bool)
                keep=retained_mask(foreground,ui_mask((int(width),int(height))))
                for mirror in (False,True):
                    for turns in HUE_TURNS:
                        image=controlled_background(rgb,keep,turns,mirror)
                        arm_values,raw_grid,p,r=encoded_features(image,keep,pretrained,random)
                        grids.append(raw_grid);raw_pretrained.append(p.astype(np.float16));raw_random.append(r.astype(np.float16))
                        for arm in ARMS:
                            y=project(arm_values[arm][None],states[arm])[0]
                            if not np.isfinite(y).all():raise RuntimeError('Nonfinite projected probe')
                            vectors[arm].append(y)
                        rows.append({'family':cid,'sample_index':int(index),'hue_turns':turns,'mirror':mirror})
                if time.monotonic()-started>freeze['max_gpu_seconds']:raise RuntimeError('Frozen probe runtime allowance')
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Probe RAM allowance')
            print('RETENTION_PROBE',cid,'rows',len(rows),flush=True)
        # Deliberately artificial flat canvases expose positional/nuisance features.
        flat=[]
        for name,color in [('black',(0,0,0)),('gray',(128,128,128)),('red',(255,0,0)),('blue',(0,0,255))]:
            image=np.tile(np.asarray(color,np.uint8),(224,224,1));keep=np.ones((224,224),bool)
            arm_values,grid,p,r=encoded_features(image,keep,pretrained,random)
            flat.append({'name':name,'pretrained_patch_position_variance':float(p.reshape(16,384).var(0).mean()),
                         'random_patch_position_variance':float(r.reshape(16,384).var(0).mean()),
                         'projected_norms':{arm:float(np.linalg.norm(project(arm_values[arm][None],states[arm]))) for arm in ARMS}})
        peak=torch.cuda.max_memory_reserved();gpu_seconds=time.monotonic()-started
        if peak>10*2**30:raise RuntimeError('VRAM cap')
    projected={arm:np.asarray(v,np.float32) for arm,v in vectors.items()};grid=np.asarray(grids,np.float32)
    groups=np.asarray([r['family'] for r in rows]);descriptors=explicit_descriptors(grid)
    if len(np.unique(groups))!=12:raise RuntimeError('A fixed calibration family has no usable probe source')
    arrays={**projected,'raw_R1':grid,'raw_pretrained':np.asarray(raw_pretrained),'raw_random':np.asarray(raw_random),
            **{'target_'+k:v for k,v in descriptors.items()}}
    write_arrays(out/'PROBES.npz',arrays);write_json(out/'ENROLLMENT.json',{'rows':rows,'excluded':excluded})
    skills={};ratios={}
    with threadpool_limits(limits=2):
        for name,target in descriptors.items():
            skills[name]={'R1_raw':family_probe_skill(grid,target,groups)}
            for arm in ARMS:skills[name][arm]=family_probe_skill(projected[arm],target,groups)
            base=skills[name]['R1_raw']['baseline_corrected_skill']
            score=skills[name]['R3']['baseline_corrected_skill']
            ratios[name]=score/base if base is not None and base>0 and score is not None else None
    distances={arm:[] for arm in ARMS}
    for cid in np.unique(groups):
        by_family=[]
        for sample in sorted({r['sample_index'] for r in rows if r['family']==cid}):
            ids=[i for i,r in enumerate(rows) if r['family']==cid and r['sample_index']==sample and not r['mirror']]
            assert len(ids)==4
            by_family.append(ids)
        for arm in ARMS:
            ds=[np.linalg.norm(projected[arm][ids[1:]]-projected[arm][ids[0]],axis=1) for ids in by_family]
            distances[arm].append(np.mean(ds,axis=0))
    mean_distances={arm:np.mean(v,axis=0).tolist() for arm,v in distances.items()}
    r3_rank=pca_meta['R3']['pre_normalization_effective_rank'];random_rank=pca_meta['Rrand']['pre_normalization_effective_rank']
    gates={'hue_retention':ratios['hue'] is not None and ratios['hue']>=.9,
           'layout_retention':ratios['layout'] is not None and ratios['layout']>=.9,
           'monotone_R3_hue_distance':bool(np.all(np.diff(mean_distances['R3'])>0)),
           'no_severe_rank_collapse':r3_rank>=.5*random_rank and r3_rank>1}
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(out/'REPORT.json',{'status':'controlled_retention_probes_complete','freeze_sha256':sha256(args.freeze),
        'families':12,'rows':len(rows),'source_mask_exclusions':excluded,'skills':skills,'R3_retention_ratios':ratios,
        'hue_distances_by_arm':mean_distances,'flat_controls':flat,'gates':gates,'original_fit_gate_pass':all(gates.values()),
        'rank_guard':{'R3':r3_rank,'Rrand':random_rank,'minimum_ratio':.5,'before_coordinate_standardization':True},
        'gpu_seconds':gpu_seconds,'elapsed_s':time.monotonic()-started,'peak_reserved_bytes':peak,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'arrays_sha256':sha256(out/'PROBES.npz'),'enrollment_sha256':sha256(out/'ENROLLMENT.json'),
        'scope':'TRAIN-only synthetic perturbation calibration. Explicit color descriptors are supplied information; reconstruction is not an independent quality gain. Static repeated-frame controls do not test temporal ordering.','test_access':False})


if __name__=='__main__':main()
