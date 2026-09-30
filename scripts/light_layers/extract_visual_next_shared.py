#!/usr/bin/env python3
"""One decoded-frame pass for H2 masks and H3 spatial/frozen-teacher targets.

TRAIN features are stored for the later frozen PCA; no fitting occurs here.
Model predictions are exploratory until independent annotations are available.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
import json,resource,shutil,time
from pathlib import Path
import numpy as np
import torch

from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.experiments.visual_next import models
from lightsem.experiments.visual_next.extraction import AnchorSchedule,TemporalMasks,FlowControl,cut_reset_indices
from lightsem.experiments.visual_next.features import spatial_color_grid
from lightsem.experiments.visual_next.observation import FrameObservation
from lightsem.model.light_layers import targets
from lightsem.model.light_layers.visual import (sha256,write_json,write_arrays,iter_rgb_frames,
    load_pose,boxes_at_time,transform_boxes,SAM2FrameSegmenter,VisualConfig,retained_mask,ui_mask,_frame_arrays)
from scripts.light_layers.extract_visual_next_cuts import native_frames,positive_intervals


class StageClock:
    def __init__(self,limits):self.seconds={'H1_H2':0.,'H3_features':0.};self.limits=limits
    @contextmanager
    def measure(self,stage):
        started=time.monotonic()
        try:yield
        finally:
            self.seconds[stage]+=time.monotonic()-started
            if self.seconds[stage]>self.limits[stage]:raise RuntimeError(stage+' frozen runtime allowance exceeded')


def save_target(out,i,target,previous,retained):
    out['hs_histogram'][i],out['rich96'][i]=target.hs_histogram,target.rich
    out['value_summary'][i]=(target.value.mean,target.value.std,target.value.p90,target.value.maximum)
    for k in ('color_valid','quadrant_color_valid','retained_pixels','valid_pixels','retained_fraction','valid_fraction','chromatic_fraction'):
        out[k][i]=getattr(target,k)
    out['value_valid'][i]=target.value.valid;out['frame_valid'][i]=True
    if previous is not None:
        out['mask_activity_known'][i]=True;out['mask_activity'][i]=np.mean(previous!=retained)


def finish_mode(data,mapping):
    novelty=targets.chromatic_novelty(mapping['source_times_s'],data['hs_histogram'],data['color_valid'],
        retained_fraction=data['retained_fraction'],chromatic_fraction=data['chromatic_fraction'],
        mask_activity=data['mask_activity'])
    result={k:v.astype(np.float16) if v.dtype.kind=='f' else v for k,v in data.items()}
    for k,v in vars(novelty).items():
        if isinstance(v,np.ndarray):result['novelty_'+k]=v.astype(np.float16) if v.dtype.kind=='f' and k!='times_s' else v
    result.update(mapping)
    known=data['mask_activity_known'];times=mapping['source_times_s']
    activity_known=np.zeros(len(times),bool)
    for i,t in enumerate(times):
        edges=(times>t-.3+1e-10)&(times<t+.3-1e-10)
        activity_known[i]=novelty.window_eligible[i] and bool(known[edges].all())
    result['novelty_window_mask_activity_known']=activity_known
    return result


def check_space(root,max_bytes=3*2**30):
    if any(shutil.disk_usage(p).free<2**30 for p in (root,'/mnt')):raise RuntimeError('Free-space floor')
    cache=Path('/root/lightsem/runs/visual_next_20260929')
    total=sum(p.stat().st_size for d in cache.iterdir() if d.is_dir() and d.name!='papers' for p in d.rglob('*') if p.is_file())
    if total>max_bytes:raise RuntimeError('Study cache exceeds frozen byte allowance')
    return total


def extract(row,out,freeze,loaded,clock):
    segmenter,cutie,cfg,dino,random_dino,transnet=loaded
    started=time.monotonic();stage_before=clock.seconds.copy()
    cap=json.loads(Path(row['capture_receipt']['path']).read_text())
    n=freeze['frames_per_record'];g=cap['geometry'];out.mkdir()
    with clock.measure('H1_H2'):
        with np.load(Path(row['capture_receipt']['path']).parent/'sam2.npz',allow_pickle=False) as z:
            mapping={k:z[k][:n].copy() for k in ('source_times_s','requested_times_s','source_pts','frame_indices')}
            old_rich=z['rich96'][:n].astype(np.float32)
            valid=z['frame_valid'][:n].copy()
        if row.get('cut_traces'):
            with np.load(row['cut_traces']['path'],allow_pickle=False) as z:edits=z['edit_intervals_s'].copy()
        else:
            video,times,indices=native_frames(row);pred=models.transnet_predict(transnet,video)
            edits=positive_intervals(times,pred[:,0]>.5)
            write_arrays(out/'new_cut_traces.npz',{'native_source_times_s':times,'native_frame_indices':indices,
                'transnet_single_many':pred,'edit_intervals_s':edits})
            del video,pred
        boxes,pose_times,_=load_pose(cap['input'])
        ui=ui_mask(g['decode_wh'])
        schedule=AnchorSchedule(cut_reset_indices(mapping['source_times_s'],edits))
        tracker=TemporalMasks(cutie,cfg);flow=FlowControl();lag=deque(maxlen=10)
        data={mode:_frame_arrays(n) for mode in ('cutie','hold','lag_cutie')}
        previous={k:None for k in data};mask_indices=[i for i in range(0,n,70)]
        masks={k:[] for k in ('sam2','cutie','hold','lag_cutie')};mask_present={k:[] for k in masks}
        anchors=[];missing_refresh=[];flow_values=np.zeros((n,2),np.float32);flow_known=np.zeros((n,2),bool)
        r1=np.zeros((n,1200),np.float16);dp=np.zeros((n,16,384),np.float16);dr=np.zeros_like(dp)
        max_baseline_error=0.;max_quantization_error=0.
        frames=iter_rgb_frames(row['source_video']['path'],mapping,g['decode_wh'])
    try:
        for i in range(n):
            with clock.measure('H1_H2'):
                rgb=next(frames);foreground=None;observer=FrameObservation(rgb)
                if valid[i]:
                    original_prompts,_,_=boxes_at_time(boxes,pose_times,mapping['source_times_s'][i])
                    prompts=transform_boxes(original_prompts,g['original_wh'],g['decode_wh'])
                    foreground,_=segmenter(rgb,prompts)
                    retained=retained_mask(foreground,ui)
                    fresh=observer.target(retained)
                    err=float(np.abs(fresh.rich-old_rich[i]).max());max_baseline_error=max(max_baseline_error,err)
                    if err>freeze['cached_baseline_max_abs_error']:
                        raise RuntimeError(f'SAM2 replay differs from original quantized observation at{i}: {err}')
                reset,anchor,missing=schedule.step(i,foreground is not None)
                if anchor:anchors.append(i)
                if missing:missing_refresh.append(i)
                predicted,held=tracker.step(rgb,foreground if anchor else None,reset=reset)
                delayed=lag[0] if len(lag)==10 else None
                frame_masks={'sam2':foreground,'cutie':predicted,'hold':held,'lag_cutie':delayed}
                lag.append(None if predicted is None else predicted.copy())
                for mode in data:
                    m=frame_masks[mode]
                    if m is None:previous[mode]=None;continue
                    keep=retained_mask(m,ui);t=observer.target(keep)
                    save_target(data[mode],i,t,previous[mode],keep);previous[mode]=keep
                fv=flow.step(rgb,predicted);flow_known[i]=np.isfinite(fv);flow_values[i]=np.nan_to_num(fv)
                if i in mask_indices:
                    for mode,m in frame_masks.items():
                        masks[mode].append(np.packbits(np.zeros(rgb.shape[:2],bool) if m is None else m,axis=-1))
                        mask_present[mode].append(m is not None)
            with clock.measure('H3_features'):
                if valid[i]:
                    grid=observer.grid(retained)
                    pretrained=next(models.dino_grid(dino,[rgb]));random=next(models.dino_grid(random_dino,[rgb]))
                    if not all(np.isfinite(v).all() for v in (grid,pretrained,random)):raise RuntimeError('Nonfinite teacher feature')
                    r1[i]=grid;dp[i]=pretrained;dr[i]=random
                    max_quantization_error=max(max_quantization_error,float(np.max(np.abs(dp[i].astype(float)-pretrained))),
                                               float(np.max(np.abs(dr[i].astype(float)-random))))
            if (i+1)%80==0 or i+1==n:
                if torch.cuda.max_memory_reserved()>10*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:
                    raise RuntimeError('Memory cap')
                check_space(out,freeze.get('max_study_bytes',3*2**30))
                print('SHARED_FRAME',row['canonical_id'],i+1,n,clock.seconds,flush=True)
    finally:frames.close()
    artifacts={}
    with clock.measure('H1_H2'):
        for mode,arrays in data.items():
            path=out/(mode+'.npz');write_arrays(path,finish_mode(arrays,mapping));artifacts[path.name]=sha256(path)
        mask_payload={'sample_indices':np.asarray(mask_indices),'source_times_s':mapping['source_times_s'][mask_indices],
            'geometry_wh':np.asarray(g['decode_wh']),'flow_agreement':flow_values,'flow_known':flow_known}
        for mode in masks:
            mask_payload[mode+'_packed']=np.stack(masks[mode]);mask_payload[mode+'_present']=np.asarray(mask_present[mode],bool)
        write_arrays(out/'mask_controls.npz',mask_payload);artifacts['mask_controls.npz']=sha256(out/'mask_controls.npz')
    with clock.measure('H3_features'):
        write_arrays(out/'teacher_raw.npz',{'R1':r1,'dino_pretrained':dp,'dino_random_seed17':dr,
            'frame_valid':valid,'requested_times_s':mapping['requested_times_s'],'source_times_s':mapping['source_times_s']})
        artifacts['teacher_raw.npz']=sha256(out/'teacher_raw.npz')
    if (out/'new_cut_traces.npz').exists():artifacts['new_cut_traces.npz']=sha256(out/'new_cut_traces.npz')
    report={'status':'extracted','canonical_id':row['canonical_id'],'recording_id':row['recording_id'],
        'partition':row['partition'],'frames':n,'artifacts':artifacts,'source_sha256':row['source_video']['sha256'],
        'baseline_replay_max_abs_rich_error':max_baseline_error,'float16_teacher_max_abs_roundoff':max_quantization_error,
        'anchors':anchors,'missing_requested_anchors':missing_refresh,'cut_reset_indices':sorted(schedule.resets),
        'original_valid_frames':int(valid.sum()),'prediction_present_frames':{k:int(v['frame_valid'].sum()) for k,v in data.items()},
        'mode_color_valid_frames':{k:int(v['color_valid'].sum()) for k,v in data.items()},
        'elapsed_s':time.monotonic()-started,'stage_seconds':{k:clock.seconds[k]-stage_before[k] for k in clock.seconds},
        'independent_mask_accuracy':None,'independent_background_color_error':None,
        'scope':'Frozen observer/feature extraction; stability, presence and replay checks are not independent accuracy.'}
    write_json(out/'REPORT.json',report);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    root=Path(freeze['output']);root.mkdir(exist_ok=False)
    torch.set_num_threads(2);clock=StageClock(freeze['stage_allowance_seconds']);rows=[]
    with gpu_lock():
        gpu_started=time.monotonic();overhead_seconds=0.
        torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        with clock.measure('H1_H2'):
            segmenter=SAM2FrameSegmenter(Path(freeze['sam_checkpoint']),VisualConfig())
            cutie,cfg=models.load_cutie();transnet=models.load_transnet()
        with clock.measure('H3_features'):dino=models.load_dino();random_dino=models.load_dino(pretrained=False)
        loaded=segmenter,cutie,cfg,dino,random_dino,transnet
        for i,row in enumerate(freeze['records']):
            rows.append(extract(row,root/row['canonical_id'],freeze,loaded,clock))
            unaccounted=max(0.,time.monotonic()-gpu_started-sum(clock.seconds.values()))
            clock.seconds['H1_H2']+=unaccounted;overhead_seconds+=unaccounted
            if clock.seconds['H1_H2']>clock.limits['H1_H2']:raise RuntimeError('H1_H2 allowance including overhead exceeded')
            print('SHARED_RECORD',i+1,len(freeze['records']),row['canonical_id'],flush=True)
        peak=torch.cuda.max_memory_reserved()
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(root/'REPORT.json',{'status':'shared_extraction_complete','scope':freeze['scope'],
        'freeze_sha256':sha256(args.freeze),'rows':rows,'stage_seconds':clock.seconds,
        'common_GPU_process_overhead_charged_H1_H2_s':overhead_seconds,
        'peak_reserved_bytes':peak,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'study_cache_bytes':check_space(root,freeze.get('max_study_bytes',3*2**30)),
        'test_access':False,'original_hypotheses_closed':False})


if __name__=='__main__':main()
