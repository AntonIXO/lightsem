#!/usr/bin/env python3
"""Full-track candidate inference and RGB-only Shpongle review streams."""
from __future__ import annotations
import argparse,json,os,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.model.light_layers.training import load_checkpoint,generated_history,_normalize
from lightsem.model.light_layers.visual import sha256,write_json,write_arrays
from lightsem.experiments.visual_next.delivery import color_delivery


class FinitePaletteCapture:
    """Capture the actual all-state head batch before canonical argmax."""
    def __init__(self,model):self.model=model;self.cfg=model.cfg;self.tables=None;self.calls=0
    def palette_logits(self,context,previous):
        logits=self.model.palette_logits(context,previous);self.calls+=1
        if not torch.isfinite(logits).all() or logits.dtype!=torch.float32:raise RuntimeError('Invalid palette logits')
        if self.tables is not None or logits.shape[1:]!=(33,32):raise RuntimeError('Unexpected state-head batching')
        if not torch.equal(previous,torch.arange(33,device=previous.device).expand_as(previous)):raise RuntimeError('State inventory changed')
        self.tables=logits.detach().cpu().numpy().copy();return logits


def full_track(model,normalized):
    if normalized.shape!=(11092,94) or model.training:raise RuntimeError('Fixed full Shpongle inference contract')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        output=model(torch.from_numpy(normalized[None]).cuda())
        if set(output)!={'accent_logit','chromatic_change_logit','context10'}:raise RuntimeError('Inference output schema')
        if not all(torch.isfinite(x).all() for x in output.values()):raise RuntimeError('Nonfinite encoder output')
        capture=FinitePaletteCapture(model)
        rollout=generated_history(capture,output['context10'],output['chromatic_change_logit'],
            initial_ids=None,state_ablation=False,cue_threshold=.5,initialize=True)
        result={k:v[0].cpu().numpy() for k,v in {**output,**rollout}.items()}
        actions=np.unique(np.r_[0,np.flatnonzero(result['color_cues'])]).astype(np.int64)
        if capture.calls!=1 or capture.tables.shape!=(len(actions),33,32):raise RuntimeError('Actual palette action table missing')
        previous=result['previous_ids'][actions]
        selected=capture.tables.argmax(-1)[np.arange(len(actions)),previous]
        if not np.array_equal(selected,result['generated_palette_ids'][actions]):raise RuntimeError('Captured palette actions disagree with rollout')
        if result['previous_ids'][0]!=32 or not np.array_equal(result['previous_ids'][1:],result['generated_palette_ids'][:-1]):raise RuntimeError('Generated history is not the prior own state')
        result.update(action_indices=actions,action_previous_ids=previous,
            action_argmax_choices=capture.tables.argmax(-1).astype(np.uint8),
            action_logits=capture.tables[np.arange(len(actions)),previous],
            accent_score=output['accent_logit'][0].sigmoid().cpu().numpy(),
            color_score=output['chromatic_change_logit'][0].sigmoid().cpu().numpy(),
            time40_s=np.arange(11092)/40,time10_s=np.arange(2773)/10)
    if not all(np.isfinite(x).all() for x in result.values()):raise RuntimeError('Nonfinite saved inference')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path);ap.add_argument('--candidate-id',required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    matches=[r for r in freeze['candidates'] if r['id']==args.candidate_id]
    if len(matches)!=1:raise RuntimeError('Candidate outside frozen review matrix')
    candidate=matches[0]
    if freeze['files'].get(candidate['checkpoint']['path'])!=candidate['checkpoint']['sha256']:raise RuntimeError('Checkpoint is not bound to review inputs')
    if any(shutil.disk_usage(path).free<2**30 for path in (freeze['study_root'],'/mnt')):raise RuntimeError('Free-space floor')
    out=Path(freeze['output'])/candidate['id'];out.mkdir(parents=True,exist_ok=False)
    with np.load(freeze['audio94']['path'],allow_pickle=False) as z:
        raw=z['audio94'];normalized=z['normalized_audio94'];times=z['time40_s']
    transforms=freeze['transforms']
    if not np.array_equal(normalized,_normalize(raw,transforms['audio'])):raise RuntimeError('C audio normalization drift')
    with np.load(freeze['baseline_C']['path'],allow_pickle=False) as z:baseline=z['candidate']
    with np.load(freeze['companion']['path'],allow_pickle=False) as z:companion=z['companion']
    torch.set_num_threads(2);started=time.monotonic()
    with gpu_lock():
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        others=[x.strip() for x in active.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
        if others:raise RuntimeError('GPU is in use: '+','.join(others))
        model,metadata=load_checkpoint(candidate['checkpoint']['path'],device='cuda');torch.cuda.reset_peak_memory_stats()
        if candidate['role'] in ('final','demonstration'):
            if metadata['step']!=6000:raise RuntimeError('Review model is not the fixed final checkpoint')
            if (metadata['arm'],metadata['seed'])!=(candidate['arm'],candidate['seed']):raise RuntimeError('Review checkpoint arm/seed mismatch')
            if metadata['main_targets_sha256']!=candidate['main_targets_sha256']:raise RuntimeError('Review supervision recipe mismatch')
        scores=full_track(model,normalized);peak=torch.cuda.max_memory_reserved()
        if peak>10*2**30:raise RuntimeError('Inference VRAM allowance')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Inference RAM allowance')
        if time.monotonic()-started>freeze['per_candidate_max_seconds']:raise RuntimeError('Inference runtime allowance')
    write_arrays(out/'model_outputs.npz',scores)
    variants=[]
    for layout in candidate['layouts']:
        stream,report=color_delivery(baseline,companion,times,scores['generated_palette_ids'],scores['color_cues'],
            transforms['palette_prototypes'],layout=layout)
        path=out/(layout+'.npz');write_arrays(path,{'candidate':stream,'time':times})
        variants.append({'layout':layout,'streams':str(path),'sha256':sha256(path),'invariants':report})
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    root=Path(freeze['study_root'])
    if sum(p.stat().st_size for d in root.iterdir() if d.is_dir() and d.name!='papers' for p in d.rglob('*') if p.is_file())>freeze['max_study_bytes']:raise RuntimeError('Study artifact allowance')
    write_json(out/'REPORT.json',{'status':'actual_candidate_inference_and_RGB_streams_ready',
        'candidate':candidate,'freeze_sha256':sha256(args.freeze),'checkpoint_metadata':metadata,
        'model_outputs_sha256':sha256(out/'model_outputs.npz'),'variants':variants,
        'elapsed_s':time.monotonic()-started,'peak_reserved_bytes':peak,
        'original_representation_fit_gate_pass':metadata.get('original_representation_fit_gate_pass'),
        'learned_color_active':bool(scores['color_cues'].any()),
        'context':'one whole277.3s forward, FP32, own generated history; identical route to accepted06',
        'scope':'Personal review only; no reference image or target history at playback. C movement/dimmers held fixed; fixed group modes are design previews, not learned spatial allocation.',
        'videos_rendered':0})


if __name__=='__main__':main()
