#!/usr/bin/env python3
"""Generate independent baseline/R1/R2 native-body demo candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.native_dynamics.evaluation import setup, cpu_noise
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.eval.training_health import motion_metrics


BASELINE = Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/learning/runs/joint/seed_0/final.pt')
SONGS = {
    'shpongle': Path('/root/shpongle.flac'),
    'cure_for_me': Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool_native(value: np.ndarray, count: int) -> np.ndarray:
    """Use the same zero-pad/partial-bin mean as authenticated native C."""
    bins = (count + 14) // 15
    pad = bins * 15 - count
    padded = np.pad(value[:count].astype(np.float32), ((0, pad), (0, 0)))
    summed = padded.reshape(bins, 15, -1).sum(1)
    denom = np.minimum(15, count - np.arange(bins) * 15)[:, None]
    return (summed / denom).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-checkpoint', type=Path, required=True)
    p.add_argument('--context-manifest', type=Path, required=True)
    p.add_argument('--translator', type=Path,
                   help='TRAIN-only frozen feature translator from SpaIn/CRW/oBSRNN to R2 context')
    p.add_argument('--base92-root', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/demo_features/features'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--condition', default='native_stereo')
    p.add_argument('--stereo-mode', choices=('normal','zero','reverse'), default='normal')
    p.add_argument('--adapter-scale', type=float, help='override native stereo adapter scale; recorded in provenance')
    p.add_argument('--stereo-adapter-scale', type=float,
                   help='override only stereo adapter scale; preserves native baseline adapter scale')
    p.add_argument('--stereo-plan-scale', type=float,
                   help='override stereo planner scale; defaults to --stereo-adapter-scale when supplied')
    p.add_argument('--safety-reference-root', type=Path,
                   help='baseline body directory used for raw speed/acceleration/jerk admission')
    p.add_argument('--max-speed-ratio', type=float, default=1.5)
    p.add_argument('--max-acceleration-ratio', type=float, default=2.0)
    p.add_argument('--max-jerk-ratio', type=float, default=2.0)
    args = p.parse_args(argv)
    if args.out.exists(): raise FileExistsError(args.out)
    # Match the authenticated native-C inference path exactly.  In particular,
    # the flow noise is the stationary CPU draw filtered by the learned noise
    # kernel.  A raw device torch.randn draw creates a different high-frequency
    # process and can look like constant shaking even with zero stereo adapters.
    setup()
    saved=torch.load(args.adapter_checkpoint,map_location='cpu',weights_only=False)
    cfg_values=dict(saved['config'])
    if args.adapter_scale is not None:
        if args.adapter_scale <= 0 or args.adapter_scale > 1:
            raise ValueError('--adapter-scale must be in (0,1]')
        cfg_values['adapter_scale']=float(args.adapter_scale)
    if args.stereo_adapter_scale is not None:
        if args.stereo_adapter_scale <= 0 or args.stereo_adapter_scale > 1:
            raise ValueError('--stereo-adapter-scale must be in (0,1]')
        cfg_values['stereo_adapter_scale']=float(args.stereo_adapter_scale)
        cfg_values['stereo_plan_scale']=float(args.stereo_adapter_scale if args.stereo_plan_scale is None else args.stereo_plan_scale)
    elif args.stereo_plan_scale is not None:
        if args.stereo_plan_scale <= 0 or args.stereo_plan_scale > 1:
            raise ValueError('--stereo-plan-scale must be in (0,1]')
        cfg_values['stereo_plan_scale']=float(args.stereo_plan_scale)
    cfg=Config(**cfg_values)
    model=NativeDynamicsModel(MusicConfig(**saved['base_config']),saved['normalizers'],cfg)
    load=model.load_state_dict(saved['model'],strict=True)
    if load.missing_keys or load.unexpected_keys: raise ValueError(f'Adapter checkpoint mismatch: {load}')
    model.to(args.device).eval()
    context_manifest=json.loads(args.context_manifest.read_text()); context_rows={r['id']:r for r in context_manifest['rows']}
    translator = None
    if args.translator is not None:
        with np.load(args.translator, allow_pickle=False) as z:
            required = ('weights', 'input_mean', 'input_std', 'output_mean', 'output_std')
            if any(key not in z.files for key in required):
                raise ValueError(f'Invalid frozen translator schema: {args.translator}')
            translator = {key: z[key].astype(np.float32) for key in required}
        if translator['weights'].shape != (65, 64) or any(value.shape != (64,) for key, value in translator.items() if key != 'weights'):
            raise ValueError(f'Invalid frozen translator shapes: {args.translator}')
    base_mean=torch.tensor(saved['audio_normalizers']['mean'],device=args.device);base_std=torch.tensor(saved['audio_normalizers']['std'],device=args.device)
    extra_mean=torch.tensor(saved['extra_normalizers']['mean'],device=args.device);extra_std=torch.tensor(saved['extra_normalizers']['std'],device=args.device)
    stereo_mean=torch.tensor(saved['stereo_mean'],device=args.device);stereo_std=torch.tensor(saved['stereo_std'],device=args.device)
    args.out.mkdir(parents=True)
    rows={}
    for song,audio in SONGS.items():
        root=Path('/root/lightsem/experiments/native_bf16_v7b')
        timeline=MusicTimeline.load(root/'demo_features/features/timelines'/f'{song}.npz')
        target=timeline.time.astype(np.float64)
        extra,_=model_features(timeline,target)
        with np.load(root/'demo_features/features/base92'/f'{song}.npz') as z:
            bt=z['time30'] if 'time30' in z else z['time']
            bx=z['x30'] if 'x30' in z else z['x']
        base=np.stack([np.interp(target,bt,bx[:,i]) for i in range(92)],axis=1).astype(np.float32)
        context=context_rows[song]
        with np.load(context['context_path']) as z:
            stereo=np.stack([np.interp(target,z['time30'],z['fused30'][:,i]) for i in range(64)],1).astype(np.float32)
        if translator is not None:
            normalized=(stereo-translator['input_mean'])/np.maximum(translator['input_std'],1e-4)
            design=np.concatenate((normalized,np.ones((len(normalized),1),dtype=np.float32)),axis=1)
            stereo=(design@translator['weights']*translator['output_std']+translator['output_mean']).astype(np.float32)
            if not np.isfinite(stereo).all():
                raise ValueError(f'Frozen translator emitted nonfinite context for {song}')
        if args.stereo_mode == 'zero':
            stereo[:] = stereo.mean(0, keepdims=True)
        elif args.stereo_mode == 'reverse':
            stereo = stereo[::-1].copy()
        n=len(target); bins=(n+14)//15
        base_t=(torch.from_numpy(base).to(args.device)-base_mean)/base_std
        extra_t=(torch.from_numpy(extra).to(args.device)-extra_mean)/extra_std
        stereo_t=(torch.from_numpy(stereo).to(args.device)-stereo_mean)/stereo_std
        coarse=torch.from_numpy(pool_native(extra,n)).to(args.device); coarse=(coarse-extra_mean)/extra_std
        coarse_stereo=torch.from_numpy(pool_native(stereo,n)).to(args.device); coarse_stereo=(coarse_stereo-stereo_mean)/stereo_std
        fine=torch.cat((base_t,extra_t),-1)[None]; stereo_fine=stereo_t[None]; coarse=coarse[None]; coarse_stereo=coarse_stereo[None]
        valid=torch.ones(1,n,dtype=torch.bool,device=args.device); coarse_valid=torch.ones(1,bins,dtype=torch.bool,device=args.device)
        indices=torch.arange(n,device=args.device,dtype=torch.long)[None]
        noise=cpu_noise(model.flow,n,7001).to(args.device)
        with torch.inference_mode(): body,plan=model.generate(fine,coarse,coarse_valid,indices,noise,stereo_context=stereo_fine,coarse_stereo=coarse_stereo,core_frames=960)
        native=body[0].float().cpu().numpy(); body=joints_from_features(torch.from_numpy(native)[None])[0].cpu().numpy(); plan_np={k:v[0].float().cpu().numpy() for k,v in plan.items() if torch.is_tensor(v)}
        path=args.out/f'{song}.npz'; np.savez_compressed(path,time=target,body=body,native=native,**{f'plan_{k}':v for k,v in plan_np.items()})
        metrics=motion_metrics(body,fps=30)
        safety={'finite':metrics['finite'],'speed_p95':metrics['speed_p95'],'acceleration_p95':metrics['acceleration_raw']['p95'],'jerk_p95':metrics['jerk_raw']['p95']}
        if args.safety_reference_root:
            ref_path=args.safety_reference_root/f'{song}.npz'
            with np.load(ref_path,allow_pickle=False) as z: ref=z['body'].astype(np.float32)
            ref_metrics=motion_metrics(ref,fps=30)
            safety.update({'reference_body':str(ref_path),'reference_speed_p95':ref_metrics['speed_p95'],'reference_acceleration_p95':ref_metrics['acceleration_raw']['p95'],'reference_jerk_p95':ref_metrics['jerk_raw']['p95'],'speed_ratio':metrics['speed_p95']/max(ref_metrics['speed_p95'],1e-6),'acceleration_ratio':metrics['acceleration_raw']['p95']/max(ref_metrics['acceleration_raw']['p95'],1e-6),'jerk_ratio':metrics['jerk_raw']['p95']/max(ref_metrics['jerk_raw']['p95'],1e-6)})
            safety['pass']=bool(safety['finite'] and safety['speed_ratio']<=args.max_speed_ratio and safety['acceleration_ratio']<=args.max_acceleration_ratio and safety['jerk_ratio']<=args.max_jerk_ratio)
        rows[song]={'audio':str(audio),'audio_sha256':sha(audio),'adapter_checkpoint':str(args.adapter_checkpoint),'adapter_checkpoint_sha256':sha(args.adapter_checkpoint),'context_manifest':str(args.context_manifest),'context_manifest_sha256':sha(args.context_manifest),'translator':None if args.translator is None else str(args.translator),'translator_sha256':None if args.translator is None else sha(args.translator),'body':str(path),'body_sha256':sha(path),'frames':n,'body_fps':30,'condition':args.condition,'stereo_mode':args.stereo_mode,'adapter_scale':cfg.adapter_scale,'stereo_adapter_scale':cfg.stereo_adapter_scale,'source_body_used':False,'native_time':'original30Hz','d4max':False,'d6':False,'safety':safety,'diagnostic_only':True}
    safety_pass=all(row.get('safety',{}).get('pass',True) for row in rows.values())
    status='complete' if safety_pass else 'rejected_raw_safety'
    manifest={'schema_version':1,'status':status,'condition':args.condition,'stereo_mode':args.stereo_mode,'adapter_scale':cfg.adapter_scale,'stereo_adapter_scale':cfg.stereo_adapter_scale,'adapter_checkpoint':str(args.adapter_checkpoint),'adapter_checkpoint_sha256':sha(args.adapter_checkpoint),'translator':None if args.translator is None else str(args.translator),'translator_sha256':None if args.translator is None else sha(args.translator),'rows':rows,'raw_safety_gate':{'reference_root':None if args.safety_reference_root is None else str(args.safety_reference_root),'max_speed_ratio':args.max_speed_ratio,'max_acceleration_ratio':args.max_acceleration_ratio,'max_jerk_ratio':args.max_jerk_ratio,'pass':safety_pass},'scope':'Native RTMPose stereo-adapter body candidates; raw safety required before mapper/render'}
    mpath=args.out/'MANIFEST.json';mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');mpath.with_name(mpath.name+'.sha256.json').write_text(json.dumps({'sha256':sha(mpath)})+'\n')
    print(json.dumps({'status':status,'condition':args.condition,'songs':list(rows)})); return 0


if __name__=='__main__': raise SystemExit(main())
