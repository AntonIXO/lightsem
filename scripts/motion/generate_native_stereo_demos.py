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
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.motion_study.geometry import joints_from_features


BASELINE = Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/learning/runs/joint/seed_0/final.pt')
SONGS = {
    'shpongle': Path('/root/shpongle.flac'),
    'cure_for_me': Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool(value: np.ndarray, bins: int) -> np.ndarray:
    pad = bins * 15 - len(value)
    if pad > 0: value = np.pad(value, ((0, pad), (0, 0)), mode='edge')
    return value[:bins * 15].reshape(bins, 15, -1).mean(1).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-checkpoint', type=Path, required=True)
    p.add_argument('--context-manifest', type=Path, required=True)
    p.add_argument('--base92-root', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/demo_features/features'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--condition', default='native_stereo')
    p.add_argument('--stereo-mode', choices=('normal','zero','reverse'), default='normal')
    args = p.parse_args(argv)
    if args.out.exists(): raise FileExistsError(args.out)
    saved=torch.load(args.adapter_checkpoint,map_location='cpu',weights_only=False)
    cfg=Config(**saved['config'])
    model=NativeDynamicsModel(MusicConfig(**saved['base_config']),saved['normalizers'],cfg)
    load=model.load_state_dict(saved['model'],strict=True)
    if load.missing_keys or load.unexpected_keys: raise ValueError(f'Adapter checkpoint mismatch: {load}')
    model.to(args.device).eval()
    context_manifest=json.loads(args.context_manifest.read_text()); context_rows={r['id']:r for r in context_manifest['rows']}
    base_mean=torch.tensor(saved['audio_normalizers']['mean'],device=args.device);base_std=torch.tensor(saved['audio_normalizers']['std'],device=args.device)
    extra_mean=torch.tensor(saved['extra_normalizers']['mean'],device=args.device);extra_std=torch.tensor(saved['extra_normalizers']['std'],device=args.device)
    stereo_mean=torch.tensor(saved['stereo_mean'],device=args.device);stereo_std=torch.tensor(saved['stereo_std'],device=args.device)
    args.out.mkdir(parents=True)
    rows={}
    for song,audio in SONGS.items():
        root=Path('/root/lightsem/experiments/native_bf16_v7b')
        timeline=MusicTimeline.load(root/'demo_features/features/timelines'/f'{song}.npz')
        target=timeline.time
        extra,_=model_features(timeline,target)
        with np.load(root/'demo_features/features/base92'/f'{song}.npz') as z: base=z['x30'].astype(np.float32)
        context=context_rows[song]
        with np.load(context['context_path']) as z:
            stereo=np.stack([np.interp(target,z['time30'],z['fused30'][:,i]) for i in range(64)],1).astype(np.float32)
        if args.stereo_mode == 'zero':
            stereo[:] = stereo.mean(0, keepdims=True)
        elif args.stereo_mode == 'reverse':
            stereo = stereo[::-1].copy()
        n=len(target); bins=(n+14)//15
        base_t=(torch.from_numpy(base).to(args.device)-base_mean)/base_std
        extra_t=(torch.from_numpy(extra).to(args.device)-extra_mean)/extra_std
        stereo_t=(torch.from_numpy(stereo).to(args.device)-stereo_mean)/stereo_std
        coarse=torch.from_numpy(pool(extra,bins)).to(args.device); coarse=(coarse-extra_mean)/extra_std
        coarse_stereo=torch.from_numpy(pool(stereo,bins)).to(args.device); coarse_stereo=(coarse_stereo-stereo_mean)/stereo_std
        fine=torch.cat((base_t,extra_t),-1)[None]; stereo_fine=stereo_t[None]; coarse=coarse[None]; coarse_stereo=coarse_stereo[None]
        valid=torch.ones(1,n,dtype=torch.bool,device=args.device); coarse_valid=torch.ones(1,bins,dtype=torch.bool,device=args.device)
        indices=torch.arange(n,device=args.device,dtype=torch.long)[None]
        noise=torch.randn((1,n,139),device=args.device,generator=torch.Generator(device=args.device).manual_seed(7001))
        with torch.inference_mode(): body,plan=model.generate(fine,coarse,coarse_valid,indices,noise,stereo_context=stereo_fine,coarse_stereo=coarse_stereo,core_frames=960)
        native=body[0].float().cpu().numpy(); body=joints_from_features(torch.from_numpy(native)[None])[0].cpu().numpy(); plan_np={k:v[0].float().cpu().numpy() for k,v in plan.items() if torch.is_tensor(v)}
        path=args.out/f'{song}.npz'; np.savez_compressed(path,time=target,body=body,native=native,**{f'plan_{k}':v for k,v in plan_np.items()})
        rows[song]={'audio':str(audio),'audio_sha256':sha(audio),'adapter_checkpoint':str(args.adapter_checkpoint),'adapter_checkpoint_sha256':sha(args.adapter_checkpoint),'context_manifest':str(args.context_manifest),'context_manifest_sha256':sha(args.context_manifest),'body':str(path),'body_sha256':sha(path),'frames':n,'body_fps':30,'condition':args.condition,'stereo_mode':args.stereo_mode,'source_body_used':False,'native_time':'original30Hz','d4max':False,'d6':False,'diagnostic_only':True}
    manifest={'schema_version':1,'status':'complete','condition':args.condition,'stereo_mode':args.stereo_mode,'adapter_checkpoint':str(args.adapter_checkpoint),'adapter_checkpoint_sha256':sha(args.adapter_checkpoint),'rows':rows,'scope':'Native RTMPose stereo-adapter body candidates; no DMX/render promotion'}
    mpath=args.out/'MANIFEST.json';mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');mpath.with_name(mpath.name+'.sha256.json').write_text(json.dumps({'sha256':sha(mpath)})+'\n')
    print(json.dumps({'status':'complete','condition':args.condition,'songs':list(rows)})); return 0


if __name__=='__main__': raise SystemExit(main())
