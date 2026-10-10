#!/usr/bin/env python3
"""Map stereo-adapter bodies through fixed mapper 01, P2, and LUFS brightness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.conductor.body_transfer_v2 import make_outputs
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world


SONGS = {
    'shpongle': Path('/root/shpongle.flac'),
    'cure_for_me': Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bodies-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, default=Path('/root/lightsem/experiments/native_bf16_v7b'))
    args = p.parse_args(argv)
    if args.out.exists(): raise FileExistsError(args.out)
    cfg = json.loads(Path('/root/lightsem/configs/dance/fixed_c.json').read_text())
    plan = json.loads(Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/PLAN.json').read_text())
    transfer = cfg['lighting']['transfer']; counts = cfg['lighting']['counts']
    rows=[]; args.out.mkdir(parents=True)
    for condition_dir in sorted(args.bodies_root.iterdir()):
        if not condition_dir.is_dir(): continue
        for song,audio in SONGS.items():
            body_path=condition_dir/f'{song}.npz'
            if not body_path.is_file(): raise FileNotFoundError(body_path)
            with np.load(body_path,allow_pickle=False) as z:
                body=z['body'].astype(np.float32); time=z['time'].astype(np.float64)
            colors=args.baseline_root/'demo_colors_v2'/song/'P2.npz'
            loudness=args.baseline_root/'demo_loudness'/song/'loudness.json'
            with np.load(colors,allow_pickle=False) as z: frames=len(z['left'])
            t40=np.arange(frames,dtype=np.float64)/40.
            world=_resample_body_full_interval(body,time,t40)
            left,right,_,_,_=_load_p2_colors(colors,audio,len(t40)); loud,_=load_verified_loudness(loudness,audio)
            dmx,patch,intent,brightness,hard=compile_world(world,t40,make_outputs(**counts),left,right,loud,transfer,mapper_id='01')
            dest=args.out/f'{condition_dir.name}_{song}.npz'; np.savez_compressed(dest,time=t40,body=world,universes=dmx,dimmer=intent['dimmer'])
            rows.append({'condition':condition_dir.name,'song':song,'body':str(body_path),'body_sha256':sha(body_path),'audio':str(audio),'audio_sha256':sha(audio),'p2':str(colors),'p2_sha256':sha(colors),'loudness':str(loudness),'loudness_sha256':sha(loudness),'emission':str(dest),'emission_sha256':sha(dest),'mapper':'01','counts':counts,'p2_fixed':True,'brightness_fixed':True,'fixture_limits_off':transfer.get('apply_motion_limits') is False,'hard_checks':hard,'diagnostic_only':True})
    manifest={'schema_version':1,'status':'complete','rows':rows,'scope':'stereo adapter body candidates mapped through fixed mapper 01/P2/LUFS; no product promotion','d4max':False,'d6':False}
    path=args.out/'MANIFEST.json'; path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');path.with_name(path.name+'.sha256.json').write_text(json.dumps({'sha256':sha(path)})+'\n');print(json.dumps({'status':'complete','rows':len(rows)}));return 0


if __name__=='__main__': raise SystemExit(main())
