"""Map the two fresh native bodies through fixed mapper 01 and P2 brightness."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from lightsem.experiments.native_dynamics.data import sha
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.conductor.body_transfer_v2 import make_outputs


def main():
    root = Path('/root/lightsem/experiments/native_bf16_v7b')
    combo = json.loads(Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/PLAN.json').read_text())
    transfer, counts = combo['transfer'], combo['counts']
    audio = {
        'shpongle': Path('/root/shpongle.flac'),
        'cure_for_me': root/'demo_audio/Cure_For_Me.wav',
    }
    methods = {}
    out = root/'demo_emission'/'joint_seed0'; out.mkdir(parents=True, exist_ok=True)
    for song, audio_path in audio.items():
        with np.load(root/'demo_bodies/joint_seed0'/f'{song}.npz', allow_pickle=False) as z:
            body, native, time = z['body'], z['native'], z['time']
        colors = root/'demo_colors_v2'/song/'P2.npz'
        loud_report, _ = load_verified_loudness(root/'demo_loudness'/song/'loudness.json', audio_path)
        frames = len(np.load(colors, allow_pickle=False)['left'])
        t40 = np.arange(frames, dtype=np.float64) / 40
        world = _resample_body_full_interval(body, time, t40)
        left, right, palette, active, identity = _load_p2_colors(colors, audio_path, frames)
        outputs = make_outputs(**counts)
        dmx, patch, intent, bright, hard = compile_world(world, t40, outputs, left, right, loud_report, transfer, mapper_id='01')
        path = out/f'{song}.npz'; np.savez_compressed(path, time=t40, body=world, universes=dmx, dimmer=intent['dimmer'])
        report={'song':song,'status':'engineering_valid' if hard.get('pass',hard.get('passed',True)) else 'diagnostic',
                'body_sha256':sha(root/'demo_bodies/joint_seed0'/f'{song}.npz'),'stream_sha256':sha(path),
                'audio_sha256':sha(audio_path),'p2_sha256':sha(colors),'loudness_receipt_sha256':sha(root/'demo_loudness'/song/'loudness.json'),
                'mapper':'01','counts':counts,'hard_checks':hard,'P2_RGB_exact':True,
                'raw_stream_shape':list(dmx.shape),'source_body_used':False}
        (out/f'{song}.json').write_text(json.dumps(report,indent=2)+'\n'); methods[song]=report
        print(song, json.dumps(report), flush=True)
    (out/'MANIFEST.json').write_text(json.dumps({'schema_version':1,'status':'complete','rows':methods},indent=2)+'\n')


if __name__ == '__main__': main()
