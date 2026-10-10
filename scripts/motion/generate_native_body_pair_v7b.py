"""Generate independent raw native bodies for the two authenticated showcase songs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from lightsem.experiments.native_dynamics.evaluation import setup, load_final, cpu_noise
from lightsem.experiments.native_dynamics.data import save_npz, sha
from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.motion_study.geometry import joints_from_features


def main():
    root = Path('/root/lightsem/experiments/native_bf16_v7b')
    feature_root = root / 'demo_features' / 'features'
    out = root / 'demo_bodies' / 'joint_seed0'
    out.mkdir(parents=True, exist_ok=True)
    setup()
    model, norms, identity = load_final(root, 'joint', 0)
    rows = []
    for song in ('shpongle', 'cure_for_me'):
        timeline = MusicTimeline.load(feature_root / 'timelines' / f'{song}.npz')
        duration = float(timeline.metadata['duration_s'])
        n = int(np.ceil(duration * 30))
        time = np.arange(n, dtype=np.float64) / 30
        extra, names = model_features(timeline, time)
        with np.load(feature_root / 'base92' / f'{song}.npz', allow_pickle=False) as z:
            bt, bx = z['time30'], z['x30']
        base = np.stack([np.interp(time, bt, bx[:, i]) for i in range(92)], axis=1).astype(np.float32)
        bins = int(np.ceil(n / 15))
        pad = bins * 15 - n
        coarse = np.pad(extra.astype(np.float32), ((0, pad), (0, 0))).reshape(bins, 15, 1061)
        coarse = (coarse.sum(1) / np.minimum(15, n - np.arange(bins) * 15)[:, None]).astype(np.float32)
        if bins > CONTEXT_BINS:
            raise ValueError(f'{song} exceeds planner context')
        fine = torch.cat(((torch.from_numpy(base).cuda() - norms[0][0]) / norms[0][1],
                          (torch.from_numpy(extra).cuda() - norms[1][0]) / norms[1][1]), dim=1)[None]
        c = torch.zeros((1, CONTEXT_BINS, 1061), device='cuda')
        valid = torch.zeros((1, CONTEXT_BINS), device='cuda', dtype=torch.bool)
        c[0, :bins] = (torch.from_numpy(coarse).cuda() - norms[1][0]) / norms[1][1]
        valid[0, :bins] = True
        indices = torch.arange(n, device='cuda')[None]
        noise = cpu_noise(model.flow, n, 7001).to('cuda')
        with torch.inference_mode():
            native, planner = model.generate(fine, c, valid, indices, noise)
        native = native[0].cpu().numpy()
        body = joints_from_features(torch.from_numpy(native).cuda()[None])[0].cpu().numpy()
        if not np.isfinite(native).all() or not np.isfinite(body).all():
            raise FloatingPointError(f'Nonfinite generated body: {song}')
        path = out / f'{song}.npz'
        save_npz(path, body=body, native=native, time=time)
        rows.append({'song':song,'body_path':str(path),'body_sha256':sha(path),
                     'native_shape':list(native.shape),'body_shape':list(body.shape),
                     'fps':30.,'source_timeline':str(feature_root/'timelines'/f'{song}.npz'),
                     'source_timeline_sha256':sha(feature_root/'timelines'/f'{song}.npz'),
                     'checkpoint':identity,'noise_seed':7001,
                     'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                     'audio_conditioned':True,'source_body_used':False})
        print(song, rows[-1], flush=True)
    (out/'MANIFEST.json').write_text(json.dumps({'schema_version':1,'status':'complete','rows':rows},indent=2)+'\n')


if __name__ == '__main__': main()
