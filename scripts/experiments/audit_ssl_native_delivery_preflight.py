"""Pre-render gate for the SSL native before/after delivery."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from lightsem.conductor.body_transfer import decode


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def metrics(path: Path, patch: list[dict]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        u = np.asarray(z['universes'], np.uint8)
        body = np.asarray(z['body'], np.float32)
        t = np.asarray(z['time'], np.float64)
    if u.ndim != 3 or u.shape[1:] != (1, 512): raise ValueError(f'bad DMX shape: {path}')
    if not np.isfinite(body).all() or not np.isfinite(t).all() or not np.array_equal(t, np.arange(len(t))/40):
        raise ValueError(f'nonfinite or nonuniform source: {path}')
    d = decode(u, patch)
    angles = np.asarray(d['angles_deg'], np.float64)
    dt = 1 / 40
    speed = np.diff(angles, axis=0) / dt
    accel = np.diff(speed, axis=0) / dt
    rgb = np.asarray(d['rgb'], np.uint8)
    inactive = np.mean(np.all(u == u[0:1], axis=(1, 2)))
    return {
        'sha256': sha(path), 'frames': int(len(u)), 'duration_s': float(len(u)/40),
        'finite': bool(np.isfinite(angles).all()), 'speed_max_deg_s': float(np.max(np.abs(speed))),
        'speed_p99_deg_s': float(np.percentile(np.abs(speed), 99)),
        'acceleration_max_deg_s2': float(np.max(np.abs(accel))),
        'acceleration_p99_deg_s2': float(np.percentile(np.abs(accel), 99)),
        'inactive_exact_first_frame_fraction': float(inactive),
        'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
        'exact_duplicate_stream_fraction': float(np.mean([np.array_equal(u[:,0,i], u[:,0,0]) for i in range(u.shape[-1])])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--before-root', type=Path, required=True)
    ap.add_argument('--after-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    patch = json.loads((args.before_root / 'shpongle' / 'patch.json').read_text())
    rows = {}
    for song in ('shpongle', 'cure_for_me'):
        b = args.before_root / song / 'timeline.npz'
        a = args.after_root / song / 'timeline.npz'
        bm, am = metrics(b, patch), metrics(a, patch)
        if bm['frames'] != am['frames']: raise ValueError(f'frame count changed: {song}')
        with np.load(b, allow_pickle=False) as bz, np.load(a, allow_pickle=False) as az:
            if not np.array_equal(bz['body'], az['body']): raise ValueError(f'body changed: {song}')
            bd, ad = decode(bz['universes'], patch), decode(az['universes'], patch)
            if not np.array_equal(bd['rgb'], ad['rgb']): raise ValueError(f'RGB changed: {song}')
            changed = np.mean(np.any(bz['universes'] != az['universes'], axis=(1,2)))
        rows[song] = {'before': bm, 'after': am, 'changed_dmx_frame_fraction': float(changed),
                      'ratios': {k: am[k] / bm[k] for k in ('speed_max_deg_s','speed_p99_deg_s','acceleration_max_deg_s2','acceleration_p99_deg_s2')}}
    report = {'schema_version': 1, 'status': 'passed_pre_render_gate', 'scope': 'raw saved 30Hz body and decoded 40Hz DMX',
              'source_dependence': 'body and limb-response checks inherited from v5 receipt; body is intentionally unchanged', 'songs': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
