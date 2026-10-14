#!/usr/bin/env python
"""Map raw-safe current D4/D6 modifier bodies to audited emitted DMX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import load_config, sha
from lightsem.dance.lighting import (_normalise_counts, _resample_body_full_interval,
    _source_dependence_audit, _stream_duplicate_groups, _validate_body_for_lighting)
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import make_outputs
from scripts.experiments.run_color_motion_field import _sealed_loudness_without_wave, _sealed_p2_without_wave


MOD_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/current_d4_d6_20261015_v1')
BASE_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_full_native_repair_20261018_v2')
SONGS = ('shpongle', 'cure_for_me')
MODIFIERS = ('C_fixed', 'D4', 'D6', 'D4_D6')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def one(config, song, out):
    track = config['cached_tracks'][song]
    with np.load(track['colors']['path'], allow_pickle=False) as z:
        frames = len(z['left'])
    source = DanceResult.load(BASE_ROOT / 'render_full' / song / 'body.npz')
    time = np.arange(frames, dtype=np.float64) / 40.
    world, body_time = _validate_body_for_lighting(source, float(time[-1]))
    levels, loudness_identity = _sealed_loudness_without_wave(
        Path(track['loudness_receipt']['path']), track['audio_sha256'])
    left, right, palette, active, color_identity = _sealed_p2_without_wave(
        Path(track['colors']['path']), Path(track['color_report']['path']), track['colors']['sha256'],
        track['audio_sha256'], frames)
    outputs = make_outputs(**_normalise_counts(config['lighting']['counts']))
    root = out / song
    root.mkdir(parents=True, exist_ok=False)
    rows = {}
    for modifier in MODIFIERS:
        body_path = MOD_ROOT / song / f'{modifier}.npz'
        body = DanceResult.load(body_path)
        world40 = _resample_body_full_interval(body.world, body.time, time)
        dmx, patch, intent, brightness, hard = compile_world(
            world40, time, outputs, left, right, levels,
            config['lighting']['transfer'], mapper_id=config['lighting']['mapper'])
        controls = decode(dmx, patch)
        source_check = _source_dependence_audit(
            world40, time, outputs, left, right, levels, config['lighting']['transfer'],
            config['lighting']['mapper'], intent,
            lambda w, t, o, l, r, lv, tr, mapper_id=None: compile_world(
                w, t, o, l, r, lv, tr, mapper_id=mapper_id))
        np.savez_compressed(root / f'{modifier}.npz', time=time, dmx=dmx,
                            angles_deg=controls['angles_deg'], directions=controls['directions'],
                            rgb=controls['rgb'], dimmer=controls['dimmer'], body=world40,
                            targets=intent['targets'], field_points=intent['field_points'])
        write(root / f'{modifier}_patch.json', patch)
        rows[modifier] = {
            'status': 'pass_source_dmx' if source_check['status'] == 'pass' and np.isfinite(dmx).all() else 'failed_source_dmx',
            'body': str(body_path), 'body_sha256': sha(body_path),
            'dmx': str(root / f'{modifier}.npz'), 'dmx_sha256': sha(root / f'{modifier}.npz'),
            'patch': str(root / f'{modifier}_patch.json'), 'patch_sha256': sha(root / f'{modifier}_patch.json'),
            'mapper': config['lighting']['mapper'], 'source_dependence': source_check,
            'hard_checks': hard, 'finite_dmx': bool(np.isfinite(dmx).all()),
            'exact_duplicate_groups': _stream_duplicate_groups(controls),
            'fixture_motion_limits': False, 'color': 'authenticated P2',
            'render_authorized': False,
        }
    write(root / 'REPORT.json', {'status': 'diagnostic_only', 'song': song,
        'source_body_sha256': sha(BASE_ROOT / 'render_full' / song / 'body.npz'),
        'color_identity': color_identity, 'loudness_identity': loudness_identity,
        'modifiers': rows, 'active_baseline_unchanged': True})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/current_d4_d6_20261015_v1_mapped'))
    ap.add_argument('--baseline', type=Path, default=Path('configs/dance/fixed_c.json'))
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    config, config_path = load_config(args.baseline)
    out.mkdir(parents=True)
    result = {song: one(config, song, out) for song in SONGS}
    write(out / 'REPORT.json', {'status': 'diagnostic_only', 'baseline_config': str(config_path),
        'baseline_config_sha256': sha(config_path), 'songs': result,
        'scope': 'current saved body with D4/D6 modifiers, actual emitted DMX, no video/promotion'})
    print(json.dumps({'out': str(out), 'status': 'diagnostic_only'}, indent=2))


if __name__ == '__main__':
    main()

