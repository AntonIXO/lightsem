#!/usr/bin/env python
"""Apply D4/D6 to the saved active bodies without hiding raw failures.

The native body, MuQ timeline and checkpoint provenance stay frozen. This is a
modifier study only: each modified 30 Hz body is audited against its own saved
active body before any mapping or rendering is considered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import load_config, sha
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from lightsem.dance.pipeline import d6_prediction, modify_body
from lightsem.experiments.native_repair.assessment import raw_metrics
from lightsem.experiments.dynamic_dance.audio import model_features


BASE = Path('/mnt/lightsem-data/experiments/lightsem/native_full_native_repair_20261018_v2')
SONGS = ('shpongle', 'cure_for_me')
MODIFIERS = {'C_fixed': (False, False), 'D4': (True, False), 'D6': (False, True), 'D4_D6': (True, True)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run_song(config, song, out):
    source_path = BASE / 'render_full' / song / 'body.npz'
    source = DanceResult.load(source_path)
    timeline_path = Path(config['cached_tracks'][song]['timeline']['path'])
    music = MusicTimeline.load(timeline_path)
    n = len(source.world)
    if len(music.time) != n:
        raise ValueError(f'{song}: timeline/body clocks differ')
    # D6's planner is genuinely MuQ-conditioned; it does not invent a second
    # temporal oscillator. Run once and reuse its sealed tensor across arms.
    extra, _ = model_features(music, np.arange(n, dtype=np.float64) / 30.)
    telemetry, d6_identity = d6_prediction(extra, config, device='cuda')
    song_out = out / song
    song_out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, (use_d4, use_d6) in MODIFIERS.items():
        body = DanceResult.load(source_path)
        result, plan = modify_body(body, music, config, d4max=use_d4, d6=use_d6,
                                   telemetry=telemetry if use_d6 else None)
        path = song_out / f'{name}.npz'
        result.save(path)
        candidate = raw_metrics(result.world, source.world)
        source_metrics = raw_metrics(source.world, source.world)
        ratios = {
            key: float(candidate[key] / max(source_metrics[key], 1e-12))
            for key in ('raw_relative_speed_mean_m_s', 'raw_relative_jerk_p95_m_s3',
                        'high_frequency_residual_RMS_m')}
        finite = bool(np.isfinite(result.world).all())
        # This gate is deliberately conservative. No mapper, smoothing, D4/D6
        # cap or video may conceal a raw failure.
        safety = finite and ratios['raw_relative_speed_mean_m_s'] <= 1.5 and ratios['raw_relative_jerk_p95_m_s3'] <= 2.0
        rows[name] = {
            'status': 'pass_raw_safety' if safety else 'rejected_raw_safety',
            'd4max': use_d4, 'd6': use_d6, 'body': str(path), 'body_sha256': sha(path),
            'source_body': str(source_path), 'source_body_sha256': sha(source_path),
            'raw_metrics': candidate, 'source_raw_metrics': source_metrics, 'ratios': ratios,
            'finite': finite, 'd6_checkpoint': d6_identity if use_d6 else None,
            'render_authorized': False,
            'reason': 'modifier study remains diagnostic-only; source and musical gates are not qualification evidence',
        }
    write(song_out / 'REPORT.json', {'song': song, 'status': 'diagnostic_only',
          'source_body': str(source_path), 'source_body_sha256': sha(source_path),
          'timeline': str(timeline_path), 'timeline_sha256': sha(timeline_path),
          'd6_prediction': d6_identity, 'modifiers': rows})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/current_d4_d6_20261015_v1'))
    ap.add_argument('--baseline', type=Path, default=Path('configs/dance/fixed_c.json'))
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'preserve existing artifact root: {out}')
    config, config_path = load_config(args.baseline)
    out.mkdir(parents=True)
    songs = {song: run_song(config, song, out) for song in SONGS}
    write(out / 'PROTOCOL.json', {'schema_version': 1, 'id': 'current_d4_d6_20261015_v1',
        'status': 'diagnostic_only', 'baseline_config': str(config_path),
        'baseline_config_sha256': sha(config_path), 'source_root': str(BASE),
        'modifiers': list(MODIFIERS), 'songs': list(SONGS),
        'raw_gate': {'speed_ratio_max': 1.5, 'jerk_ratio_max': 2.0},
        'mapping_policy': 'map only a pass_raw_safety candidate after independent source/DMX checks; no automatic render/promotion',
        'active_baseline_unchanged': True})
    write(out / 'REPORT.json', {'status': 'diagnostic_only', 'songs': songs,
                                'active_baseline_unchanged': True})
    print(json.dumps({'out': str(out), 'status': 'diagnostic_only'}, indent=2))


if __name__ == '__main__':
    main()

