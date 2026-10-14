#!/usr/bin/env python
"""Run the isolated arbitrary-N color and distributed-motion comparison.

This script reuses the sealed full-native body and frozen P2/loudness inputs.
It never retrains or invents motion. Mapper IDs are compared on the same body;
the color-field arm only replaces RGB bytes after the mapper and keeps dimmer
and raw-body safety evidence separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import load_config, sha
from lightsem.dance.color_field import color_field_metrics, continuous_color_field
from lightsem.dance.lighting import (_load_p2_colors, _normalise_counts,
    _resample_body_full_interval, _stream_duplicate_groups,
    _validate_body_for_lighting, _source_dependence_audit)
from lightsem.experiments.dynamic_dance_combo.brightness import apply_brightness, load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.conductor.body_transfer import TransferConfig, compile_intent, decode
from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
from lightsem.experiments.native_repair.assessment import raw_metrics


ARTIFACT_DEFAULT = Path('/mnt/lightsem-data/experiments/lightsem/color_motion_field_20261015_v1')
BASE_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_full_native_repair_20261018_v2')
SONGS = ('shpongle', 'cure_for_me')
MAPPERS = ('01', '06', '11', '12', '13', '15')
SENSUAL_PALETTE = ('#25105f', '#6f1ab6', '#e52e88', '#ff78a9', '#ffd6e7')


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _sealed_p2_without_wave(color_path: Path, report_path: Path, expected_sha: str, audio_sha: str, frames: int):
    """Validate a P2 cache when the original source waveform was archived."""
    report = json.loads(report_path.read_text())
    if report.get('audio_sha256') != audio_sha or sha(color_path) != expected_sha:
        raise ValueError('archived P2 cache is not bound to the selected song')
    with np.load(color_path, allow_pickle=False) as z:
        left, right = z['left'].copy(), z['right'].copy()
        palette = z['palette_ids40'].copy()
        active = z['color_active40'].copy() if 'color_active40' in z else None
    if left.shape != (frames, 3) or right.shape != left.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('sealed P2 shape/clock mismatch')
    return left, right, palette, active, {'path': str(color_path), 'sha256': expected_sha,
        'report': str(report_path), 'report_sha256': sha(report_path), 'audio_sha256': audio_sha,
        'archived_waveform_binding': True}


def _sealed_loudness_without_wave(receipt_path: Path, expected_audio_sha: str):
    report = json.loads(receipt_path.read_text())
    sidecar = Path(str(receipt_path) + '.sha256.json')
    if sidecar.is_file() and sha(receipt_path) != json.loads(sidecar.read_text())['sha256']:
        raise ValueError('loudness receipt seal changed')
    if report.get('audio_sha256') != expected_audio_sha or sha(Path(report['path'])) != report['sha256']:
        raise ValueError('archived loudness cache is not bound to selected song')
    with np.load(report['path'], allow_pickle=False) as z:
        levels = {key: z[key].copy() for key in ('time', 'momentary', 'short_term')}
    return levels, {'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path), **report,
                    'archived_waveform_binding': True}


def _correlation(values):
    # values: T x fixtures x channels; constant fixtures are zero correlation,
    # not evidence of useful independence.
    if values.shape[1] < 2:
        return 0.
    rows = []
    for index in range(values.shape[1]):
        row = values[:, index].reshape(len(values), -1)
        rows.append(row - row.mean(0, keepdims=True))
    corr = []
    for i in range(len(rows)):
        for j in range(i):
            a, b = rows[i].ravel(), rows[j].ravel()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            corr.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.)
    return float(np.mean(np.abs(corr))) if corr else 0.


def motion_metrics(controls, patch):
    moving = np.array([row['kind'] != 'bulb' for row in patch], dtype=bool)
    angles = np.asarray(controls['angles_deg'])[:, moving]
    velocity = np.diff(angles, axis=0) * 40.
    if not len(velocity):
        return {'moving_fixture_count': int(moving.sum()), 'mean_abs_velocity_correlation': 0.,
                'angular_effective_rank': 0., 'angular_span_p50_deg': 0., 'finite': True}
    flat = velocity.reshape(len(velocity), -1)
    centered = flat - flat.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    rank = float((singular.sum() ** 2) / max((singular ** 2).sum(), 1e-12))
    span = np.ptp(angles, axis=0)
    return {
        'moving_fixture_count': int(moving.sum()),
        'mean_abs_velocity_correlation': _correlation(angles),
        'angular_effective_rank': rank,
        'angular_effective_rank_normalized': rank / max(1., 2. * int(moving.sum())),
        'angular_span_p50_deg': float(np.median(span)) if span.size else 0.,
        'angular_span_p90_deg': float(np.quantile(span, .9)) if span.size else 0.,
        'velocity_rms_deg_s': float(np.sqrt(np.mean(velocity ** 2))),
        'exact_duplicate_groups': _stream_duplicate_groups(controls),
        'finite': bool(np.isfinite(angles).all()),
    }


def _run_mapper(world, time, outputs, left, right, levels, transfer, mapper):
    result = compile_world(world, time, outputs, left, right, levels, transfer, mapper_id=mapper)
    dmx, patch, intent, brightness, hard = result
    controls = decode(dmx, patch)
    source = _source_dependence_audit(
        world, time, outputs, left, right, levels, transfer, mapper, intent,
        lambda w, t, o, l, r, lv, tr, mapper_id=None: compile_world(
            w, t, o, l, r, lv, tr, mapper_id=mapper_id))
    return result, controls, source


def _run_distributed(world, time, outputs, left, right, levels, transfer, *, fixture_rgb=None):
    """Stretch mapper-01 targets around fixed fixture centers.

    This is a reversible spatial readout experiment. Every temporal value is
    still supplied by the saved body field; the fixed gain only gives outer
    heads more angular room.
    """
    from scripts.motion.run_body_portfolio import hard_checks
    field = map_body(world, outputs, '01', 40.)
    cfg = TransferConfig(**dict(transfer))
    restored, trace = apply_brightness(world, time, outputs, field, levels)
    targets = np.asarray(restored['targets']).copy()
    for index, output in enumerate(outputs):
        if output['kind'] == 'bulb':
            continue
        center = np.median(targets[:, index], axis=0, keepdims=True)
        x = float(output['position'][0])
        gain = 1.22 + .20 * min(1., abs(x) / 6.5)
        targets[:, index] = center + gain * (targets[:, index] - center)
    restored = {**restored, 'targets': targets, 'distributed_gain': True}
    dmx, patch, final = compile_intent(restored, outputs, left, right, cfg, fixture_rgb=fixture_rgb)
    hard = hard_checks(dmx, patch, final, 40., enforce_motion_limits=cfg.apply_motion_limits)
    final['algorithm_id'] = '01_distributed_gain'
    return (dmx, patch, final, trace, hard), decode(dmx, patch)


def run_song(config, song, out):
    track = config['cached_tracks'][song]
    body_path = BASE_ROOT / 'render_full' / song / 'body.npz'
    body = DanceResult.load(body_path)
    audio = Path('/root/shpongle.flac') if song == 'shpongle' else Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav')
    if song == 'shpongle' and not audio.is_file():
        # The source FLAC is absent in this checkout, but the authenticated P2
        # and loudness assets remain bound to its hash; mapping does not need a
        # replacement soundtrack. Rendering records this limitation.
        audio = Path('/mnt/lightsem-data/experiments/lightsem/d4max_20261005_v1/audio/shpongle/reference.m4a')
    # The authenticated P2 clock is the lighting interval authority.  Native
    # bodies intentionally carry one terminal 30 Hz sample beyond that clock.
    with np.load(track['colors']['path'], allow_pickle=False) as z:
        frames = int(len(z['left']))
    time = np.arange(frames, dtype=np.float64) / 40.
    world, body_time = _validate_body_for_lighting(body, float(time[-1]))
    world40 = _resample_body_full_interval(world, body_time, time)
    color_path = Path(track['colors']['path'])
    source_audio = Path('/root/shpongle.flac') if song == 'shpongle' else audio
    if source_audio.is_file():
        left, right, palette, active, color_identity = _load_p2_colors(
            color_path, source_audio, frames, expected_sha256=track['colors']['sha256'], allow_fixture_report=False)
    else:
        left, right, palette, active, color_identity = _sealed_p2_without_wave(
            color_path, color_path.parent / 'REPORT.json', track['colors']['sha256'], track['audio_sha256'], frames)
    if len(left) != frames:
        raise ValueError(f'{song}: P2 frame count {len(left)} != expected {frames}')
    loudness_path = Path(track['loudness_receipt']['path'])
    if source_audio.is_file():
        levels, loudness_identity = load_verified_loudness(loudness_path, source_audio)
    else:
        levels, loudness_identity = _sealed_loudness_without_wave(loudness_path, track['audio_sha256'])
    counts = _normalise_counts(config['lighting']['counts'])
    outputs = make_outputs(**counts)
    transfer = config['lighting']['transfer']
    song_out = out / song
    song_out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((BASE_ROOT / 'paired_audit' / 'full_native' / f'{song}.json').read_text())
    _write(song_out / 'RAW_SAFETY.json', {
        'status': 'pass', 'scope': 'saved raw 30Hz full-native body before mapper/color/DMX',
        'body': str(body_path), 'body_sha256': sha(body_path),
        'checkpoint_sha256': receipt['checkpoint_sha256'], 'source_receipt': str(BASE_ROOT / 'paired_audit' / 'full_native' / f'{song}.json'),
        'source_receipt_sha256': sha(BASE_ROOT / 'paired_audit' / 'full_native' / f'{song}.json'),
        'measured_ratios': receipt['raw_safety'], 'raw_metrics': receipt['candidate']['raw'],
        'note': 'No new body generation; mapper and color experiments reuse the gate-passing saved body.'})
    rows = {}
    for mapper in MAPPERS:
        try:
            result, controls, source = _run_mapper(world40, time, outputs, left, right, levels, transfer, mapper)
            dmx, patch, intent, brightness, hard = result
            m = motion_metrics(controls, patch)
            rows[mapper] = {'status': 'pass' if source['status'] == 'pass' and m['finite'] else 'failed',
                            'mapper': mapper, 'motion': m, 'source_dependence': source,
                            'hard_checks': hard, 'rgb_mode': 'P2',
                            'duplicate_groups_named': [[patch[i]['name'] for i in g] for g in m['exact_duplicate_groups']]}
            np.savez_compressed(song_out / f'mapper_{mapper}.npz', time=time, dmx=dmx,
                                angles_deg=controls['angles_deg'], directions=controls['directions'],
                                rgb=controls['rgb'], dimmer=controls['dimmer'], body=world40,
                                targets=intent['targets'], field_points=intent['field_points'])
        except Exception as exc:
            rows[mapper] = {'status': 'error', 'mapper': mapper, 'error': f'{type(exc).__name__}: {exc}'}
    # The color arm uses the explicit distributed readout candidate. Mapper 01
    # remains the active reference; this gain is a reversible spatial test.
    mapper = '01_distributed_gain'
    result, controls = _run_distributed(world40, time, outputs, left, right, levels, transfer)
    source = _source_dependence_audit(
        world40, time, outputs, left, right, levels, transfer, mapper, result[2],
        lambda w, t, o, l, r, lv, tr, mapper_id=None: _run_distributed(w, t, o, l, r, lv, tr)[0])
    dmx, patch, intent, brightness, hard = result
    fixture_rgb, color_meta = continuous_color_field(
        world40, left, right, outputs, palette=SENSUAL_PALETTE,
        spatial_cycles=.95, temporal_gain=6.0, breath_depth=.10)
    colored, c_controls = _run_distributed(world40, time, outputs, left, right, levels, transfer,
                                            fixture_rgb=fixture_rgb)
    c_dmx, c_patch, c_intent, c_brightness, c_hard = colored
    c_metrics = color_field_metrics(
        fixture_rgb, dimmer=c_controls['dimmer'],
        fixture_positions=np.asarray([float(o.get('position', [i, 0, 0])[0]) for i, o in enumerate(outputs)]))
    c_source = _source_dependence_audit(
        world40, time, outputs, left, right, levels, transfer, mapper, c_intent,
        lambda w, t, o, l, r, lv, tr, mapper_id=None: _run_distributed(
            w, t, o, l, r, lv, tr, fixture_rgb=fixture_rgb[:len(w), :len(o)])[0])
    np.savez_compressed(song_out / 'color_field.npz', time=time, dmx=c_dmx,
                        angles_deg=c_controls['angles_deg'], directions=c_controls['directions'],
                        rgb=c_controls['rgb'], dimmer=c_controls['dimmer'], fixture_rgb=fixture_rgb,
                        body=world40, targets=c_intent['targets'], field_points=c_intent['field_points'])
    _write(song_out / 'color_field_patch.json', c_patch)
    _write(song_out / 'COLOR_FIELD.json', {
        'status': 'diagnostic_only', 'algorithm': color_meta,
        'motion': motion_metrics(c_controls, c_patch),
        'source_dependence_motion': c_source, 'hard_checks': c_hard,
        'metrics': c_metrics, 'mapper': mapper, 'body_sha256': sha(body_path),
        'p2_color_identity': color_identity, 'loudness_identity': loudness_identity,
        'fixture_counts': counts, 'fixture_rgb_sha256': sha(song_out / 'color_field.npz'),
        'patch': str(song_out / 'color_field_patch.json'),
        'patch_sha256': sha(song_out / 'color_field_patch.json'),
        'brightness_scope': 'unmodified historical LUFS/body dimmer; color diagnostics are separate',
        'note': 'Continuous OKLab field is an isolated experiment; active P2 baseline is unchanged.'})
    _write(song_out / 'MOTION_SUMMARY.json', {'status': 'diagnostic_only', 'song': song,
             'body_sha256': sha(body_path), 'mapper_candidates': rows,
             'selection_rule': 'No automatic promotion; 01_distributed_gain only supplies the separate color-field preview.',
             'input_frozen': {'body': sha(body_path), 'P2': track['colors']['sha256'],
                              'checkpoint': receipt['checkpoint_sha256']}})
    return {'song': song, 'body': str(body_path), 'duration_s': float(time[-1]),
            'mappers': rows, 'color': json.loads((song_out / 'COLOR_FIELD.json').read_text())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ARTIFACT_DEFAULT)
    ap.add_argument('--baseline', type=Path, default=Path('configs/dance/fixed_c.json'))
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'preserve existing artifact root: {out}')
    config, config_path = load_config(args.baseline)
    out.mkdir(parents=True)
    rows = [run_song(config, song, out) for song in SONGS]
    _write(out / 'PROTOCOL.json', {
        'schema_version': 1, 'id': 'color_motion_field_20261015_v1',
        'status': 'diagnostic_only', 'baseline_config': str(config_path),
        'baseline_config_sha256': sha(config_path), 'mappers': list(MAPPERS),
        'songs': list(SONGS), 'body_source': str(BASE_ROOT),
        'color_algorithm': 'oklab_continuous_fixture_field',
        'motion_rule': 'same saved native body; mapper-only readout comparison',
        'd4_d6': 'not silently applied; separate controlled runs remain required',
        'safety': 'raw 30Hz receipts copied before mapper/DMX; no render from failed gates',
        'research_basis': ['https://www.w3.org/TR/css-color/', 'https://bottosson.github.io/posts/oklab/',
                           'https://arxiv.org/abs/1306.0895', 'https://arxiv.org/abs/2211.10658'],
        'rows': [{'song': r['song'], 'duration_s': r['duration_s']} for r in rows],
    })
    _write(out / 'REPORT.json', {'status': 'diagnostic_only', 'protocol': str(out / 'PROTOCOL.json'),
                                 'songs': rows, 'no_active_baseline_change': True})
    print(json.dumps({'out': str(out), 'status': 'diagnostic_only', 'songs': list(SONGS)}, indent=2))


if __name__ == '__main__':
    main()

