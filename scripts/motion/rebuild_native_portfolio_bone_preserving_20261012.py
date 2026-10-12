"""Build and gate the D7-calibrated, bone-preserving 30-demo portfolio.

This campaign has an immutable source plan, one same-song D7 reference per
track, and a strict order: corrected native body -> raw safety/per-joint gate
-> mapper -> decoded per-output/angular gate -> render release.  No output gain,
clock, smoothing, fixture cap, or fallback motion is introduced.
"""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import make_outputs
from lightsem.conductor.body_transfer_v2.diagnostics import measure
from lightsem.conductor.d7_amplitude_adapter import calibrate_to_d7, body_amplitude_metrics
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.experiments.native_repair.assessment import raw_metrics

BASE = Path('/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8')
D7 = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/bodies/D7_reference')
OUT = Path(os.environ.get('NATIVE_BONE_PORTFOLIO_ROOT', '/mnt/lightsem-data/experiments/lightsem/native_portfolio_bone_preserving_20261012_v1'))
ROOT = Path('/root/lightsem')
N = 7200


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def write_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if any(not np.isfinite(np.asarray(v)).all() for v in values.values() if np.issubdtype(np.asarray(v).dtype, np.number)):
        raise ValueError('nonfinite array in immutable artifact')
    with path.open('xb') as f: np.savez_compressed(f, **values)


def load_world(path):
    with np.load(path, allow_pickle=False) as z:
        key = 'world' if 'world' in z.files else 'body'
        return z[key].astype(float), z['time'].astype(float) if 'time' in z.files else np.arange(len(z[key])) / 30.


def raw_acceleration_p95(body):
    value = np.asarray(body, dtype=float); relative = value - value[:, :1]
    acceleration = np.diff(relative, n=2, axis=0) * 30. ** 2
    return float(np.quantile(np.linalg.norm(acceleration, axis=-1), .95))


def freeze():
    if (OUT / 'PLAN.json').exists(): raise FileExistsError(OUT / 'PLAN.json')
    base = json.loads((BASE / 'PLAN.json').read_text())
    tracks = []
    for t in base['tracks']:
        song = t['id']; v8 = BASE / 'bodies' / song / 'body.npz'; d7 = D7 / song / 'body.npz'
        tracks.append({**t, 'v8_body': str(v8), 'v8_body_sha256': sha(v8), 'd7_reference_body': str(d7), 'd7_reference_body_sha256': sha(d7)})
    source_paths = [ROOT / 'scripts/motion/rebuild_native_portfolio_bone_preserving_20261012.py',
                    ROOT / 'lightsem/conductor/d7_amplitude_adapter.py', ROOT / 'lightsem/conductor/body_transfer_v2/installation.py',
                    ROOT / 'lightsem/conductor/body_transfer_v2/common.py', ROOT / 'lightsem/conductor/body_transfer_v2/composition.py',
                    ROOT / 'lightsem/conductor/body_transfer.py', ROOT / 'configs/dance/fixed_c.json']
    files = {str(p.resolve()): sha(p) for p in source_paths}
    for t in tracks:
        for key in ('audio', 'audio_reference', 'colors', 'color_report', 'loudness', 'v8_body', 'd7_reference_body'):
            p = t[key]['path'] if isinstance(t[key], dict) else t[key]; files[str(Path(p).resolve())] = sha(p)
    plan = {'schema_version': 1, 'id': OUT.name, 'status': 'frozen_pre_render_gate_campaign',
            'base_plan': str(BASE / 'PLAN.json'), 'base_plan_sha256': sha(BASE / 'PLAN.json'),
            'baseline_id': base['baseline_id'], 'baseline_config': base['baseline_config'],
            'baseline_config_sha256': base['baseline_config_sha256'], 'checkpoint': base['checkpoint'], 'muq': base['muq'],
            'stereo_conditioning': base['stereo_conditioning'], 'tracks': tracks, 'algorithms': base['algorithms'],
            'counts': base['counts'], 'prefix_s': 180, 'review_start_s': 60, 'review_duration_s': 120,
            'fps': 40., 'body_fps': 30., 'd4max': False, 'd6': False, 'motion_limits': False,
            'correction': {'name': 'bone_preserving_D7_angular_amplitude_v1', 'max_gain': 2.5,
                           'body_coordinates_stretched': False, 'mapper_projection': 'bounded_softclip_v4',
                           'stage_projection_scale': [7.5, 5.0, 2.8], 'stage_projection_slope_multiplier': 1.5,
                           'pelvis_slot_redirect': 'moving role-0 slot uses neck joint 12; pelvis remains global/garland-only',
                           'temporal_controller': False, 'training_checkpoint_changed': False},
            'gates': {'body_angular_ratio_min': .75, 'body_angular_ratio_max': 1.35,
                      'body_speed_ratio_max': 1.7, 'body_acceleration_ratio_max': 1.7,
                      'body_jerk_ratio_max': 1.7, 'body_hf_ratio_max': 1.6, 'root_fast_ratio_max': 1.5,
                      'support_ratio_min': .50,
                      'support_slide_ratio_max': 2.20, 'bone_length_error_max_m': 1e-6,
                      'output_angular_ratio_min': .30, 'output_angular_ratio_max': 2.50,
                      'output_axis_range_ratio_min': .30, 'output_axis_range_ratio_max': 2.50,
                      'output_direction_excursion_min_deg': .75, 'output_active_fraction_min': .20,
                      'decoded_angle_rmse_max_deg': .15, 'target_saturation_fraction_max': .25,
                      'decoded_speed_ratio_max': 1.80, 'decoded_acceleration_ratio_max': 1.80},
            'files': files}
    OUT.mkdir(parents=True); write_json(OUT / 'PLAN.json', plan); write_json(OUT / 'source_manifest.json', {'files': files})
    print(json.dumps({'stage': 'freeze', 'root': str(OUT), 'plan_sha256': sha(OUT / 'PLAN.json')}, indent=2))


def correct_bodies():
    plan = json.loads((OUT / 'PLAN.json').read_text()); rows = {}
    for t in plan['tracks']:
        song = t['id']; src, st = load_world(t['v8_body']); ref, _ = load_world(t['d7_reference_body'])
        corrected, calibration = calibrate_to_d7(src, ref, max_gain=plan['correction']['max_gain'])
        metrics = body_amplitude_metrics(corrected, src, ref); rm = raw_metrics(corrected, ref); sm = raw_metrics(src, ref)
        speed_ratio = rm['raw_relative_speed_mean_m_s'] / max(sm['raw_relative_speed_mean_m_s'], 1e-12)
        acceleration_ratio = raw_acceleration_p95(corrected) / max(raw_acceleration_p95(src), 1e-12)
        jerk_ratio = rm['raw_relative_jerk_p95_m_s3'] / max(sm['raw_relative_jerk_p95_m_s3'], 1e-12)
        hf_ratio = rm['high_frequency_residual_RMS_m'] / max(sm['high_frequency_residual_RMS_m'], 1e-12)
        root_fast_ratio = rm['root_vertical_fast_residual_RMS_m'] / max(sm['root_vertical_fast_residual_RMS_m'], 1e-12)
        passed = (metrics['finite'] and not metrics['constant_body'] and metrics['bone_length_error_max_m'] <= plan['gates']['bone_length_error_max_m']
                  and plan['gates']['body_angular_ratio_min'] <= metrics['per_joint_angular_range_ratio_min'] <= plan['gates']['body_angular_ratio_max']
                  and plan['gates']['body_angular_ratio_min'] <= metrics['per_joint_angular_range_ratio_max'] <= plan['gates']['body_angular_ratio_max']
                  and speed_ratio <= plan['gates']['body_speed_ratio_max'] and acceleration_ratio <= plan['gates']['body_acceleration_ratio_max']
                  and jerk_ratio <= plan['gates']['body_jerk_ratio_max'] and hf_ratio <= plan['gates']['body_hf_ratio_max']
                  and root_fast_ratio <= plan['gates']['root_fast_ratio_max']
                  and rm['geometric_support_fraction'] / max(sm['geometric_support_fraction'], 1e-9) >= plan['gates']['support_ratio_min']
                  and rm['geometric_slide_p95_m_s'] / max(sm['geometric_slide_p95_m_s'], 1e-9) <= plan['gates']['support_slide_ratio_max'])
        if not passed: raise RuntimeError(f'raw body gate failed for {song}: {metrics}, speed={speed_ratio}, jerk={jerk_ratio}')
        path = OUT / 'bodies' / song / 'body.npz'
        write_npz(path, time=st, world=corrected, source_body_sha256=np.asarray(sha(t['v8_body'])))
        write_json(path.with_suffix('.json'), {'song': song, 'body_sha256': sha(path), 'source_body_sha256': sha(t['v8_body']),
                                               'd7_reference_sha256': sha(t['d7_reference_body']), 'calibration': calibration,
                                               'metrics': metrics, 'speed_ratio_vs_v8': speed_ratio, 'acceleration_ratio_vs_v8': acceleration_ratio,
                                               'jerk_ratio_vs_v8': jerk_ratio, 'high_frequency_ratio_vs_v8': hf_ratio,
                                               'root_fast_ratio_vs_v8': root_fast_ratio,
                                               'raw_gate_passed': True, 'provenance': 'bone-preserving native-body correction before mapper'})
        rows[song] = {'body': str(path), 'body_sha256': sha(path), 'metrics': metrics, 'raw_metrics': rm,
                      'source_metrics': sm, 'speed_ratio_vs_v8': speed_ratio, 'acceleration_ratio_vs_v8': acceleration_ratio,
                      'jerk_ratio_vs_v8': jerk_ratio, 'high_frequency_ratio_vs_v8': hf_ratio, 'root_fast_ratio_vs_v8': root_fast_ratio,
                      'support_ratio_vs_v8': rm['geometric_support_fraction'] / max(sm['geometric_support_fraction'], 1e-9),
                      'support_slide_ratio_vs_v8': rm['geometric_slide_p95_m_s'] / max(sm['geometric_slide_p95_m_s'], 1e-9),
                      'calibration': calibration, 'pass': True}
    write_json(OUT / 'RAW_SAFETY_AND_PER_JOINT_GATES.json', {'status': 'complete', 'rows': rows, 'gate_scope': 'saved raw 30 Hz body before mapping'})
    print(json.dumps({'stage': 'correct_bodies', 'status': 'complete', 'songs': list(rows)}, indent=2))


def _angles_for_body(body, track, algorithm, cfg, outputs):
    time = np.arange(N) / 40.; body40 = _resample_body_full_interval(body, np.arange(len(body)) / 30., time)
    left, right, _, _, _ = _load_p2_colors(Path(track['colors']['path']), Path(track['audio']), None)
    left, right = left[:N], right[:N]; loud, _ = load_verified_loudness(Path(track['loudness']['path']), Path(track['audio']))
    dmx, patch, intent, bright, hard = compile_world(body40, time, outputs, left, right, loud, cfg, mapper_id=algorithm)
    return time, body40, dmx, patch, intent, bright, hard


def _angular_gate(candidate, reference, intent, patch, plan):
    decoded = decode(candidate, patch); ref = decode(reference[0], reference[1])
    moving = [i for i, p in enumerate(patch) if p['kind'] != 'bulb']
    ca, ra = decoded['angles_deg'][:, moving], ref['angles_deg'][:, moving]
    cr, rr = np.ptp(ca, axis=0), np.ptp(ra, axis=0)
    cdir = decoded['directions'][:, moving]; rdir = ref['directions'][:, moving]
    cexc = np.rad2deg(np.arccos(np.clip(np.sum(cdir * cdir[:1], axis=-1), -1., 1.)).max(axis=0))
    rexc = np.rad2deg(np.arccos(np.clip(np.sum(rdir * rdir[:1], axis=-1), -1., 1.)).max(axis=0))
    ratios = cexc / np.maximum(rexc, 1.)
    active = (np.linalg.norm(np.diff(ca, axis=0), axis=-1) > .01).mean(axis=0)
    target_pan, target_tilt = [], []
    from scripts.motion.run_body_portfolio import hard_checks
    desired = []
    for i in moving:
        from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
        p, t = _angles(intent['targets'][:, i] - np.asarray(patch[i]['position']))
        desired.append(np.column_stack((p, t)))
    desired = np.stack(desired, axis=1); err = np.sqrt(np.mean((desired - ca) ** 2))
    axis_ratios = cr / np.maximum(rr, .5)
    saturation = ((np.isclose(intent['targets'][:, moving, 0], -7.5) | np.isclose(intent['targets'][:, moving, 0], 7.5)
                   | np.isclose(intent['targets'][:, moving, 1], 0.) | np.isclose(intent['targets'][:, moving, 1], 5.)
                   | np.isclose(intent['targets'][:, moving, 2], -5.8) | np.isclose(intent['targets'][:, moving, 2], -.2))).mean()
    speed_max = float(np.max(np.linalg.norm(np.diff(ca, axis=0) * 40., axis=-1)))
    acceleration_max = float(np.max(np.linalg.norm(np.diff(ca, n=2, axis=0) * 40.**2, axis=-1)))
    gate = bool(np.all(axis_ratios >= plan['gates']['output_axis_range_ratio_min']) and np.all(axis_ratios <= plan['gates']['output_axis_range_ratio_max'])
                and np.all(ratios >= plan['gates']['output_angular_ratio_min']) and np.all(ratios <= plan['gates']['output_angular_ratio_max'])
                and np.all(cexc >= plan['gates']['output_direction_excursion_min_deg'])
                and np.all(active >= plan['gates']['output_active_fraction_min']) and err <= plan['gates']['decoded_angle_rmse_max_deg']
                and saturation <= plan['gates']['target_saturation_fraction_max'])
    return decoded, {'per_output_axis_range_ratio_to_d7': axis_ratios.tolist(),
                     'per_output_direction_excursion_ratio_to_d7': ratios.tolist(),
                     'per_output_direction_excursion_deg': cexc.tolist(), 'per_output_active_fraction': active.tolist(),
                     'min_range_ratio': float(np.min(ratios)), 'max_range_ratio': float(np.max(ratios)),
                     'min_direction_excursion_deg': float(np.min(cexc)), 'target_saturation_fraction': float(saturation),
                     'decoded_speed_max_deg_s': speed_max, 'decoded_acceleration_max_deg_s2': acceleration_max,
                     'min_active_fraction': float(np.min(active)), 'decoded_angle_rmse_deg': float(err), 'passed': gate}


def map_all():
    plan = json.loads((OUT / 'PLAN.json').read_text()); safety = json.loads((OUT / 'RAW_SAFETY_AND_PER_JOINT_GATES.json').read_text())
    cfg = json.loads((ROOT / 'configs/dance/fixed_c.json').read_text())['lighting']['transfer']; outputs = make_outputs(**plan['counts']); refs = {}
    for t in plan['tracks']:
        song = t['id']; body, _ = load_world(OUT / 'bodies' / song / 'body.npz'); source, _ = load_world(t['v8_body']); d7, _ = load_world(t['d7_reference_body'])
        for algorithm in plan['algorithms']:
            aid = algorithm['id']; time, body40, dmx, patch, intent, bright, hard = _angles_for_body(body, t, aid, cfg, outputs)
            _, _, dmxr, patchr, _, _, _ = _angles_for_body(d7, t, aid, cfg, outputs)
            _, _, dmxs, patchs, _, _, _ = _angles_for_body(source, t, aid, cfg, outputs)
            decoded, gate = _angular_gate(dmx, (dmxr, patchr), intent, patch, plan)
            source_decoded = decode(dmxs, patchs); moving = [i for i, p in enumerate(patch) if p['kind'] != 'bulb']
            source_angles = source_decoded['angles_deg'][:, moving]
            speed_ratio = gate['decoded_speed_max_deg_s'] / max(float(np.max(np.linalg.norm(np.diff(source_angles, axis=0) * 40., axis=-1))), 1e-6)
            accel_ratio = gate['decoded_acceleration_max_deg_s2'] / max(float(np.max(np.linalg.norm(np.diff(source_angles, n=2, axis=0) * 40.**2, axis=-1))), 1e-6)
            gate['decoded_speed_ratio_vs_v8'] = speed_ratio; gate['decoded_acceleration_ratio_vs_v8'] = accel_ratio
            gate['passed'] = bool(gate['passed'] and speed_ratio <= plan['gates']['decoded_speed_ratio_max'] and accel_ratio <= plan['gates']['decoded_acceleration_ratio_max'])
            if not gate['passed']: raise RuntimeError(f'angular/readability gate failed for {aid}/{song}: {gate}')
            diag = measure(body40, intent, dmx, patch, fps=40., start_s=0., duration_s=180.)
            folder = OUT / 'timelines' / aid / song
            write_npz(folder / 'timeline.npz', time=time, body=body40, universes=dmx, targets=intent['targets'], dimmer=intent['dimmer'],
                      source_points=intent['source_points'], field_points=intent['field_points'], brightness_level=bright['level'] if 'level' in bright else bright['envelope']['level'])
            write_json(folder / 'patch.json', patch)
            report = {'status': 'engineering_valid', 'algorithm': aid, 'song': song, 'frames': N, 'fixtures': len(outputs),
                      'body_sha256': sha(OUT / 'bodies' / song / 'body.npz'), 'timeline_sha256': sha(folder / 'timeline.npz'),
                      'patch_sha256': sha(folder / 'patch.json'), 'checkpoint': plan['checkpoint'], 'audio_sha256': t['audio_sha256'],
                      'd4max': False, 'd6': False, 'motion_limits': False, 'raw_safety_gate': safety['rows'][song],
                      'angular_readability_gate': gate, 'diagnostics': diag, 'mapping_projection': intent.get('mapping_projection', 'affine_stage_clip_v1'),
                      'scope': 'D7-calibrated bone-preserving native-body intervention; bounded softclip mapper v4; no retrained checkpoint'}
            write_json(folder / 'report.json', report); print(f'ADMITTED {aid} {song}', flush=True)
    write_json(OUT / 'PRE_RENDER_RELEASE.json', {'status': 'all_30_pairs_passed', 'pairs': 30, 'source': str(OUT / 'PLAN.json')})
    print(json.dumps({'stage': 'map', 'pairs': 30, 'status': 'complete'}, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'correct_bodies', 'map')); args = parser.parse_args()
    {'freeze': freeze, 'correct_bodies': correct_bodies, 'map': map_all}[args.stage]()


if __name__ == '__main__': main()
