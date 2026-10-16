"""Fixed P2 colors and historical brightness applied to the generated body."""
from pathlib import Path
import json
import os
import subprocess
from contextlib import nullcontext
import numpy as np

from .baseline import REPO, checked_asset, sha
from .pipeline import cached_track, write_json


_LIGHTING_FPS = 40.0
_BODY_FPS = 30.0


def _exact_clock(values, fps, name, *, minimum=2):
    """Validate a zero-based regular clock before it reaches a mapper.

    ``np.interp`` would otherwise accept a shifted or irregular clock and hide
    a corrupt cache by silently changing the phase of every output.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a finite one-dimensional clock')
    expected = np.arange(len(values), dtype=np.float64) / float(fps)
    if not np.array_equal(values, expected) and not np.allclose(values, expected, rtol=0., atol=1e-8):
        raise ValueError(f'{name} must be an exact zero-based {fps:g} Hz clock')
    return values


def _normalise_counts(counts):
    """Return explicit fixture counts with no truthiness or implicit defaults."""
    if counts is None or not hasattr(counts, 'keys'):
        raise ValueError('Fixture counts are required')
    required = ('beams', 'lasers', 'bulbs')
    if set(counts) != set(required):
        raise ValueError(f'Fixture counts must contain exactly {required}')
    result = {}
    for key in required:
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 or value > 10000:
            raise ValueError(f'Fixture count {key!r} must be an integer in 0..10000')
        result[key] = int(value)
    if sum(result.values()) < 1:
        raise ValueError('At least one fixture is required')
    return result


def _load_p2_colors(color_path, audio_path, frames=None, *, expected_sha256=None,
                    allow_fixture_report=False):
    """Load an authenticated P2 stream and verify its 40 Hz schema/clock.

    Historical files use ``left``/``right``; a few early native files use the
    explicit ``*_rgb`` spelling.  Both are accepted, but the timestamp and
    waveform binding are mandatory for production artifacts.
    """
    color_path = Path(color_path).resolve()
    report_path = color_path.parent / 'REPORT.json'
    if not color_path.is_file() or not report_path.is_file():
        raise FileNotFoundError('P2 colors require P2.npz and its REPORT.json')
    report = json.loads(report_path.read_text())
    fixture_report = bool(report.get('test_fixture'))
    audio_digest = sha(audio_path) if Path(audio_path).is_file() else None
    if report.get('audio_sha256') != audio_digest and not (allow_fixture_report and fixture_report):
        raise ValueError('P2 colors belong to a different waveform')
    # Production reports seal the output either directly or through artifacts.
    expected = expected_sha256 or report.get('P2_sha256')
    artifacts = report.get('artifacts', {})
    if expected is None:
        expected = artifacts.get(str(color_path)) or artifacts.get(color_path.name)
    if expected is None and isinstance(artifacts, dict):
        for artifact_path, artifact_sha in artifacts.items():
            try:
                if Path(artifact_path).resolve() == color_path:
                    expected = artifact_sha
                    break
            except (OSError, TypeError):
                continue
    if expected is not None and expected != sha(color_path):
        raise ValueError('P2 output changed after its report was written')
    if expected is None and not (allow_fixture_report and fixture_report):
        raise ValueError('P2 report does not authenticate P2.npz')
    with np.load(color_path, allow_pickle=False) as data:
        left_key = 'left_rgb' if 'left_rgb' in data else 'left'
        right_key = 'right_rgb' if 'right_rgb' in data else 'right'
        if left_key not in data or right_key not in data:
            raise ValueError('P2.npz must contain left/right RGB streams')
        left, right = data[left_key].copy(), data[right_key].copy()
        palette = data['palette_ids40'].copy() if 'palette_ids40' in data else None
        active = data['color_active40'].copy() if 'color_active40' in data else None
        timestamp = data['time'].copy() if 'time' in data else None
    if (left.ndim != 2 or left.shape[1] != 3 or right.shape != left.shape
            or left.dtype != np.uint8 or right.dtype != np.uint8):
        raise ValueError('P2 RGB streams must be matching uint8 [frames,3] arrays')
    if palette is None and not (allow_fixture_report and fixture_report):
        raise ValueError('P2.npz must contain palette_ids40')
    if palette is not None and (palette.ndim != 1 or len(palette) != len(left)):
        raise ValueError('P2 palette_ids40 must have one value per lighting frame')
    if active is not None and (active.ndim != 1 or len(active) != len(left)):
        raise ValueError('P2 color_active40 must have one value per lighting frame')
    if timestamp is None:
        if not (allow_fixture_report and fixture_report):
            raise ValueError('P2.npz must contain its 40 Hz timestamps')
    else:
        _exact_clock(timestamp, _LIGHTING_FPS, 'P2 timestamps')
        if len(timestamp) != len(left):
            raise ValueError('P2 timestamps and RGB streams have different lengths')
    if frames is not None and len(left) != int(frames):
        raise ValueError('P2 colors must cover exactly the requested lighting interval')
    identity = {'path': str(color_path), 'sha256': sha(color_path),
                'report': str(report_path), 'report_sha256': sha(report_path),
                'audio_sha256': report.get('audio_sha256'), 'schema_version': report.get('schema_version')}
    return left, right, palette, active, identity


def _validate_body_for_lighting(body, end_time):
    world = np.asarray(body.world)
    clock = np.asarray(body.time, dtype=np.float64)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or len(world) != len(clock) or not np.isfinite(world).all():
        raise ValueError('Body world and timestamps must be finite and have matching [T,22,3] shape')
    _exact_clock(clock, _BODY_FPS, 'Native body timestamps')
    # A ceil(duration*30) native sequence represents ``len(world)/30`` seconds
    # even though its last sample is one frame earlier.  Permit only that
    # explicit terminal sample interval; a larger deficit is a short body.
    if float(end_time) > len(world) / _BODY_FPS + 1e-8:
        raise ValueError('Native body does not cover the complete lighting interval')
    return world, clock


def _resample_body_full_interval(world, clock, target_time):
    """Resample without clipping; extend only the declared final native tick."""
    from lightsem.experiments.dynamic_dance.retiming import resample_body
    target_time = np.asarray(target_time, dtype=np.float64)
    if target_time[-1] <= clock[-1] + 1e-8:
        return resample_body(world, target_time, _BODY_FPS)
    if target_time[-1] > len(world) / _BODY_FPS + 1e-8:
        raise ValueError('Target lighting clock exceeds native body duration')
    extended_world = np.concatenate((world, world[-1:]), axis=0)
    extended_clock = np.arange(len(extended_world), dtype=np.float64) / _BODY_FPS
    return resample_body(extended_world, target_time, _BODY_FPS)


def _stream_duplicate_groups(controls):
    """Return exact duplicate emitted streams by fixture index."""
    count = np.asarray(controls['dimmer']).shape[1]
    rows = {}
    for index in range(count):
        key = tuple(np.ascontiguousarray(controls[name][:, index]).tobytes()
                    for name in ('angles_deg', 'directions', 'rgb', 'dimmer'))
        rows.setdefault(key, []).append(index)
    return [indices for indices in rows.values() if len(indices) > 1]


def _source_dependence_audit(world40, time, outputs, left, right, levels, transfer,
                             mapper_id, intent, compile_world):
    """Check frozen-body stability and assigned-joint response on a short prefix."""
    from lightsem.conductor.body_transfer import decode

    frames = min(len(world40), 120)
    starts = sorted(set((0, max(0, (len(world40) - frames) // 2), max(0, len(world40) - frames))))
    frozen_angle_range = 0.
    frozen_direction_range = 0.
    patch = None
    dependencies = {}
    moving_dependencies = {}
    for output, joints in enumerate(intent.get('attribution', ())):
        for joint in joints:
            dependencies.setdefault(int(joint), []).append(output)
            if output < len(outputs) and outputs[output].get('kind') != 'bulb':
                moving_dependencies.setdefault(int(joint), []).append(output)
    responses = {}
    for start in starts:
        body = np.asarray(world40[start:start + frames], dtype=np.float64)
        clock = np.arange(frames, dtype=np.float64) / _LIGHTING_FPS
        left_window, right_window = left[start:start + frames], right[start:start + frames]
        frozen = np.repeat(body[:1], frames, axis=0)

        def emit(value):
            universes, current_patch, _, _, _ = compile_world(
                value, clock, outputs, left_window, right_window, levels,
                transfer, mapper_id=mapper_id)
            return decode(universes, current_patch), current_patch

        frozen_controls, patch = emit(frozen)
        frozen_angle_range = max(frozen_angle_range, float(np.max(np.ptp(frozen_controls['angles_deg'], axis=0))) if len(frozen_controls['angles_deg']) else 0.)
        frozen_direction_range = max(frozen_direction_range, float(np.max(np.ptp(frozen_controls['directions'], axis=0))) if len(frozen_controls['directions']) else 0.)
        for joint, fixture_indices in sorted(dependencies.items()):
            if joint == 0:
                responses.setdefault(str(joint), {
                    'fixture_indices': fixture_indices,
                    'moving_fixture_indices': [index for index in fixture_indices if patch[index]['kind'] != 'bulb'],
                    'responsive': None, 'moving_responsive': None,
                    'unobservable': 'root translation is intentionally removed by the mapper reference frame',
                })
                continue
            perturbed = frozen.copy(); perturbed[:, joint, 0] += .05
            changed, _ = emit(perturbed)
            moving_indices = [index for index in fixture_indices if patch[index]['kind'] != 'bulb']
            delta = np.linalg.norm(changed['directions'][:, fixture_indices] - frozen_controls['directions'][:, fixture_indices], axis=-1)
            dimmer_delta = np.abs(changed['dimmer'][:, fixture_indices] - frozen_controls['dimmer'][:, fixture_indices])
            moving_delta = np.linalg.norm(changed['directions'][:, moving_indices] - frozen_controls['directions'][:, moving_indices], axis=-1)
            row = responses.setdefault(str(joint), {
                'fixture_indices': fixture_indices, 'moving_fixture_indices': moving_indices,
                'max_direction_delta': 0., 'max_moving_direction_delta': 0., 'max_dimmer_delta': 0.,
                'responsive': True, 'moving_responsive': True, 'windows': []})
            response = max(float(np.max(delta)) if delta.size else 0., float(np.max(dimmer_delta)) if dimmer_delta.size else 0.)
            row['max_direction_delta'] = max(row['max_direction_delta'], float(np.max(delta)) if delta.size else 0.)
            row['max_moving_direction_delta'] = max(row['max_moving_direction_delta'], float(np.max(moving_delta)) if moving_delta.size else 0.)
            row['max_dimmer_delta'] = max(row['max_dimmer_delta'], float(np.max(dimmer_delta)) if dimmer_delta.size else 0.)
            row['responsive'] = row['responsive'] and response > 1e-7
            row['moving_responsive'] = row['moving_responsive'] and (bool(moving_delta.size and np.max(moving_delta) > 1e-7) if moving_indices else True)
            row['windows'].append({'start_frame': int(start), 'responsive': response > 1e-7,
                                   'moving_responsive': bool(moving_delta.size and np.max(moving_delta) > 1e-7) if moving_indices else None})
    frozen_stable = frozen_angle_range <= 1e-7 and frozen_direction_range <= 1e-7
    assigned_responsive = bool(responses) and all(
        row['moving_responsive'] if row.get('moving_fixture_indices') else row['responsive']
        for row in responses.values() if not row.get('unobservable'))
    return {
        'status': 'pass' if frozen_stable and assigned_responsive else 'failed',
        'scope': 'Three 120-frame frozen-body and assigned-joint perturbation windows (start/middle/end)',
        'frozen_motion_stable': frozen_stable,
        'frozen_angle_range_deg': frozen_angle_range,
        'frozen_direction_range': frozen_direction_range,
        'window_start_frames': starts,
        'assigned_joint_responses': responses,
        'uncovered_joints': sorted(set(range(22)) - set(dependencies)),
        'moving_uncovered_joints': sorted(set(range(22)) - set(moving_dependencies)),
        'moving_coverage_status': 'complete' if len(moving_dependencies) == 22 else 'partial',
    }


def _validate_p2_recipe_report(report, color_path, config_path, config_sha256):
    """Accept both sealed P2 receipt schemas used by the native cache.

    The current P2 producer stores the recipe as structured delivery fields,
    while older receipts encoded the same contract in a descriptive string and
    a source binding.  Both forms remain authenticated by ``_load_p2_colors``;
    this check only verifies that the accepted P2 layout is the one configured
    for native rendering.
    """
    recipe = str(report.get('recipe', '')).lower()
    legacy = all(token in recipe for token in
                 ('i_vr_p0', 'palette history', 'two groups', '300ms'))
    delivery = report.get('delivery', {})
    try:
        fade_s = float(delivery.get('fade_s', -1.0))
    except (TypeError, ValueError):
        fade_s = -1.0
    compact = (
        report.get('recipe') == 'P2'
        and delivery.get('layout') == 'two_groups'
        and abs(fade_s - 0.3) <= 1e-9
        and delivery.get('history') == 'own_generated_state'
        and report.get('P2_sha256') == sha(color_path)
    )
    if not (legacy or compact):
        raise ValueError('P2 report does not declare the accepted P2 recipe')
    bindings = report.get('source_bindings', {})
    if bindings:
        if bindings.get(str(config_path)) != config_sha256:
            raise ValueError('P2 report is not bound to the configured color recipe')
    elif not compact:
        raise ValueError('P2 report is not bound to the configured color recipe')


def prepare_colors(audio, root, config, *, color_path=None, audio_python=None, audio_device=None):
    _, cached = cached_track(config, audio)
    expected_sha256 = cached['colors']['sha256'] if cached is not None else None
    if color_path is not None:
        color_path = Path(color_path).resolve()
    elif cached is not None:
        color_path = checked_asset(cached['colors'])
        checked_asset(cached['color_report'])
    else:
        command = [audio_python or config['extraction']['python'], '-m', 'lightsem.dance.p2',
                   '--wav', str(audio), '--out', str(root / 'colors'),
                   '--device', audio_device or config['extraction']['device']]
        subprocess.run(command, cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO)}, check=True)
        color_path = root / 'colors/P2.npz'
    # The caller supplies the expected frame count once the source waveform is
    # inspected.  Loading here still verifies the report and clock; map_lights
    # performs the exact whole-song length check below.
    left, right, palette, active, identity = _load_p2_colors(
        color_path, audio, expected_sha256=expected_sha256, allow_fixture_report=False)
    report = json.loads(Path(identity['report']).read_text())
    config_path = (REPO / config['lighting']['color_config']).resolve()
    _validate_p2_recipe_report(report, color_path, config_path,
                               config['lighting']['color_config_sha256'])
    identity['color_active_present'] = active is not None
    return left, right, palette, identity


def map_lights(body, audio, root, config, *, counts=None, color_path=None,
               audio_python=None, audio_device=None, fixture_rgb=None,
               color_experiment=None, music=None, ssl_light=None):
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
    from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
    from lightsem.experiments.motion_study.audio_dynamics import loudness
    from lightsem.experiments.dynamic_dance.retiming import resample_body
    from lightsem.conductor.body_transfer_v2 import make_outputs
    from lightsem.conductor.body_transfer import decode
    import soundfile as sf
    color_config = REPO / config['lighting']['color_config']
    if sha(color_config) != config['lighting']['color_config_sha256']:
        raise ValueError('Accepted P2 color configuration changed')
    _, cached = cached_track(config, audio)
    if cached:
        receipt = checked_asset(cached['loudness_receipt'])
    else:
        loudness(audio, root / 'loudness')
        receipt = root / 'loudness/loudness.json'
    levels, level_identity = load_verified_loudness(receipt, audio)
    info = sf.info(audio)
    duration = info.frames / info.samplerate
    frames = int(np.ceil(duration * _LIGHTING_FPS - 1e-9))
    if frames < 2:
        raise ValueError('Audio must provide at least two lighting frames')
    left, right, palette, color_identity = prepare_colors(audio, root, config, color_path=color_path,
                                                         audio_python=audio_python, audio_device=audio_device)
    color_active = None
    color_cues10 = None
    with np.load(color_identity['path'], allow_pickle=False) as color_arrays:
        if 'color_active40' in color_arrays:
            color_active = color_arrays['color_active40'].copy()
        if 'color_cues10' in color_arrays:
            color_cues10 = color_arrays['color_cues10'].copy()
    if len(left) != frames or len(right) != frames:
        raise ValueError('P2 colors must cover exactly the complete original 40 Hz song')
    time = np.arange(frames, dtype=np.float64) / _LIGHTING_FPS
    if left.shape != (frames, 3) or right.shape != left.shape:
        raise ValueError('P2 colors must cover the full original40Hz song')
    world, body_time = _validate_body_for_lighting(body, float(time[-1]))
    world40 = _resample_body_full_interval(world, body_time, time)
    counts = _normalise_counts(config['lighting']['counts'] if counts is None else counts)
    outputs = make_outputs(**counts)
    color_experiment_report = color_experiment
    if fixture_rgb is None and config['lighting'].get('color_experiment'):
        from .color_field import continuous_color_field
        spec = config['lighting']['color_experiment']
        fixture_rgb, color_experiment_report = continuous_color_field(
            world40, left, right, outputs,
            palette=tuple(spec['palette']),
            spatial_cycles=float(spec['spatial_cycles']),
            temporal_gain=float(spec['temporal_gain']),
            breath_depth=float(spec['breath_depth']))
    if fixture_rgb is not None:
        fixture_rgb = np.asarray(fixture_rgb)
        if fixture_rgb.shape != (frames, len(outputs), 3) or fixture_rgb.dtype != np.uint8:
            raise ValueError('fixture_rgb must be uint8 (lighting_frames, fixture_count, 3)')
    universes, patch, intent, brightness, hard = compile_world(world40, time, outputs, left, right,
                                                             levels, config['lighting']['transfer'],
                                                             mapper_id=config['lighting']['mapper'],
                                                             fixture_rgb=fixture_rgb)
    ssl_report = None
    if ssl_light is not None:
        if music is None:
            raise ValueError('SSL light adapter requires the authenticated MusicTimeline')
        if body.metadata.get('d4max_enabled') or body.metadata.get('d6_enabled'):
            raise ValueError('SSL light adapter is only admitted on the plain native-C clock')
        import torch
        from lightsem.experiments.body_light_ssl.adapter import SSLAdapter
        checkpoint = Path(ssl_light['checkpoint']).expanduser().resolve()
        expected_checkpoint = ssl_light.get('checkpoint_sha256')
        if expected_checkpoint is not None and sha(checkpoint) != expected_checkpoint:
            raise ValueError('SSL light checkpoint SHA256 mismatch')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from lightsem.experiments.stage11.stage11_common import gpu_lock
        with gpu_lock() if device.type == 'cuda' else nullcontext():
            adapter = SSLAdapter.load(checkpoint, device, tuple(ssl_light.get('train_ids', ())))
            universes, ssl_report = adapter.apply(
                body_world=world, music=music,
                base92_path=Path(body.metadata['audio']['base92']),
                controls=decode(universes, patch), universes=universes, patch=patch,
                time40=time, color_active40=color_active, device=device,
                max_residual_deg=float(ssl_light.get('max_residual_deg', 1.0)),
                max_speed=float(ssl_light.get('max_speed_deg_s', 20.0)),
                max_accel=float(ssl_light.get('max_accel_deg_s2', 40.0)),
                max_ratio=float(ssl_light.get('max_candidate_baseline_ratio', 1.25)))
    controls = decode(universes, patch)
    if not all(np.isfinite(value).all() for value in controls.values()):
        raise ValueError('Nonfinite actual emitted DMX')
    source_dependence = _source_dependence_audit(
        world40, time, outputs, left, right, levels, config['lighting']['transfer'],
        config['lighting']['mapper'], intent, compile_world)
    from lightsem.conductor.body_transfer_v2.diagnostics import measure
    transfer_diagnostics = measure(world40, intent, universes, patch, fps=_LIGHTING_FPS,
                                   start_s=0., duration_s=len(time) / _LIGHTING_FPS)
    attribution = np.zeros((len(outputs), 22), dtype=np.uint8)
    for index, joints in enumerate(intent.get('attribution', ())):
        attribution[index, np.asarray(joints, dtype=int)] = 1
    duplicate_groups = _stream_duplicate_groups(controls)
    duplicate_groups_named = [[patch[index]['name'] for index in group] for group in duplicate_groups]
    moving_attribution = attribution[np.array([output['kind'] != 'bulb' for output in outputs])]
    with (root / 'lights.npz').open('xb') as stream:
        save_arrays = dict(time=time, body=world40, universes=universes,
                           targets=intent['targets'], dimmer=intent['dimmer'],
                           source_points=intent['source_points'], field_points=intent['field_points'],
                           attribution_mask=attribution, palette_ids40=palette)
        if fixture_rgb is not None:
            save_arrays['fixture_rgb_experiment'] = fixture_rgb
        for key in ('bulb_field_centers', 'limb_speed', 'field_activity'):
            if key in intent:
                save_arrays[key] = intent[key]
        if color_active is not None:
            save_arrays['color_active40'] = color_active
        if color_cues10 is not None:
            save_arrays['color_cues10'] = color_cues10
        np.savez_compressed(stream, **save_arrays)
    write_json(root / 'patch.json', patch)
    report = {'mapper': intent.get('algorithm_id', config['lighting']['mapper']), 'color': color_identity,
              'color_experiment': color_experiment_report,
              'display_label': ('OKLab flowing gradient | source-driven color field'
                                if color_experiment_report is not None else 'P2'),
              'loudness': level_identity, 'counts': counts, 'fixture_motion_limits': False,
              'hard_checks': hard, 'lights_sha256': sha(root / 'lights.npz'),
              'patch_sha256': sha(root / 'patch.json'),
              'body_sha256': sha(root / 'body.npz'),
              'brightness_scope': 'Historical full-song LUFS law with established body modulation',
              'palette_occupancy_ticks': {str(int(k)): int(v) for k, v in zip(*np.unique(palette, return_counts=True))},
              'actual_rgb_transition_ticks': int(np.any(np.diff(controls['rgb'], axis=0), axis=(1, 2)).sum()),
              'actual_rgb_unique_fixture_streams': int(len({np.ascontiguousarray(controls['rgb'][:, i]).tobytes() for i in range(len(outputs))})),
              'palette_transition_ticks': int(np.count_nonzero(np.diff(palette))) if palette is not None and len(palette) > 1 else None,
              'color_cue_count': int(np.count_nonzero(color_cues10)) if color_cues10 is not None else None,
              'color_cue_clock_hz': 10.0 if color_cues10 is not None else None,
              'color_active_ticks': int(np.count_nonzero(color_active)) if color_active is not None else None,
              'emitted_dimmer_min': float(controls['dimmer'].min()),
              'emitted_dimmer_max': float(controls['dimmer'].max()),
              'mapper_attribution': {'fixture_count': len(outputs),
                                     'covered_joint_count': int(np.count_nonzero(attribution.any(axis=0))),
                                     'uncovered_joints': np.flatnonzero(~attribution.any(axis=0)).tolist(),
                                     'moving_covered_joint_count': int(np.count_nonzero(moving_attribution.any(axis=0))) if len(moving_attribution) else 0,
                                     'moving_uncovered_joints': np.flatnonzero(~moving_attribution.any(axis=0)).tolist() if len(moving_attribution) else list(range(22)),
                                     'moving_coverage_status': source_dependence['moving_coverage_status'],
                                     'exact_duplicate_fixture_groups': duplicate_groups,
                                     'exact_duplicate_fixture_names': duplicate_groups_named},
              'transfer_diagnostics': transfer_diagnostics,
              'ssl_light': ssl_report,
              'finite': True,
              'source_dependence_checks': {**source_dependence,
                                           'no_autonomous_motion_source': source_dependence['status'] == 'pass'},
              'no_autonomous_motion_source': source_dependence['status'] == 'pass'}
    write_json(root / 'LIGHTING.json', report)
    return time, patch, controls, report
