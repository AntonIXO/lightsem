"""Fixed P2 colors and historical brightness applied to the generated body."""
from pathlib import Path
import json
import os
import subprocess
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


def _load_p2_colors(color_path, audio_path, frames=None, *, allow_fixture_report=False):
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
    expected = report.get('P2_sha256')
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
    count = len(next(iter(controls.values())))
    rows = {}
    for index in range(count):
        key = tuple(np.ascontiguousarray(controls[name][:, index]).tobytes()
                    for name in ('angles_deg', 'directions', 'rgb', 'dimmer'))
        rows.setdefault(key, []).append(index)
    return [indices for indices in rows.values() if len(indices) > 1]


def prepare_colors(audio, root, config, *, color_path=None, audio_python=None, audio_device=None):
    _, cached = cached_track(config, audio)
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
        color_path, audio, allow_fixture_report=False)
    identity['color_active_present'] = active is not None
    return left, right, palette, identity


def map_lights(body, audio, root, config, *, counts=None, color_path=None,
               audio_python=None, audio_device=None):
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
    if len(left) != frames or len(right) != frames:
        raise ValueError('P2 colors must cover exactly the complete original 40 Hz song')
    time = np.arange(frames, dtype=np.float64) / _LIGHTING_FPS
    if left.shape != (frames, 3) or right.shape != left.shape:
        raise ValueError('P2 colors must cover the full original40Hz song')
    world, body_time = _validate_body_for_lighting(body, float(time[-1]))
    world40 = _resample_body_full_interval(world, body_time, time)
    counts = _normalise_counts(config['lighting']['counts'] if counts is None else counts)
    outputs = make_outputs(**counts)
    universes, patch, intent, brightness, hard = compile_world(world40, time, outputs, left, right,
                                                             levels, config['lighting']['transfer'],
                                                             mapper_id=config['lighting']['mapper'])
    controls = decode(universes, patch)
    if not all(np.isfinite(value).all() for value in controls.values()):
        raise ValueError('Nonfinite actual emitted DMX')
    attribution = np.zeros((len(outputs), 22), dtype=np.uint8)
    for index, joints in enumerate(intent.get('attribution', ())):
        attribution[index, np.asarray(joints, dtype=int)] = 1
    duplicate_groups = _stream_duplicate_groups(controls)
    with (root / 'lights.npz').open('xb') as stream:
        np.savez_compressed(stream, time=time, body=world40, universes=universes,
                            targets=intent['targets'], dimmer=intent['dimmer'],
                            source_points=intent['source_points'], field_points=intent['field_points'],
                            attribution_mask=attribution, palette_ids40=palette)
    write_json(root / 'patch.json', patch)
    report = {'mapper': intent.get('algorithm_id', config['lighting']['mapper']), 'color': color_identity,
              'loudness': level_identity, 'counts': counts, 'fixture_motion_limits': False,
              'hard_checks': hard, 'lights_sha256': sha(root / 'lights.npz'),
              'body_sha256': sha(root / 'body.npz'),
              'brightness_scope': 'Historical full-song LUFS law with established body modulation',
              'palette_occupancy_ticks': {str(int(k)): int(v) for k, v in zip(*np.unique(palette, return_counts=True))},
              'actual_rgb_transition_ticks': int(np.any(np.diff(controls['rgb'], axis=0), axis=(1, 2)).sum()),
              'emitted_dimmer_min': float(controls['dimmer'].min()),
              'emitted_dimmer_max': float(controls['dimmer'].max()),
              'mapper_attribution': {'fixture_count': len(outputs),
                                     'covered_joint_count': int(np.count_nonzero(attribution.any(axis=0))),
                                     'uncovered_joints': np.flatnonzero(~attribution.any(axis=0)).tolist(),
                                     'exact_duplicate_fixture_groups': duplicate_groups},
              'finite': True,
              # This function has no frozen-body/limb intervention input.  Do
              # not claim that source-dependence was proven merely because the
              # mapper is structurally source-based.
              'source_dependence_checks': {'status': 'not_checked', 'no_autonomous_motion_source': None},
              'no_autonomous_motion_source': None}
    write_json(root / 'LIGHTING.json', report)
    return time, patch, controls, report
