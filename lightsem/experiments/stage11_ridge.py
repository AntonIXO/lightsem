"""CPU-only, fixed shallow diagnostic inside Stage 11 H11-D.

    python -m lightsem.experiments.stage11_ridge --root CAMPAIGN_ROOT

This ancillary baseline never opens test tensors and cannot unlock test or
override the neural campaign gates. Fit uses every 40th frame (1 Hz), while
validation covers every coach/frame at 40 Hz with the same two-second EMA R².

Regularization was clarified *before any fit*: minimize the song-weighted
mean squared vector error plus 0.001 times the squared coefficient norm,
with an unpenalized intercept. Each canonical song has mass one, subdivided
equally among recordings and coaches. Thus the sufficient-statistic equation
uses alpha = 0.001 * n_training_songs, keeping regularization strength fixed
across old-only and expanded populations. The exact objective is published
in an authenticated diagnostic protocol before the first tensor is read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .stage11_common import (
    ROOT, artifact_manifest, authenticated_json, digest, json_digest,
    read_authenticated, verify_artifacts, verify_sources,
)
from .stage11_metrics import motion_r2


A_NAMES = ('metric_level', 'vert_period_s', 'vert_bounce_amp', 'com_speed')
INPUT_DIM = 92
FIT_STRIDE = 40
FPS = 40.0
NORMALIZED_RIDGE = 0.001
MATH_THREADS = 2


def hierarchy_weights(entries: list[dict]) -> tuple[dict[str, float], dict]:
    """Per-coach mass; every canonical song contributes total mass exactly one."""
    songs = defaultdict(lambda: defaultdict(list))
    seen = set()
    for entry in entries:
        if entry['key'] in seen:
            raise ValueError('duplicate coach key in ridge population')
        seen.add(entry['key'])
        songs[entry['canonical_id']][entry['recording_id']].append(entry['key'])
    if not songs:
        raise ValueError('ridge fitting requires a nonempty training population')
    weights = {}
    for recordings in songs.values():
        for coaches in recordings.values():
            weight = 1.0 / (len(recordings) * len(coaches))
            weights.update({coach: weight for coach in coaches})
    return weights, {'canonical_songs': len(songs),
                     'recordings': sum(len(recordings) for recordings in songs.values()),
                     'coaches': len(entries), 'song_weight_mass': float(len(songs))}


def empty_statistics() -> dict:
    return {'xtx': np.zeros((INPUT_DIM + 1, INPUT_DIM + 1), np.float64),
            'xty': np.zeros((INPUT_DIM + 1, len(A_NAMES)), np.float64),
            'sampled_frames': 0, 'weight_mass': 0.0}


def accumulate(statistics: dict, x: np.ndarray, y: np.ndarray, coach_weight: float) -> None:
    """Accumulate one coach; retain only 93×93 and 93×4 sufficient statistics."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape[1] != INPUT_DIM or y.shape != (len(x), len(A_NAMES)) or len(x) == 0:
        raise ValueError('ridge samples require nonempty aligned x=(T,92), y=(T,4)')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(coach_weight) or coach_weight <= 0:
        raise ValueError('ridge samples and hierarchy weights must be finite and positive')
    augmented = np.empty((len(x), INPUT_DIM + 1), np.float64)
    augmented[:, :INPUT_DIM], augmented[:, INPUT_DIM] = x, 1.0
    frame_weight = coach_weight / len(x)
    statistics['xtx'] += frame_weight * (augmented.T @ augmented)
    statistics['xty'] += frame_weight * (augmented.T @ y)
    statistics['sampled_frames'] += len(x)
    statistics['weight_mass'] += coach_weight


def solve_ridge(statistics: dict, n_training_songs: int) -> tuple[np.ndarray, float]:
    """Solve with lambda×N regularization on slopes and zero penalty on intercept."""
    if not isinstance(n_training_songs, int) or n_training_songs <= 0:
        raise ValueError('n_training_songs must be a positive integer')
    if not np.isclose(statistics['weight_mass'], n_training_songs, rtol=1e-10, atol=1e-10):
        raise ValueError('ridge hierarchy mass does not equal the number of songs')
    alpha = NORMALIZED_RIDGE * n_training_songs
    lhs = statistics['xtx'].copy()
    lhs[np.arange(INPUT_DIM), np.arange(INPUT_DIM)] += alpha
    if not np.isfinite(lhs).all() or not np.isfinite(statistics['xty']).all():
        raise ValueError('ridge sufficient statistics are nonfinite')
    coefficients = np.linalg.solve(lhs, statistics['xty'])
    if not np.isfinite(coefficients).all():
        raise ValueError('ridge produced nonfinite coefficients')
    return coefficients, alpha


def _populations(data: dict) -> tuple[list[dict], list[dict], list[dict]]:
    entries = data['motion']
    keys, song_parts, recording_ids, paths = set(), {}, {}, {}
    for entry in entries:
        key, song, recording, part = (entry[name] for name in ('key', 'canonical_id', 'recording_id', 'partition'))
        if key in keys or part not in ('train', 'val', 'test'):
            raise ValueError('duplicate coach identity or unknown partition')
        keys.add(key)
        if song in song_parts and song_parts[song] != part:
            raise ValueError('canonical-song split overlap in ridge metadata')
        song_parts[song] = part
        if recording in recording_ids and recording_ids[recording] != (song, part):
            raise ValueError('recording identity spans canonical songs or partitions')
        recording_ids[recording] = (song, part)
        path = str(Path(entry['path']).resolve())
        if path in paths:
            raise ValueError('one tensor path appears under multiple coach identities')
        paths[path] = key
    train = sorted((e for e in entries if e['partition'] == 'train'), key=lambda e: e['key'])
    old = [e for e in train if e['is_old']]
    validation = sorted((e for e in entries if e['partition'] == 'val'), key=lambda e: e['key'])
    if not train or not old or not validation:
        raise ValueError('ridge requires old and expanded training populations and validation coaches')
    return old, train, validation


def _normalization(normalizers: dict) -> dict[str, np.ndarray]:
    norm = normalizers['motion']
    if norm.get('fit_population') != 'old_training_only' or tuple(norm.get('a_names', [])) != A_NAMES:
        raise ValueError('ridge requires pinned old-training normalization and the four deterministic channels')
    arrays = {name: np.asarray(norm[name], np.float64) for name in ('x_mean', 'x_std', 'y_center', 'y_scale')}
    if arrays['x_mean'].shape != (INPUT_DIM,) or arrays['x_std'].shape != (INPUT_DIM,):
        raise ValueError('ridge input normalization must have 92 channels')
    if any(arrays[name].ndim != 1 or len(arrays[name]) < len(A_NAMES) for name in ('y_center', 'y_scale')):
        raise ValueError('ridge target normalization is missing deterministic channels')
    arrays['y_center'], arrays['y_scale'] = arrays['y_center'][:4], arrays['y_scale'][:4]
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('ridge normalization contains nonfinite values')
    if np.any(arrays['x_std'] <= 0) or np.any(arrays['y_scale'] <= 0):
        raise ValueError('ridge normalization scales must be positive')
    return arrays


def _load_allowed_tensor(entry: dict, read_inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    # This guard lives at the only tensor-open point, not just at list selection.
    if entry['partition'] not in ('train', 'val'):
        raise RuntimeError('CPU ridge must never open a test tensor')
    path = Path(entry['path']).resolve()
    sha = digest(path)
    if sha != entry['sha256']:
        raise RuntimeError('ridge input tensor changed: ' + str(path))
    with np.load(path, allow_pickle=False) as tensor:
        x, y = tensor['x'], tensor['y']
    if (x.ndim != 2 or x.shape[1] != INPUT_DIM or len(x) != entry['n_frames']
            or y.ndim != 2 or y.shape[0] != len(x) or y.shape[1] < len(A_NAMES) or len(x) < 2):
        raise ValueError('ridge tensor shape/length differs from its authenticated metadata')
    y = y[:, :4]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('ridge tensor contains nonfinite features/targets')
    if digest(path) != sha:
        raise RuntimeError('ridge input changed during reading: ' + str(path))
    read_inputs[str(path)] = {'sha256': sha, 'bytes': path.stat().st_size}
    return x, y


def _metric_mean(metrics: list[dict]) -> dict:
    def values(key, channel=None):
        return [metric['r2_by_channel'][channel] if channel else metric[key] for metric in metrics
                if (metric['r2_by_channel'][channel] if channel else metric[key]) is not None]
    primary = values('primary_r2')
    channels = {name: values('r2_by_channel', name) for name in A_NAMES}
    return {'primary_r2': float(np.mean(primary)) if primary else None,
            'r2_by_channel': {name: float(np.mean(value)) if value else None for name, value in channels.items()},
            'eligible_units': {'primary': len(primary), **{name: len(value) for name, value in channels.items()}},
            'total_units': len(metrics)}


def _delta(old: dict, expanded: dict) -> dict:
    def difference(a, b):
        return None if a is None or b is None else float(b - a)
    return {'primary_r2': difference(old['primary_r2'], expanded['primary_r2']),
            'r2_by_channel': {name: difference(old['r2_by_channel'][name], expanded['r2_by_channel'][name]) for name in A_NAMES}}


def summarize_validation(coaches: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Average coaches within recording, then recordings within canonical song."""
    recordings = defaultdict(list)
    for row in coaches:
        recordings[(row['canonical_id'], row['recording_id'])].append(row)
    per_recording = []
    for (canonical, recording), rows in sorted(recordings.items()):
        summary = {'canonical_id': canonical, 'recording_id': recording, 'n_coaches': len(rows)}
        for arm in ('old', 'expanded'):
            summary[arm] = _metric_mean([row[arm] for row in rows])
        summary['delta'] = _delta(summary['old'], summary['expanded'])
        per_recording.append(summary)
    songs = defaultdict(list)
    for row in per_recording:
        songs[row['canonical_id']].append(row)
    per_canonical = []
    for canonical, rows in sorted(songs.items()):
        summary = {'canonical_id': canonical, 'n_recordings': len(rows)}
        for arm in ('old', 'expanded'):
            summary[arm] = _metric_mean([row[arm] for row in rows])
        summary['delta'] = _delta(summary['old'], summary['expanded'])
        per_canonical.append(summary)
    total = {arm: _metric_mean([row[arm] for row in per_canonical]) for arm in ('old', 'expanded')}
    total['delta'] = _delta(total['old'], total['expanded'])
    total['aggregation'] = 'equal canonical songs; equal recordings within song; equal coaches within recording'
    return per_recording, per_canonical, total


def _save_coefficients(path: Path, old: np.ndarray, expanded: np.ndarray, norm: dict) -> None:
    if path.exists():
        raise FileExistsError('incomplete ridge output exists; inspect before explicit recovery: ' + str(path))
    temporary = path.with_name(path.name + f'.partial.{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            np.savez(stream, old=old, expanded=expanded, **norm)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_ridge(root=ROOT) -> dict:
    """Run the frozen CPU diagnostic without importing a trainer or touching CUDA."""
    root = Path(root).resolve()
    started = time.monotonic()
    source = verify_sources(root)
    protocol = read_authenticated(root/'protocol.json')
    data = read_authenticated(root/'training_data.json')
    normalizers = read_authenticated(root/'normalizers.json')
    if protocol.get('frozen') is not True:
        raise RuntimeError('ridge diagnostics require the frozen campaign')
    if digest(root/'normalizers.json') != data['normalizers_sha256']:
        raise RuntimeError('ridge normalization binding differs from training_data.json')
    for key, path in (('data_sha256', root/'training_data.json'), ('normalizers_sha256', root/'normalizers.json')):
        if protocol.get(key) != digest(path):
            raise RuntimeError('ridge input binding differs from the frozen campaign: ' + key)
    identity = {'protocol_sha256': digest(root/'protocol.json'), 'data_sha256': digest(root/'training_data.json'),
                'normalizers_sha256': digest(root/'normalizers.json'), 'source_sha256': source['source_sha256']}
    old_entries, expanded_entries, validation = _populations(data)
    norm = _normalization(normalizers)
    weights, populations = {}, {}
    for arm, entries in (('old', old_entries), ('expanded', expanded_entries)):
        weights[arm], populations[arm] = hierarchy_weights(entries)
    recipe = {
        'schema_version': 1, 'identity': identity, 'hypothesis': 'H11-D', 'role': 'ancillary fixed shallow diagnostic',
        'input_dim': INPUT_DIM, 'target_names': list(A_NAMES), 'fit_stride_frames': FIT_STRIDE,
        'fit_sampling_hz': FPS/FIT_STRIDE, 'fit_start_frame': 0, 'validation_fps': FPS,
        'validation_ema_tau_s': 2.0, 'fit_population': 'TRAIN only; old-only and expanded',
        'normalization': 'same pinned old-training input and target normalization for both arms',
        'weighting': 'song mass 1; equal recordings per song; equal coaches per recording; divide coach mass by sampled frames',
        'objective': 'mean_songs(mean_recordings(mean_coaches(mean_sampled_frames(sum_four_targets_squared_error)))) + 0.001*sum_squared_slopes',
        'normalized_regularization': NORMALIZED_RIDGE, 'intercept_penalized': False,
        'effective_alpha': {arm: NORMALIZED_RIDGE*info['canonical_songs'] for arm, info in populations.items()},
        'regularization_clarification': 'Fixed before fitting: replace absolute alpha 1000 with normalized lambda 0.001 to keep regularization constant across population sizes.',
        'populations': populations, 'math_threads': MATH_THREADS, 'hyperparameter_search': False,
        'test_tensors_permitted': False, 'can_override_neural_gates': False,
    }
    output = root/'results'
    output.mkdir(parents=True, exist_ok=True)
    recipe_path, report_path = output/'cpu_ridge_protocol.json', output/'cpu_ridge.json'
    if recipe_path.exists():
        if read_authenticated(recipe_path) != recipe:
            raise RuntimeError('an incompatible CPU ridge protocol is already frozen')
    else:
        authenticated_json(recipe_path, recipe)
    if report_path.exists():
        existing = read_authenticated(report_path)
        if (existing.get('status') != 'completed' or existing.get('identity') != identity
                or existing.get('diagnostic_protocol_sha256') != digest(recipe_path)):
            raise RuntimeError('existing CPU ridge report has a different identity')
        verify_artifacts(existing['inputs'])
        verify_artifacts(existing['artifacts'])
        return existing
    coefficient_path = output/'cpu_ridge_coefficients.npz'
    if coefficient_path.exists():
        raise FileExistsError('incomplete CPU ridge coefficients already exist; explicit recovery is required')
    metadata_paths = [root/name for name in ('protocol.json', 'training_data.json', 'normalizers.json', 'frozen_sources.json')]
    metadata_paths += [Path(str(path)+'.sha256.json') for path in metadata_paths]
    source_paths = [Path(source['snapshot'])/name for name in source['files']]
    read_inputs = artifact_manifest(metadata_paths + source_paths)
    statistics = {arm: empty_statistics() for arm in weights}
    with threadpool_limits(limits=MATH_THREADS):
        for index, entry in enumerate(expanded_entries):
            x, y = _load_allowed_tensor(entry, read_inputs)
            sampled_x = (x[::FIT_STRIDE].astype(np.float64)-norm['x_mean'])/norm['x_std']
            sampled_y = (y[::FIT_STRIDE].astype(np.float64)-norm['y_center'])/norm['y_scale']
            for arm in ('old', 'expanded'):
                if entry['key'] in weights[arm]:
                    accumulate(statistics[arm], sampled_x, sampled_y, weights[arm][entry['key']])
            if index == 0 or (index+1) % 50 == 0 or index+1 == len(expanded_entries):
                print(json.dumps({'stage': 'ridge_fit', 'coaches_read': index+1, 'total_coaches': len(expanded_entries)}), flush=True)
        coefficients, fit_summary = {}, {}
        for arm in ('old', 'expanded'):
            coefficients[arm], alpha = solve_ridge(statistics[arm], populations[arm]['canonical_songs'])
            fit_summary[arm] = {**populations[arm], 'sampled_frames': statistics[arm]['sampled_frames'],
                                'effective_alpha': alpha, 'normalized_regularization': NORMALIZED_RIDGE,
                                'accumulated_weight_mass': statistics[arm]['weight_mass']}
        _save_coefficients(coefficient_path, coefficients['old'], coefficients['expanded'], norm)
        coaches = []
        for index, entry in enumerate(validation):
            x, truth = _load_allowed_tensor(entry, read_inputs)
            normalized_x = (x.astype(np.float64)-norm['x_mean'])/norm['x_std']
            row = {key: entry[key] for key in ('key', 'canonical_id', 'recording_id', 'n_frames')}
            for arm, beta in coefficients.items():
                predicted = (normalized_x @ beta[:INPUT_DIM]+beta[INPUT_DIM])*norm['y_scale']+norm['y_center']
                row[arm] = motion_r2(truth, predicted, A_NAMES, fps=FPS, tau=2.0)
            row['delta'] = _delta(row['old'], row['expanded'])
            coaches.append(row)
            if index == 0 or (index+1) % 20 == 0 or index+1 == len(validation):
                print(json.dumps({'stage': 'ridge_validation', 'coaches_read': index+1, 'total_coaches': len(validation)}), flush=True)
    recordings, songs, summary = summarize_validation(coaches)
    # Metadata and frozen source changes cannot be hidden by a successful fit.
    verify_artifacts(read_inputs)
    report = {
        'schema_version': 1, 'status': 'completed', 'identity': identity,
        'hypothesis': 'H11-D', 'role': 'ancillary fixed shallow diagnostic', 'split': 'val',
        'diagnostic_protocol_path': str(recipe_path), 'diagnostic_protocol_sha256': digest(recipe_path),
        'normalization': recipe['normalization'], 'normalized_regularization': NORMALIZED_RIDGE,
        'fit': fit_summary, 'validation': summary, 'per_coach': coaches,
        'per_recording': recordings, 'per_canonical': songs,
        'n_validation_coaches': len(coaches), 'n_validation_recordings': len(recordings),
        'n_validation_canonical_songs': len(songs), 'test_tensors_read': 0,
        'can_override_neural_gates': False, 'test_unlock_allowed': False,
        'comparability': 'Exposure and architecture differ from the neural arms: fixed linear ridge fits at 1 Hz, while the neural models train on temporal crops. The validation endpoint, cohort and pinned normalization are shared; scores are not directly interchangeable.',
        'inputs': read_inputs,
        'artifacts': artifact_manifest([coefficient_path, recipe_path, Path(str(recipe_path)+'.sha256.json')]),
        'math_threads': MATH_THREADS, 'elapsed_seconds': time.monotonic()-started,
        'completed_at': time.time(),
    }
    authenticated_json(report_path, report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(MATH_THREADS)
    report = run_ridge(args.root)
    print(json.dumps({'status': report['status'], 'report': str(args.root/'results/cpu_ridge.json'),
                      'validation': report['validation'], 'elapsed_seconds': report['elapsed_seconds']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
