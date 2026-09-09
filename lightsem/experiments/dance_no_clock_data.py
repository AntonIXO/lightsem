"""TRAIN/VAL dance tensors and music-only spectral features, without clocks.

40 Hz cache precision is reproduced before the fixed offline anti-alias filter.
The 129-tap 4 Hz Kaiser FIR has 1.6 s context on either side; resample_poly's
deterministic linear edge extension is used before decimation at 0,4,... .
This is an offline representation: 4–5 Hz motion is attenuated and sub-frame
events are not targets. No beat, tempo, pulse-clarity or event stream is read
by the inference extractor, and no TEST tensor is opened by preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.signal import firwin, resample_poly

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dance_no_clock_20260908')
STAGE11 = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
SR, HOP = 44100, 1024
SCALARS = ('rms', 'rms_db', 'flux', 'flux_lf', 'flux_lmid', 'flux_mid', 'flux_hmid',
           'flux_hf', 'energy_lf', 'energy_lmid', 'energy_mid', 'energy_hmid', 'energy_hf',
           'onset', 'onset_ema', 'flux_ema', 'flux_lf_ema', 'flux_hf_ema', 'centroid',
           'flatness', 'rolloff', 'percussive_ratio')
FEATURE_NAMES = tuple(f'mel_{i:02d}' for i in range(64)) + SCALARS
TARGET_NAMES = ('vert_period_s', 'vert_bounce_amp', 'com_speed', 'qom', 'head_speed',
                'foot_speed', 'weight', 'time_eff', 'spread', 'section_energy',
                'hand_speed', 'hand_spread', 'lateral_pos', 'travel_speed', 'symmetry',
                'laterality', 'space_eff', 'flow_eff', 'jerk', 'vert_bounce')
BANDS = {'lf': (50., 100.), 'lmid': (100., 400.), 'mid': (400., 1600.),
         'hmid': (1600., 6400.), 'hf': (6400., 12800.)}
LAGS = (1, 5, 10, 20, 40)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def authenticated_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    temp = path.with_name(path.name+'.tmp')
    temp.write_bytes(raw)
    os.replace(temp, path)
    side = Path(str(path)+'.sha256.json')
    side.write_text(json.dumps({'sha256': hashlib.sha256(raw).hexdigest()})+'\n')


def read_authenticated(path):
    path = Path(path)
    expected = json.loads(Path(str(path)+'.sha256.json').read_text())['sha256']
    if digest(path) != expected:
        raise ValueError('Authenticated input changed: '+str(path))
    return json.loads(path.read_text())


def fir_coefficients():
    return firwin(129, 4., fs=40., window=('kaiser', 8.6)).astype(np.float64)


def downsample_10hz(values):
    values = np.asarray(values)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite time×channel input with at least two frames is required')
    out = resample_poly(values.astype(np.float64), 1, 4, axis=0,
                        window=fir_coefficients(), padtype='line')
    if len(out) != (len(values)+3)//4 or not np.isfinite(out).all():
        raise ValueError('Anti-aliasing produced invalid output')
    return out.astype(np.float32)


def preprocessing_spec():
    taps = fir_coefficients()
    return {'input_fps': 40., 'output_fps': 10., 'decimation': 4, 'output_source_indices': '0,4,8,...',
            'fir_numtaps': 129, 'fir_cutoff_hz': 4., 'fir_window': ['kaiser', 8.6],
            'fir_coefficients': taps.tolist(), 'fir_sha256': hashlib.sha256(taps.tobytes()).hexdigest(),
            'implementation': 'scipy.signal.resample_poly(up=1,down=4,window=explicit_taps,padtype=line)',
            'edge_extension': 'linear extension according to first/last sample, independently for each channel',
            'phase': 'symmetric zero-phase FIR, delay compensated', 'offline_context_each_side_s': 1.6,
            'limitation': '4–5Hz motion deliberately attenuated; frequencies above the10Hz Nyquist cannot be represented; no event targets',
            'input_precision': 'legacy cache parity: mel float16 then float32; scalar float32; filter calculations float64'}


def _ema(values, fps, tau):
    values = np.asarray(values, np.float64)
    a = float(np.exp(-1./(tau*fps)))
    out = np.empty_like(values)
    acc = values[0]
    for i, value in enumerate(values):
        acc = a*acc+(1-a)*value
        out[i] = acc
    return out


def _hold(values, source_times, target_times):
    ids = np.searchsorted(source_times, target_times, side='right')-1
    return np.asarray(values, np.float64)[np.clip(ids, 0, len(source_times)-1)]


def extract_nonclock_audio(path=None, *, y=None, sr=SR, output_fps=10.):
    """Return x/times/names with only86 spectral/envelope features; no clocks."""
    import librosa
    from scipy.ndimage import median_filter
    if y is None:
        if path is None:
            raise ValueError('Provide an audio path or samples')
        y, sr = librosa.load(str(path), sr=SR)
    elif sr != SR:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
        sr = SR
    y = np.asarray(y, np.float32)
    if y.ndim != 1 or len(y) < HOP*4 or not np.isfinite(y).all():
        raise ValueError('Finite mono audio of at least four analysis hops is required')
    if output_fps not in (10., 40.):
        raise ValueError('Only the reviewed40Hz representation or filtered10Hz output is supported')
    n_fft = 2*HOP
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP, center=False))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    t_in = (np.arange(spectrum.shape[1])*HOP+n_fft)/sr
    fps_in = sr/HOP
    duration = len(y)/sr
    t_out = np.arange(round(duration*40))/40.
    channels = {}
    rms = librosa.feature.rms(S=spectrum, frame_length=n_fft, hop_length=HOP)[0]
    channels['rms'] = rms
    channels['rms_db'] = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1., top_db=None)
    log_spectrum = np.log1p(spectrum)
    flux = np.maximum(np.diff(log_spectrum, axis=1, prepend=log_spectrum[:, :1]), 0.)
    channels['flux'] = flux.sum(0)
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        channels['flux_'+name] = flux[mask].sum(0) if mask.any() else np.zeros(spectrum.shape[1])
        channels['energy_'+name] = spectrum[mask].mean(0) if mask.any() else np.zeros(spectrum.shape[1])
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(spectrum**2, top_db=None), sr=sr, hop_length=HOP)
    channels['onset'] = onset[:spectrum.shape[1]] if len(onset) >= spectrum.shape[1] else np.pad(onset, (0, spectrum.shape[1]-len(onset)))
    channels['centroid'] = librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]
    channels['flatness'] = librosa.feature.spectral_flatness(S=spectrum)[0]
    channels['rolloff'] = librosa.feature.spectral_rolloff(S=spectrum, sr=sr, roll_percent=.95)[0]
    # Same causal spectral HPSS ratio, without importing the beat-bearing bus.
    s = spectrum.astype(np.float64)
    harmonic = median_filter(s, size=(1, 31), origin=(0, 15), mode='nearest')
    percussive = median_filter(s, size=(31, 1), origin=(0, 0), mode='nearest')
    p2, h2 = percussive**2, harmonic**2
    mask = p2/np.maximum(p2+h2, 1e-12)
    channels['percussive_ratio'] = (s*mask).sum(0)/np.maximum(s.sum(0), 1e-9)
    smoothed = {'rms', 'percussive_ratio', 'centroid', 'flatness', 'rolloff'}
    control = {}
    for name, values in channels.items():
        values = np.nan_to_num(np.asarray(values, np.float64), nan=0., posinf=0., neginf=0.)
        if name in smoothed:
            values = _ema(values, fps_in, .08)
        control[name] = _hold(values, t_in, t_out)
    for name in ('flux', 'flux_lf', 'flux_hf', 'onset'):
        control[name+'_ema'] = _ema(control[name], 40., .15)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=spectrum**2, sr=sr, n_mels=64), top_db=None)
    mel40 = np.stack([_hold(row, t_in, t_out) for row in mel], 1).astype(np.float16).astype(np.float32)
    scalar40 = np.stack([control[name] for name in SCALARS], 1).astype(np.float32)
    x = np.concatenate([mel40, scalar40], 1)
    if output_fps == 10.:
        x = downsample_10hz(x)
    if x.shape[1] != 86 or not np.isfinite(x).all():
        raise ValueError('Nonclock feature extraction failed')
    return {'x': x, 'times': np.arange(len(x), dtype=np.float64)/output_fps, 'fps': float(output_fps),
            'feature_names': list(FEATURE_NAMES), 'meta': {'source_path': str(path) if path else None,
                'source_duration_s': duration, 'sr': sr, 'hop': HOP, 'algorithmic_clock_features': False,
                'beat_tempo_pulse_extraction_called': False, 'preprocessing': preprocessing_spec() if output_fps == 10. else None}}


def hierarchical_weights(entries):
    groups = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if entry['partition'] != 'train':
            raise ValueError('Normalizer inputs must be TRAIN only')
        groups[entry['canonical_id']][entry['recording_id']].append(entry['key'])
    weights = {}
    for recordings in groups.values():
        for coaches in recordings.values():
            for key in coaches:
                weights[key] = 1./(len(groups)*len(recordings)*len(coaches))
    return weights


def weighted_iqr(values, weights):
    values, weights = np.asarray(values), np.asarray(weights, np.float64)
    result = []
    for column in values.T:
        order = np.argsort(column, kind='stable')
        cdf = np.cumsum(weights[order]);total = cdf[-1]
        lo, hi = [column[order[min(int(np.searchsorted(cdf, q*total, side='left')), len(order)-1)]] for q in (.25, .75)]
        result.append(max(float(hi-lo), 1e-3))
    return result


def _prepare_entry(job):
    entry, root = job
    if entry.get('partition') not in ('train', 'val'):
        raise ValueError('Only TRAIN/VAL tensors may be opened')
    source = Path(entry['path'])
    if digest(source) != entry['sha256']:
        raise ValueError('Stage11 input hash mismatch: '+str(source))
    with np.load(source, allow_pickle=False) as source_arrays:
        if source_arrays['x'].shape != (entry['n_frames'], 92) or source_arrays['y'].shape != (entry['n_frames'], 21):
            raise ValueError('Unexpected Stage11 tensor layout')
        # No b/event array is loaded; no TEST entry is sent to this worker.
        x = downsample_10hz(source_arrays['x'][:, :86])
        y = downsample_10hz(source_arrays['y'][:, 1:])
    destination = root/'tensors'/f'{entry["key"]}.npz'
    temp = destination.with_suffix('.partial.npz')
    np.savez(temp, x=x, y=y)
    os.replace(temp, destination)
    return {**{key: entry[key] for key in ('key', 'canonical_id', 'recording_id', 'partition', 'offset_s', 'is_old')},
            'path': str(destination), 'sha256': digest(destination), 'n_frames': len(x), 'fps': 10.,
            'duration_s': len(x)/10., 'source_duration_s': entry['duration_s'],
            'source_path': str(source), 'source_sha256': entry['sha256'], 'source_n_frames': entry['n_frames'],
            'source_fps': 40., 'source_index_step': 4}


def prepare(root=ROOT, stage11_root=STAGE11, workers=2):
    root, source_root = Path(root).resolve(), Path(stage11_root).resolve()
    if workers not in (1, 2, 3, 4):
        raise ValueError('Use one to four CPU preparation workers')
    if (root/'data_manifest.json').exists():
        return load_manifest(root)
    root.mkdir(parents=True, exist_ok=True)
    source_index = read_authenticated(source_root/'training_data.json')
    source_norm = read_authenticated(source_root/'normalizers.json')
    if source_norm['motion']['a_names']+source_norm['motion']['c_names'] != ['metric_level']+list(TARGET_NAMES):
        raise ValueError('Stage11 target names differ from the reviewed selection')
    entries = [entry for entry in source_index['motion'] if entry['partition'] in ('train', 'val')]
    if any(entry['partition'] == 'test' for entry in entries):
        raise ValueError('TEST access prohibited')
    groups = {part: {entry['canonical_id'] for entry in entries if entry['partition'] == part} for part in ('train', 'val')}
    if groups['train'] & groups['val'] or len(groups['val']) != 40:
        raise ValueError('Source split identity failure')
    spec = preprocessing_spec()
    proposal = {'schema_version': 1, 'method': spec, 'feature_names': list(FEATURE_NAMES), 'target_names': list(TARGET_NAMES),
                'source_index_sha256': digest(source_root/'training_data.json'), 'source_normalizers_sha256': digest(source_root/'normalizers.json'),
                'preparation_code_sha256': digest(__file__), 'partitions_allowed': ['train', 'val'], 'event_targets': False,
                'metric_quantile_sampling': {'maximum_samples_per_coach_per_metric': 256,
                    'indices': 'linspace(0,available_frames-1,min(256,available_frames),dtype=int)',
                    'weights': 'equal canonical song, equal recording within song, equal coach within recording, equal selected frames',
                    'quantile': 'left inverse of weighted empirical CDF, stable sort'}}
    config_path = root/'data_preparation.json'
    if config_path.exists() and read_authenticated(config_path) != proposal:
        raise ValueError('Preparation contract changed; use a new root or explicit reviewed repair')
    authenticated_json(config_path, proposal)
    (root/'tensors').mkdir(exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(_prepare_entry, ((entry, root) for entry in entries)))
    train = [entry for entry in prepared if entry['partition'] == 'train']
    weights = hierarchical_weights(train)
    xm, x2, ym, y2 = np.zeros(86), np.zeros(86), np.zeros(20), np.zeros(20)
    for entry in train:
        with np.load(entry['path'], allow_pickle=False) as z:
            x, y = z['x'].astype(np.float64), z['y'].astype(np.float64)
        weight = weights[entry['key']]
        xm += weight*x.mean(0);x2 += weight*np.square(x).mean(0)
        ym += weight*y.mean(0);y2 += weight*np.square(y).mean(0)
    xs = np.sqrt(np.maximum(x2-xm*xm, 1e-12));ys = np.sqrt(np.maximum(y2-ym*ym, 1e-6))
    samples = {name: [] for name in ('marginal', *map(str, LAGS))}
    sample_weights = {name: [] for name in samples}
    for entry in train:
        with np.load(entry['path'], allow_pickle=False) as z:
            y = (z['y'].astype(np.float64)-ym)/ys
        for name in samples:
            values = y if name == 'marginal' else y[int(name):]-y[:-int(name)]
            indices = np.linspace(0, len(values)-1, min(256, len(values)), dtype=int)
            samples[name].append(values[indices]);sample_weights[name].append(np.full(len(indices), weights[entry['key']]/len(indices)))
    iqrs = {name: weighted_iqr(np.concatenate(samples[name]), np.concatenate(sample_weights[name])) for name in samples}
    norms = {'schema_version': 1, 'x_mean': xm.tolist(), 'x_std': xs.tolist(), 'y_center': ym.tolist(), 'y_scale': ys.tolist(),
             'feature_names': list(FEATURE_NAMES), 'target_names': list(TARGET_NAMES), 'fit_partition': 'train',
             'moment_weighting': 'exact equal-song/recording/coach/frame moments across all filtered training frames',
             'x_std_floor': 1e-6, 'y_scale_floor': 1e-3,
             'constant_target_channels': [name for name, var in zip(TARGET_NAMES, y2-ym*ym) if var <= 1e-12],
             'metric_scales': {'target_space': 'standardized', 'marginal_iqr': iqrs['marginal'],
                               'lag_iqr': {str(lag): iqrs[str(lag)] for lag in LAGS}, 'lags': list(LAGS), 'floor': 1e-3,
                               'sampling': proposal['metric_quantile_sampling']},
             'train_canonical_songs': len(groups['train']), 'train_coaches': len(train)}
    authenticated_json(root/'normalizers.json', norms)
    counts = {}
    for part in ('train', 'val'):
        selected = [entry for entry in prepared if entry['partition'] == part]
        counts[part] = {'coaches': len(selected), 'recordings': len({entry['recording_id'] for entry in selected}),
                        'canonical_songs': len({entry['canonical_id'] for entry in selected}),
                        'frames': sum(entry['n_frames'] for entry in selected)}
    manifest = {'schema_version': 1, 'root': str(root), 'entries': prepared, 'counts': counts,
                'feature_names': list(FEATURE_NAMES), 'target_names': list(TARGET_NAMES), 'x_dim': 86, 'y_dim': 20, 'fps': 10.,
                'normalizers_path': str(root/'normalizers.json'), 'normalizers_sha256': digest(root/'normalizers.json'),
                'preparation_path': str(config_path), 'preparation_sha256': digest(config_path),
                'source_index_path': str(source_root/'training_data.json'), 'source_index_sha256': proposal['source_index_sha256'],
                'test_tensors_read': False, 'event_targets': False, 'clock_features': False,
                'preprocessing': spec, 'source_code_sha256': digest(__file__), 'elapsed_s': time.monotonic()-started}
    authenticated_json(root/'data_manifest.json', manifest)
    return manifest


def load_manifest(root=ROOT, verify_artifacts=True):
    root = Path(root).resolve()
    manifest = read_authenticated(root/'data_manifest.json')
    if manifest['x_dim'] != 86 or manifest['y_dim'] != 20 or manifest['fps'] != 10.:
        raise ValueError('Wrong no-clock dataset dimensions')
    read_authenticated(root/'data_preparation.json')
    read_authenticated(root/'normalizers.json')
    if digest(root/'normalizers.json') != manifest['normalizers_sha256'] or digest(root/'data_preparation.json') != manifest['preparation_sha256']:
        raise ValueError('Dataset transform binding mismatch')
    for entry in manifest['entries']:
        path = Path(entry['path']).resolve()
        if entry['partition'] not in ('train', 'val') or not path.is_relative_to(root/'tensors'):
            raise ValueError('Nonpermitted tensor in new manifest')
        if verify_artifacts and digest(path) != entry['sha256']:
            raise ValueError('No-clock tensor changed: '+str(path))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare');p.add_argument('--stage11-root', type=Path, default=STAGE11);p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('audio');p.add_argument('--path', type=Path, required=True);p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.stage11_root, args.workers)
        print(json.dumps({'status': 'prepared', 'counts': result['counts'], 'manifest_sha256': digest(args.root/'data_manifest.json')}))
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        result = extract_nonclock_audio(args.path)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, x=result['x'], times=result['times'], fps=result['fps'], names=np.asarray(result['feature_names']), meta=json.dumps(result['meta']))
        authenticated_json(Path(str(args.out)+'.metadata.json'), {'path': str(args.out.resolve()), 'sha256': digest(args.out), 'shape': list(result['x'].shape), 'meta': result['meta']})
        print(json.dumps({'status': 'extracted', 'shape': list(result['x'].shape), 'sha256': digest(args.out)}))


if __name__ == '__main__':
    main()
