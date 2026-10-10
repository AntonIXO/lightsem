"""Whole-song, frozen-model observations and waveform counterfactuals.

No audio-derived scalar is labeled an instrument-presence probability. MuQ is
exported at native25Hz; interpolation is a separate downstream operation. The
separator and structure analyzer are real pinned models, with no DSP substitute.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import numpy as np

from .contracts import MusicTimeline, SOURCE_NAMES, SOURCE_FIELDS, MIXTURE_FIELDS, sha256

CAMPAIGN_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
AUDIO_MANIFEST = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/data/audio_manifest.json')
STUDY_SOURCE = Path('/mnt/lightsem-data/experiments/lightsem/audio_understanding_20261004/source_science_v1')
STUDY_HASHES = {
    'features.py': '638afa575af2f5b032a27168ed4dedf5dc89df1e1564fd5bcf5633716c9a87b6',
    'teachers.py': 'b1ed3044317c649ebe9621fc69406dfc99ebf3582bc5b098d15fc7d219dca012',
}
MODEL_LINKS = Path('/root/lightsem-audio-study/model_links')
BEAT_CHECKPOINT = Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/reference_models/beat_this-final0.ckpt')
BEAT_SHA256 = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
SECTION_LABELS = ('start', 'end', 'intro', 'outro', 'break', 'bridge', 'inst', 'solo', 'verse', 'chorus', 'unknown')
SHOWCASE = {
    'shpongle': '/root/shpongle.flac',
    'cure_for_me': '/root/lightsem/experiments/legacy/stages/outputs/stage8/demo_songs/Cure_For_Me.wav',
}
FREE_FLOOR = 20 * 1024 ** 3
RECIPE = {
    'version': 2, 'observation_hz': 30, 'muq_layer': 9, 'muq_native_hz': 25,
    'muq_chunk_seconds': 8, 'muq_batch_size': 4,
    'muq_boundary': 'overlap-aware native25Hz grid; overlapping real8s crops are timestamp-aligned and averaged',
    'separator': 'pinned HTDemucs955717e8; shifts0,overlap0.25,splitTrue',
    'physical_frontend': 'mono arithmetic mean;22050Hz;2048Hann centeredSTFT;hop735;linear magnitude',
    'source_fields': list(SOURCE_FIELDS), 'mixture_fields': list(MIXTURE_FIELDS),
    'source_activity': 'continuous separated waveform descriptors; no ridge threshold or absence classifier',
    'structure': 'official All-In-One, real separator stems reordered explicitly',
}


def write_receipt(path: str | Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refuse completed receipt overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.with_name(path.name + '.sha256.json').write_text(json.dumps({'sha256': sha256(path)}) + '\n')


def read_receipt(path: str | Path) -> dict:
    path = Path(path)
    expected = json.loads(path.with_name(path.name + '.sha256.json').read_text())['sha256']
    if sha256(path) != expected:
        raise ValueError(f'Receipt changed: {path}')
    return json.loads(path.read_text())


def guard_disk(path: str | Path, required_bytes: int = 0) -> None:
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < FREE_FLOOR + required_bytes:
        raise RuntimeError('20GiB free-space floor would be exceeded; no automatic deletion')


def import_study(module: str):
    if module not in ('features', 'teachers'):
        raise ValueError('Only frozen extraction APIs are authorized here')
    file = STUDY_SOURCE / 'audio_study' / (module + '.py')
    if sha256(file) != STUDY_HASHES[module + '.py']:
        raise ValueError('Frozen audio-study implementation changed')
    if str(STUDY_SOURCE) not in sys.path:
        sys.path.insert(0, str(STUDY_SOURCE))
    imported = importlib.import_module('audio_study.' + module)
    if Path(imported.__file__).resolve() != file.resolve():
        raise RuntimeError('Different audio_study source already imported; isolate this process')
    return imported


def interpolate(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values, source_time, target_time = np.asarray(values), np.asarray(source_time), np.asarray(target_time)
    if len(source_time) != len(values) or len(source_time) == 0 or np.any(np.diff(source_time) <= 0):
        raise ValueError('Invalid interpolation source timestamps')
    if not all(np.isfinite(x).all() for x in (values, source_time, target_time)):
        raise ValueError('Interpolation inputs must be finite')
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(target_time, source_time, flat[:, i]) for i in range(flat.shape[1])], 1)
    return result.reshape((len(target_time),) + values.shape[1:]).astype(np.float32)


def physical_observations(waveform: np.ndarray, sample_rate: int, target_time: np.ndarray) -> np.ndarray:
    """RMS, positive energy attack, positive spectral flux, sustained RMS.

    All are linear waveform measurements; the sustain descriptor is a trailing
    half-second RMS mean. Source silence gives zero rather than invented events.
    """
    from scipy import signal
    wave = np.asarray(waveform, np.float64)
    if wave.ndim == 2:
        wave = wave.mean(axis=0)
    if wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all() or sample_rate <= 0:
        raise ValueError('Expected finite mono or channels-first waveform')
    factor = math.gcd(sample_rate, 22050)
    if sample_rate != 22050:
        wave = signal.resample_poly(wave, 22050 // factor, sample_rate // factor)
    sr, hop, size = 22050, 735, 2048
    padded = np.pad(wave, (size // 2, size // 2), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(size)[None], axis=1)) / size
    flux = np.maximum(np.diff(magnitude, axis=0, prepend=magnitude[:1]), 0).mean(axis=1)
    onset = np.maximum(np.diff(rms, prepend=rms[:1]), 0)
    # Uniform trailing mean using convolution avoids a centered "sustain" lookahead.
    sustain = np.convolve(rms, np.ones(15), mode='full')[:len(rms)] / np.minimum(np.arange(len(rms)) + 1, 15)
    features = np.stack((rms, onset, flux, sustain), axis=1)
    return interpolate(features, np.arange(len(features)) * hop / sr, target_time)


def mixture_observations(waveform: np.ndarray, sample_rate: int, target_time: np.ndarray) -> np.ndarray:
    from scipy import signal
    physical = physical_observations(waveform, sample_rate, target_time)
    wave = np.asarray(waveform, np.float64)
    if wave.ndim == 2:
        wave = wave.mean(axis=0)
    factor = math.gcd(sample_rate, 22050)
    if sample_rate != 22050:
        wave = signal.resample_poly(wave, 22050 // factor, sample_rate // factor)
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, (1024, 1024), mode='reflect'), 2048)[::735]
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(2048)[None], axis=1))
    centroid = (magnitude * np.fft.rfftfreq(2048, 1 / 22050)).sum(1) / np.maximum(magnitude.sum(1), 1e-12)
    center = interpolate(centroid[:, None], np.arange(len(centroid)) / 30, target_time)[:, 0]
    return np.stack((physical[:, 0], physical[:, 1], center, physical[:, 2]), axis=1).astype(np.float32)


def section_recurrence(muq: np.ndarray, muq_time: np.ndarray, sections: list[dict]) -> np.ndarray:
    means = []
    for row in sections:
        keep = (muq_time >= row['start']) & (muq_time < row['end'])
        if not keep.any():
            # Very short edge labels still reference the nearest real observation.
            keep[np.argmin(np.abs(muq_time - (row['start'] + row['end']) / 2))] = True
        means.append(np.asarray(muq[keep], np.float64).mean(0))
    if not means:
        return np.empty((0, 0), np.float32)
    means = np.stack(means)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    normalized = means / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1, 1).astype(np.float32)


def _event_distances(time_grid: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(events) == 0:
        return np.full(len(time_grid), 8.), np.full(len(time_grid), 8.)
    before = np.searchsorted(events, time_grid, side='right') - 1
    after = np.searchsorted(events, time_grid, side='left')
    previous = np.where(before >= 0, time_grid - events[np.clip(before, 0, len(events) - 1)], 8.)
    following = np.where(after < len(events), events[np.clip(after, 0, len(events) - 1)] - time_grid, 8.)
    return np.clip(previous, 0, 8), np.clip(following, 0, 8)


def model_features(timeline: MusicTimeline, target_time: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Raw fixed-schema model inputs. Fit normalizers on TRAIN in the caller.

    Result is [T,1061]:1024MuQ+16source+4mixture+6event-distance+11labels.
    Event distances contain no generated motion and no periodic oscillator.
    """
    timeline.validate()
    target = timeline.time if target_time is None else np.asarray(target_time, np.float64)
    muq = interpolate(timeline.muq, timeline.muq_time, target)
    sources = interpolate(timeline.sources, timeline.time, target).reshape(len(target), 16)
    mixture = interpolate(timeline.mixture, timeline.time, target)
    boundaries = np.asarray(sorted(set(float(s['start']) for s in timeline.sections) | set(float(s['end']) for s in timeline.sections)))
    distances = np.stack([v for e in (timeline.beats, timeline.downbeats, boundaries) for v in _event_distances(target, e)], axis=1)
    labels = np.zeros((len(target), len(SECTION_LABELS)), np.float32)
    labels[:, -1] = 1
    for row in timeline.sections:
        mask = (target >= row['start']) & (target < row['end'])
        index = SECTION_LABELS.index(row['label']) if row['label'] in SECTION_LABELS else len(SECTION_LABELS) - 1
        labels[mask] = 0
        labels[mask, index] = 1
    names = ([f'muq_{i}' for i in range(1024)] + [f'{source}_{field}' for source in SOURCE_NAMES for field in SOURCE_FIELDS]
             + [f'mixture_{k}' for k in MIXTURE_FIELDS] + [f'{event}_{side}_s' for event in ('beat', 'downbeat', 'section') for side in ('previous', 'next')]
             + [f'section_{label}' for label in SECTION_LABELS])
    x = np.concatenate((muq, sources, mixture, distances, labels), axis=1).astype(np.float32)
    if x.shape != (len(target), 1061) or not np.isfinite(x).all():
        raise ValueError('Invalid fixed model feature schema')
    return x, names


def source_attack_events(timeline: MusicTimeline) -> list[dict]:
    from scipy.signal import find_peaks
    events = []
    for source, name in enumerate(SOURCE_NAMES):
        strength = timeline.sources[:, source, 1]
        if not np.any(strength > 0):
            continue
        peaks, _ = find_peaks(strength, height=float(np.quantile(strength, .8)), distance=max(1, round(.15 / np.median(np.diff(timeline.time)))))
        events.extend({'time': float(timeline.time[i]), 'source': name, 'strength': float(strength[i]),
                       'evidence': 'positive separated-waveform RMS difference peak; not annotated onset'} for i in peaks)
    return sorted(events, key=lambda e: (e['time'], e['source']))


def preselect_intervals(timeline: MusicTimeline, start: float = 60., end: float = 180.) -> list[dict]:
    """Deterministic audio-only diagnostic windows, frozen before candidate views."""
    time_grid = timeline.time
    valid = (time_grid >= start + 4) & (time_grid <= end - 4)
    if not valid.any():
        raise ValueError('Insufficient song coverage for requested audio-only interval')
    ids = np.flatnonzero(valid)
    rms = timeline.sources[:, :, 0]
    proportions = rms / np.maximum(rms.sum(1, keepdims=True), 1e-12)
    diversity = -(proportions * np.log(np.maximum(proportions, 1e-12))).sum(1)
    rows = []
    for reason, scores in [('texture_thinning', -diversity), ('strong_attack', timeline.mixture[:, 1]), ('sustained_source', timeline.sources[:, :, 3].max(1))]:
        center = float(time_grid[ids[np.argmax(scores[ids])]])
        rows.append({'start': center - 4, 'end': center + 4, 'reason': reason, 'selection': 'audio_only'})
    boundaries = [float(s['start']) for s in timeline.sections if start + 4 <= s['start'] <= end - 4]
    if boundaries:
        center = min(boundaries, key=lambda x: abs(x - (start + end) / 2))
        rows.append({'start': center - 4, 'end': center + 4, 'reason': 'functional_boundary', 'selection': 'audio_only'})
    # Reserve a measured boundary/thinning release candidate plus immediately
    # preceding non-overlapping context. These are diagnostic hypotheses, not
    # human-annotated choreographic gold or self-certifying controller targets.
    candidates = [r for r in rows if r['reason'] in ('texture_thinning', 'functional_boundary') and r['start'] >= start + 8]
    if candidates:
        selected = candidates[-1]
        rows.append({'start': selected['start'], 'end': selected['end'],
                     'reason': selected['reason'], 'role': 'release_hypothesis',
                     'selection': 'audio_only', 'human_annotated': False,
                     'reference_start': selected['start'] - (selected['end'] - selected['start']),
                     'reference_end': selected['start']})
    return rows


def integrated_loudness(waveform: np.ndarray, sample_rate: int) -> float:
    """Measure the actual floating stereo waveform with FFmpeg EBU R128."""
    import re
    import subprocess
    wave = np.asarray(waveform, np.float32)
    if wave.ndim != 2 or wave.shape[0] != 2 or not np.isfinite(wave).all():
        raise ValueError('EBU R128 input must be finite stereo[2,N]')
    completed = subprocess.run(
        ['ffmpeg', '-hide_banner', '-nostdin', '-f', 'f32le', '-ar', str(sample_rate), '-ac', '2',
         '-i', 'pipe:0', '-af', 'ebur128=framelog=verbose', '-f', 'null', '-'],
        input=np.ascontiguousarray(wave.T).astype('<f4').tobytes(), capture_output=True, check=True)
    found = re.findall(r'I:\s+(-?[\d.]+) LUFS', completed.stderr.decode())
    if not found or np.sqrt(np.mean(wave.astype(np.float64) ** 2)) < 1e-9:
        raise ValueError('Cannot measure integrated loudness of silent waveform')
    return float(found[-1])


def waveform_counterfactual(mixture: np.ndarray, sr: int, intervention: dict,
                            stems: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Change actual stereo waveform; callers must rerun every analysis model.

    Time shift uses zero fill (no wrapped samples). Source removal is a sum of
    remaining estimated stems, never cached feature subtraction.
    """
    wave = np.asarray(mixture, np.float32)
    if wave.ndim != 2 or wave.shape[0] != 2 or not np.isfinite(wave).all():
        raise ValueError('Counterfactual requires finite stereo[2,N]')
    kind = intervention['kind']
    meta = dict(intervention)
    if kind == 'gain':
        factor = 10 ** (float(intervention.get('db', -6)) / 20)
        result = wave * factor
    elif kind == 'time_shift':
        shift = int(round(float(intervention['seconds']) * sr))
        result = np.zeros_like(wave)
        if abs(shift) >= wave.shape[-1]:
            raise ValueError('Shift removes the complete song')
        if shift > 0:
            result[:, shift:] = wave[:, :-shift]
        elif shift < 0:
            result[:, :shift] = wave[:, -shift:]
        else:
            result[:] = wave
        meta['boundary'] = 'zero-filled; no circular wrapping'
    elif kind in ('source_removal', 'source_reconstruction'):
        if stems is None or np.asarray(stems).shape != (4, *wave.shape):
            raise ValueError('Source removal needs four actual separated stems')
        if kind == 'source_removal':
            remove = SOURCE_NAMES.index(intervention['source'])
            selected = np.arange(4) != remove
        else:
            selected = np.ones(4, dtype=bool)
        result = np.asarray(stems)[selected].sum(axis=0).astype(np.float32)
        if intervention.get('loudness_match', True):
            # BS.1770 loudness measurement is used, rather than equal-RMS claims.
            before = integrated_loudness(wave, sr)
            after = integrated_loudness(result, sr)
            if not np.isfinite([before, after]).all():
                raise ValueError('Cannot loudness-match a silent source-removal mixture')
            result *= 10 ** ((before - after) / 20)
            meta.update(loudness_before_lufs=before, removal_before_match_lufs=after,
                        match_method='FFmpeg ebur128 BS.1770 integrated loudness; floating waveform may exceed0dBFS')
    else:
        raise ValueError('Unknown waveform intervention')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite counterfactual waveform')
    meta['recompute_required'] = ['MuQ', 'HTDemucs', 'BeatThis', 'AllInOne', 'physical_observations']
    return result.astype(np.float32), meta


def _stitch_muq_chunks(chunks: list[tuple[np.ndarray, np.ndarray]], native_hz: float = 25.,
                       target_end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Merge real MuQ crops on a native timestamp grid without filling gaps."""
    if not chunks:
        raise ValueError('MuQ export contains no chunks')
    start = min(float(timestamp[0]) for _, timestamp in chunks)
    end = max(float(timestamp[-1]) for _, timestamp in chunks)
    if target_end is not None:
        if end > float(target_end) + 1. / native_hz + 1e-6:
            raise ValueError('MuQ chunk exceeds exact waveform duration')
        end = max(end, float(target_end))
    grid = np.arange(start, end + 0.5 / native_hz, 1. / native_hz, dtype=np.float64)
    if grid[-1] < end - 1e-8:
        # Keep the final real encoder timestamp when the last crop starts at a
        # fractional sample offset; never extrapolate past the crop endpoint.
        grid = np.concatenate((grid, np.asarray([end], np.float64)))
    total = np.zeros((len(grid), 1024), np.float64)
    count = np.zeros(len(grid), np.float64)
    for value, timestamp in chunks:
        timestamp = np.asarray(timestamp, np.float64)
        value = np.asarray(value, np.float32)
        # A fractional final crop can place the global endpoint less than one
        # native hop after the encoder's last center. Carry that real endpoint
        # onto the grid with the last encoder value; never bridge a larger gap.
        right = timestamp[-1] + (1. / native_hz + 1e-6 if target_end is not None else 1e-8)
        inside = (grid >= timestamp[0] - 1e-8) & (grid <= right)
        if not inside.any():
            continue
        projected = np.stack([np.interp(grid[inside], timestamp, value[:, i])
                              for i in range(value.shape[1])], axis=1)
        total[inside] += projected
        count[inside] += 1.
    keep = count > 0
    if not keep.any():
        raise ValueError('MuQ crops do not overlap a native timestamp grid')
    return (total[keep] / count[keep, None]).astype(np.float16), grid[keep]


def extract_muq_stream(extractor, mono_24k: np.ndarray, batch_size: int = 4,
                       target_duration_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(mono_24k, np.float32)
    size = 8 * 24000
    if len(wave) < size:
        raise ValueError('MuQ recipe requires8real seconds; no padded short-song fallback')
    starts = list(range(0, len(wave) - size + 1, size))
    if starts[-1] != len(wave) - size:
        starts.append(len(wave) - size)
    chunks = []
    for offset in range(0, len(starts), max(1, int(batch_size))):
        crop_starts = starts[offset:offset + max(1, int(batch_size))]
        exports = extractor.extract_batch([wave[start:start + size] for start in crop_starts])
        for start, (value, native_time) in zip(crop_starts, exports):
            chunks.append((value, np.asarray(native_time, np.float64) + start / 24000.))
    duration = len(wave) / 24000. if target_duration_s is None else float(target_duration_s)
    z, t = _stitch_muq_chunks(chunks, native_hz=25., target_end=duration - 1. / 25.)
    if z.shape != (len(t), 1024) or not np.isfinite(z).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid native MuQ export')
    return z, t


def _resample_muq_input(waveform: np.ndarray, duration_s: float) -> np.ndarray:
    """Resample to 24 kHz and bind the sample count to decoded duration."""
    from scipy.signal import resample_poly
    target_samples = int(round(float(duration_s) * 24000.))
    value = resample_poly(np.asarray(waveform, np.float32), 80, 147).astype(np.float32)
    if len(value) < target_samples:
        raise ValueError('24 kHz resampling under-covers the decoded waveform')
    return value[:target_samples]


def load_waveform(path: str | Path, sample_rate: int = 44100) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, sr = sf.read(path, dtype='float32', always_2d=True)
    if wave.shape[1] == 1:
        wave = np.repeat(wave, 2, axis=1)
    if wave.shape[1] != 2 or len(wave) < sr * 8 or not np.isfinite(wave).all():
        raise ValueError('Need finite mono/stereo song at least8seconds; channel duplication carries no spatial evidence')
    if sr != sample_rate:
        factor = math.gcd(sr, sample_rate)
        wave = resample_poly(wave, sample_rate // factor, sr // factor, axis=0)
    return wave.T.astype(np.float32)


def prepare_dependency_manifest(root: str | Path = CAMPAIGN_ROOT) -> dict:
    """Bind model weights, external inference code, and compiled CPU attention."""
    from .structure import SOURCE_DIRECTORY, WEIGHT_DIRECTORY, WEIGHTS, DEFAULT_PYTHON
    files = {}
    def add(path, category):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing extraction dependency: {path}')
        files[str(path)] = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size, 'category': category}
    for name in ('config.json', 'model.safetensors', 'receipt.json', 'receipt.json.sha256.json'):
        add(MODEL_LINKS / 'muq' / name, 'MuQ')
    for name in STUDY_HASHES:
        add(STUDY_SOURCE / 'audio_study' / name, 'frozen_audio_study')
    add(BEAT_CHECKPOINT, 'BeatThis')
    add('/root/.cache/huggingface/hub/models--adefossez--HTDemucs/blobs/d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd', 'HTDemucs')
    for filename, expected in WEIGHTS.items():
        path = WEIGHT_DIRECTORY / filename
        add(path, 'AllInOne_weight')
        if files[str(path)]['sha256'] != expected:
            raise ValueError('AllInOne dependency weight changed')
    for path in (SOURCE_DIRECTORY / 'src' / 'allin1').rglob('*.py'):
        add(path, 'AllInOne_source')
    compatibility = DEFAULT_PYTHON.parent.parent / 'natten_compatibility_receipt.json'
    add(compatibility, 'NATTEN_build_receipt')
    receipt = json.loads(compatibility.read_text())
    for name in ('installed_shared_library', 'installed_python_compatibility_file', 'upstream_sdist'):
        item = receipt[name]
        add(item['path'], 'NATTEN_' + name)
        if files[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Compiled NATTEN dependency changed')
    for item in receipt['modified_source_files']:
        add(item['path'], 'NATTEN_build_source')
        if files[item['path']]['sha256'] != item['sha256']:
            raise ValueError('NATTEN build source changed')
    add(receipt['patch'], 'NATTEN_exact_patch')
    add(receipt['validation'], 'NATTEN_reference_validation')
    for name in ('audio.py', 'contracts.py', 'structure.py'):
        add(Path(__file__).with_name(name), 'dynamic_dance_audio_source')
    for package, directory in (
        ('muq', '/mnt/lightsem-data/venv-night-encoders/lib/python3.12/site-packages/muq'),
        ('beat_this', '/mnt/lightsem-data/venv-night-encoders/lib/python3.12/site-packages/beat_this'),
        ('demucs', '/mnt/lightsem-data/venv-torch/lib/python3.12/site-packages/demucs'),
    ):
        for path in Path(directory).rglob('*.py'):
            add(path, package + '_implementation')
    result = {'schema_version': 1, 'artifacts': list(files.values()), 'test_audio_access': False,
              'structure_interpreter': str(DEFAULT_PYTHON), 'feature_interpreter': '/root/lightsem/.venv/night-access/bin/python'}
    path = Path(root) / 'features' / 'dependencies.json'
    if path.exists():
        previous = read_receipt(path)
        if previous != result:
            raise ValueError('Dependencies changed since freeze; preserve receipt and freeze an explicit new phase')
        return previous
    write_receipt(path, result)
    return result


def prepare_manifest(root: str | Path = CAMPAIGN_ROOT) -> dict:
    root = Path(root)
    output = root / 'features' / 'input_manifest.json'
    source = read_receipt(AUDIO_MANIFEST)
    entries = []
    for entry in source['entries']:
        if entry['partition'] not in ('train', 'val'):
            raise ValueError('TEST is prohibited in dynamic dance extraction')
        if entry.get('excluded') or entry.get('audio_quarantine_reason') or not entry['audio_eligible']:
            raise ValueError('Quarantined recording entered the approved audio manifest')
        entries.append({**entry, 'timeline_path': str(root / 'features' / 'timelines' / (entry['key'] + '.npz'))})
    if len({e['canonical_id'] for e in entries if e['partition'] == 'train'}) != 109 or len({e['canonical_id'] for e in entries if e['partition'] == 'val'}) != 31:
        raise ValueError('Frozen FineDance split counts changed')
    train_ids = {e['canonical_id'] for e in entries if e['partition'] == 'train'}
    if train_ids & {e['canonical_id'] for e in entries if e['partition'] == 'val'}:
        raise ValueError('Song-group leakage')
    for key, path in SHOWCASE.items():
        entries.append({'key': key, 'partition': 'showcase', 'canonical_id': key, 'wav_path': path,
                        'wav_sha256': sha256(path), 'timeline_path': str(root / 'features' / 'timelines' / (key + '.npz'))})
    result = {'schema_version': 1, 'recipe': RECIPE, 'source_manifest': str(AUDIO_MANIFEST),
              'source_manifest_sha256': sha256(AUDIO_MANIFEST), 'entries': entries, 'test_access': False,
              'study_source': str(STUDY_SOURCE), 'study_file_hashes': STUDY_HASHES,
              'beat_checkpoint': str(BEAT_CHECKPOINT), 'beat_checkpoint_sha256': BEAT_SHA256}
    if output.exists():
        old = read_receipt(output)
        if old != result:
            raise ValueError('Input manifest differs from already frozen manifest')
        return old
    guard_disk(root)
    write_receipt(output, result)
    return result


def extract_base92(wave: np.ndarray, target_time: np.ndarray, entry: dict,
                   root: str | Path, analyzed_waveform_sha256: str,
                   intervention: dict | None) -> dict:
    """Recompute native flow conditioning on the exact analyzed waveform.

    The existing extractor's five offline beat-grid channels remain disclosed;
    this call supplies audio features, never a replacement motion oscillator.
    """
    from ...audio.online import extract_bus
    from ...model.light_data import AUDIO_SCALARS
    path = Path(root) / 'features' / 'base92' / (entry['key'] + '.npz')
    receipt_path = path.with_suffix('.json')
    if receipt_path.exists():
        receipt = read_receipt(receipt_path)
        if receipt['analyzed_waveform_sha256'] != analyzed_waveform_sha256 or receipt['intervention'] != intervention or sha256(path) != receipt['npz_sha256']:
            raise ValueError('Existing Base92 belongs to another analyzed waveform')
        return receipt
    if path.exists():
        raise FileExistsError('Preserve unreceipted Base92 artifact')
    bus = extract_bus(y=np.asarray(wave, np.float32).mean(axis=0), sr=44100, fps_out=40, causal=True)
    # Preserve the extractor's float32 mel values through interpolation. The
    # old float16 round-trip quantized low-level tracks before they reached the
    # native 30 Hz clock; v2 binds the unquantized values to this cache receipt.
    mel = bus.mel.astype(np.float32, copy=False)
    features = np.concatenate((mel.T, np.stack([bus.channels[name] for name in AUDIO_SCALARS], axis=1)), axis=1).astype(np.float32)
    if features.shape[1] != 92 or not np.isfinite(features).all():
        raise ValueError('Invalid recomputed Base92 conditioning')
    aligned = interpolate(features, bus.times, target_time)
    guard_disk(root, required_bytes=features.nbytes + aligned.nbytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, x=features, time=bus.times, x30=aligned, time30=target_time)
    temporary.replace(path)
    receipt = {'schema_version': 1, 'kind': 'Base92', 'path': str(path), 'npz_sha256': sha256(path),
               'analyzed_waveform_sha256': analyzed_waveform_sha256,
               'input_path': entry['wav_path'], 'input_sha256': entry['wav_sha256'],
               'intervention': intervention,
               'recipe': 'existing extract_bus causalTrue40Hz; mel float32; exact30Hz interpolation',
               'features': ['mel_' + str(i) for i in range(64)] + list(AUDIO_SCALARS),
               'flags': bus.flags,
               'timestamps': 'original analyzed waveform start0; explicit40Hz time and30Hz time30',
               'extractor_sha256': sha256(Path(__file__).parents[2] / 'audio' / 'online.py'),
               'source_sha256': sha256(__file__)}
    write_receipt(receipt_path, receipt)
    return receipt


class AudioAnalyzer:
    """Reusable frozen model handles; single curator owns its device."""
    def __init__(self, device: str = 'cuda', structure_device: str = 'cpu'):
        self.device, self.structure_device = device, structure_device
        self._muq = self._separator = self._beat = None
        self._separator_provenance = None

    def muq(self):
        if self._muq is None:
            self._muq = import_study('features').Extractor('muq', device=self.device, link_root=MODEL_LINKS)
        return self._muq

    def separate(self, wave: np.ndarray):
        module = import_study('teachers')
        if self._separator is None:
            self._separator, self._separator_provenance = module.load_pinned_htdemucs(device=self.device)
        return module.separate_excerpt(self._separator, wave, device=self.device)

    def beats(self, wave: np.ndarray):
        import importlib.metadata
        import torch
        from beat_this.inference import Audio2Frames
        from beat_this.model.postprocessor import Postprocessor
        if self._beat is None:
            if sha256(BEAT_CHECKPOINT) != BEAT_SHA256:
                raise ValueError('BeatThis checkpoint changed')
            self._beat = Audio2Frames(checkpoint_path=str(BEAT_CHECKPOINT), device=self.device, float16=False)
        with torch.inference_mode():
            beat, down = self._beat(wave.T, 44100)
            if not torch.isfinite(beat).all() or not torch.isfinite(down).all():
                raise ValueError('Nonfinite BeatThis predictions')
            beats, downbeats = Postprocessor(type='minimal')(beat, down)
        return np.asarray(beats, np.float64), np.asarray(downbeats, np.float64), {
            'checkpoint_sha256': BEAT_SHA256, 'version': importlib.metadata.version('beat_this'),
            'postprocessor': 'minimal', 'frame_rate_hz': 50, 'confidence': 'event estimates; not calibrated accuracy',
            'beat_logit_std': float(beat.float().std().cpu()), 'downbeat_logit_std': float(down.float().std().cpu())}

    def extract(self, entry: dict, root: str | Path = CAMPAIGN_ROOT, intervention: dict | None = None) -> dict:
        from scipy.signal import resample_poly
        from .structure import extract_from_stems
        root = Path(root)
        key = entry['key']
        if entry['partition'] not in ('train', 'val', 'showcase', 'counterfactual'):
            raise ValueError('No TEST analysis')
        if '/' in key or key in ('.', '..'):
            raise ValueError('Unsafe record key')
        destination = Path(entry['timeline_path'])
        receipt_path = root / 'features' / 'receipts' / (key + '.json')
        if receipt_path.exists():
            previous = read_receipt(receipt_path)
            if previous['input_sha256'] != sha256(entry['wav_path']) or previous.get('intervention') != intervention:
                raise ValueError('Completed record input/intervention changed')
            MusicTimeline.load(destination)
            return previous
        if destination.exists():
            raise FileExistsError('Unreceipted timeline exists; preserve for independent audit')
        if sha256(entry['wav_path']) != entry['wav_sha256']:
            raise ValueError('Source waveform changed')
        guard_disk(root, required_bytes=256 * 1024 ** 2)
        begun = time.monotonic()
        timings = {}
        wave = load_waveform(entry['wav_path'])
        counterfactual_meta = None
        if intervention:
            original_stems = self.separate(wave) if intervention['kind'] in ('source_removal', 'source_reconstruction') else None
            wave, counterfactual_meta = waveform_counterfactual(wave, 44100, intervention, original_stems)
            del original_stems
        duration = wave.shape[-1] / 44100
        target = np.arange(math.ceil(duration * 30), dtype=np.float64) / 30
        import hashlib
        wave_id = hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
        base92 = None
        if entry['partition'] in ('showcase', 'counterfactual'):
            start = time.monotonic()
            base92 = extract_base92(wave, target, entry, root, wave_id, counterfactual_meta)
            timings['base92_s'] = time.monotonic() - start
        elif entry.get('base92_receipt'):
            base92 = dict(entry['base92_receipt'])
            if base92.get('npz_sha256') != sha256(base92['path']):
                raise ValueError('Precomputed Base92 receipt does not match its artifact')
        elif entry.get('base92_path'):
            if sha256(entry['base92_path']) != entry['base92_sha256']:
                raise ValueError('Frozen TRAIN/VAL Base92 identity changed')
            base92 = {'path': entry['base92_path'], 'npz_sha256': entry['base92_sha256'],
                      'original_waveform_sha256': entry['wav_sha256'],
                      'provenance': entry.get('base92_provenance', 'authenticated FineDance audio manifest; no waveform intervention')}
        start = time.monotonic()
        muq, muq_time = extract_muq_stream(self.muq(), _resample_muq_input(wave.mean(0), duration),
                                           target_duration_s=duration)
        timings['muq_s'] = time.monotonic() - start
        start = time.monotonic()
        stems = self.separate(wave)
        timings['separator_s'] = time.monotonic() - start
        start = time.monotonic()
        sources = np.stack([physical_observations(stem, 44100, target) for stem in stems], axis=1)
        mixture = mixture_observations(wave, 44100, target)
        timings['physical_s'] = time.monotonic() - start
        start = time.monotonic()
        beats, downbeats, beat_meta = self.beats(wave)
        timings['beat_this_s'] = time.monotonic() - start
        start = time.monotonic()
        structure_output = root / 'features' / 'structure' / key
        # Counterfactual structure analyzes the new waveform and newly separated stems.
        # Explicit provenance binds raw floating waveform bytes even without a saved wav.
        structure = extract_from_stems(entry['wav_path'], stems, structure_output, device=self.structure_device,
                                       stem_provenance={'separator': self._separator_provenance,
                                                        'analyzed_waveform_sha256': wave_id,
                                                        'duration_seconds': duration,
                                                        'counterfactual': counterfactual_meta})
        timings['all_in_one_s'] = time.monotonic() - start
        del stems
        sections = structure['sections']
        timeline = MusicTimeline(time=target, muq=muq, muq_time=muq_time, sources=sources, mixture=mixture,
                                 beats=beats, downbeats=downbeats, sections=sections,
                                 recurrence=section_recurrence(muq, muq_time, sections),
                                 metadata={'key': key, 'partition': entry['partition'], 'recipe': RECIPE,
                                           'duration_s': duration, 'input_path': entry['wav_path'], 'input_sha256': entry['wav_sha256'],
                                           'analyzed_waveform_sha256': wave_id, 'intervention': counterfactual_meta,
                                           'base92': base92,
                                           'muq': self.muq().metadata, 'separator': self._separator_provenance,
                                           'beat_this': beat_meta, 'structure': structure,
                                           'recurrence_definition': 'cosine similarity of raw MuQ section means; recurrence estimate, not motif truth',
                                           'timestamps': 'original waveform start t=0; physical30Hz; nativeMuQ25Hz; no fitted offsets'})
        timeline.validate()
        timeline.save(destination)
        intervals = preselect_intervals(timeline) if duration >= 180 else []
        receipt = {'schema_version': 1, 'key': key, 'input_sha256': entry['wav_sha256'], 'intervention': intervention,
                   'timeline_path': str(destination), 'timeline_sha256': sha256(destination),
                   'timeline_metadata_sha256': sha256(destination.with_suffix('.json')),
                   'duration_s': duration, 'elapsed_s': time.monotonic() - begun, 'timings': timings,
                   'base92': base92,
                   'device': self.device, 'structure_device': self.structure_device,
                   'test_access': False, 'source_sha256': sha256(__file__), 'contracts_sha256': sha256(Path(__file__).with_name('contracts.py')),
                   'audio_only_diagnostic_intervals': intervals,
                   'diagnostics': {'finite': True, 'muq_mean_temporal_variance': float(np.asarray(muq, np.float64).var(0).mean()),
                                   'muq_exact_adjacent_duplicate_fraction': float(np.all(muq[1:] == muq[:-1], axis=1).mean()),
                                   'source_rms_mean': sources[:, :, 0].mean(0).tolist(),
                                   'detected_beats': len(beats), 'detected_downbeats': len(downbeats), 'section_count': len(sections)}}
        write_receipt(receipt_path, receipt)
        return receipt


def extraction_preflight(keys: list[str], root: str | Path = CAMPAIGN_ROOT,
                         device: str = 'cuda', seconds: float = 8.) -> dict:
    """Bounded throughput capture only; no checkpoint/quality selection."""
    from scipy.signal import resample_poly
    root = Path(root)
    manifest = read_receipt(root / 'features' / 'input_manifest.json')
    lookup = {row['key']: row for row in manifest['entries']}
    receipt_path = root / 'features' / 'performance_preflight.json'
    if receipt_path.exists():
        raise FileExistsError('Preserve original performance preflight; do not rerun for a better timing')
    if seconds < 8 or seconds > 30:
        raise ValueError('Performance crop must be8to30seconds')
    analyzer = AudioAnalyzer(device=device)
    rows = []
    for key in keys:
        entry = lookup[key]
        if entry['partition'] not in ('train', 'val'):
            raise ValueError('Throughput preflight uses approved TRAIN/VAL recordings')
        if sha256(entry['wav_path']) != entry['wav_sha256']:
            raise ValueError('Preflight waveform drift')
        wave = load_waveform(entry['wav_path'])[:, :round(seconds * 44100)]
        begin = time.monotonic()
        crop_duration = wave.shape[-1] / 44100.
        z, t = extract_muq_stream(analyzer.muq(), _resample_muq_input(wave.mean(0), crop_duration),
                                  target_duration_s=crop_duration)
        muq_seconds = time.monotonic() - begin
        begin = time.monotonic()
        stems = analyzer.separate(wave)
        separation_seconds = time.monotonic() - begin
        begin = time.monotonic()
        beats, downbeats, beat_meta = analyzer.beats(wave)
        beat_seconds = time.monotonic() - begin
        rows.append({'key': key, 'input_sha256': entry['wav_sha256'], 'crop_start': 0.,
                     'crop_duration': wave.shape[-1] / 44100, 'muq_s': muq_seconds,
                     'separator_s': separation_seconds, 'beat_this_s': beat_seconds,
                     'finite': bool(np.isfinite(z).all() and np.isfinite(stems).all()),
                     'muq_shape': list(z.shape), 'stems_shape': list(stems.shape),
                     'model_loading_included_in_first_row': True})
        del stems
    result = {'schema_version': 1, 'purpose': 'performance-only; not scientific outcome selection',
              'rows': rows, 'device': device, 'source_sha256': sha256(__file__),
              'muq': analyzer.muq().metadata, 'separator': analyzer._separator_provenance,
              'beat_this': beat_meta, 'structure': 'separately measured real official model CPU forward'}
    write_receipt(receipt_path, result)
    return result


def build_index(root: str | Path = CAMPAIGN_ROOT, require_complete: bool = True) -> dict:
    root = Path(root)
    manifest = read_receipt(root / 'features' / 'input_manifest.json')
    entries, missing = [], []
    for entry in manifest['entries']:
        path = root / 'features' / 'receipts' / (entry['key'] + '.json')
        if not path.exists():
            missing.append(entry['key'])
            continue
        receipt = read_receipt(path)
        timeline_path = Path(receipt['timeline_path'])
        if sha256(timeline_path) != receipt['timeline_sha256'] or sha256(timeline_path.with_suffix('.json')) != receipt['timeline_metadata_sha256']:
            raise ValueError('Timeline identity drift')
        entries.append({**entry, **{k: receipt[k] for k in ('timeline_path', 'timeline_sha256', 'timeline_metadata_sha256')}})
    if missing and require_complete:
        raise ValueError(f'Missing {len(missing)} analyses: {missing[:12]}')
    result = {'schema_version': 1, 'entries': entries, 'missing': missing, 'complete': not missing,
              'input_manifest_sha256': sha256(root / 'features' / 'input_manifest.json'),
              'model_feature_dimension': 1061, 'normalization': 'none; caller must fit TRAIN only', 'test_access': False}
    if require_complete:
        write_receipt(root / 'features' / 'index.json', result)
    return result
