"""Non-mutating pilot for MuQ batching/stitching and Base92 precision.

This script writes only its JSON report. It never touches the active feature
cache and does not choose a training recipe.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1')
REPORT = ROOT / 'features' / 'MUQ_BASE92_V2_OVERLAP_PILOT.json'
MODEL_LINKS = Path('/root/lightsem-audio-study/model_links')
STUDY_SOURCE = Path('/root/lightsem-audio-study')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stitch_first(chunks):
    values, times = [], []
    for value, timestamp in chunks:
        absolute = timestamp + (0.0 if not times else 0.0)
        keep = np.ones(len(absolute), bool) if not times else absolute > times[-1][-1] + 1e-8
        if keep.any():
            values.append(value[keep])
            times.append(absolute[keep])
    return np.concatenate(values), np.concatenate(times)


def stitch_blend(chunks, tolerance=1e-5):
    """Average same-time overlap rows, retaining native timestamps."""
    rows = []
    for value, timestamp in chunks:
        rows.extend((float(t), np.asarray(v, np.float32)) for t, v in zip(timestamp, value))
    rows.sort(key=lambda row: row[0])
    values, times = [], []
    i = 0
    while i < len(rows):
        group = [rows[i][1]]
        j = i + 1
        while j < len(rows) and abs(rows[j][0] - rows[i][0]) <= tolerance:
            group.append(rows[j][1])
            j += 1
        times.append(rows[i][0])
        values.append(np.mean(np.stack(group), axis=0))
        i = j
    return np.asarray(values, np.float32), np.asarray(times, np.float64)


def stitch_overlap_grid(chunks, hz=25.0):
    """Blend overlap on a fixed native grid; leave non-overlap frames unchanged."""
    start = min(float(t[0]) for _, t in chunks)
    end = max(float(t[-1]) for _, t in chunks)
    grid = np.arange(start, end + 0.5 / hz, 1.0 / hz, dtype=np.float64)
    values, counts = [], []
    for value, timestamp in chunks:
        inside = (grid >= timestamp[0] - 1e-8) & (grid <= timestamp[-1] + 1e-8)
        projected = np.zeros((len(grid), value.shape[1]), np.float32)
        projected[inside] = np.stack([np.interp(grid[inside], timestamp, value[:, i])
                                      for i in range(value.shape[1])], axis=1)
        values.append(projected)
        counts.append(inside.astype(np.float32))
    total = np.sum(np.stack(values), axis=0)
    denom = np.sum(np.stack(counts), axis=0)
    keep = denom > 0
    return (total[keep] / denom[keep, None]).astype(np.float32), grid[keep]


def boundary_jumps(values, timestamps):
    if len(values) < 2:
        return {'count': 0, 'p95': 0.0, 'max': 0.0}
    delta = np.linalg.norm(np.diff(values.astype(np.float64), axis=0), axis=1)
    return {'count': int(len(delta)), 'p95': float(np.quantile(delta, .95)),
            'max': float(delta.max())}


def main():
    import soundfile as sf
    from lightsem.audio.online import extract_bus
    sys.path.insert(0, str(STUDY_SOURCE))
    from audio_study.features import Extractor

    paths = [
        ROOT / 'raw/audio/--74o20SmOs.wav',
        ROOT / 'raw/audio/-G-6DNPGEfk_gain_normalized.wav',
    ]
    rows = []
    extractor = Extractor('muq', device='cpu', link_root=MODEL_LINKS)
    for path in paths:
        wave, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        mono = wave.mean(axis=1)
        wave24 = resample_poly(mono, 80, 147).astype(np.float32)
        size = 8 * 24000
        starts = list(range(0, len(wave24) - size + 1, size))
        if starts[-1] != len(wave24) - size:
            starts.append(len(wave24) - size)
        selected_starts = [starts[0], starts[-2], starts[-1]] if len(starts) > 2 else starts
        crops = [wave24[start:start + size] for start in selected_starts]
        t0 = time.monotonic()
        scalar = []
        for start, crop in zip(selected_starts, crops):
            value, timestamp = extractor.extract_batch([crop])[0]
            scalar.append((value, timestamp + start / 24000.0))
        scalar_s = time.monotonic() - t0
        t0 = time.monotonic()
        batched = []
        for start, (value, timestamp) in zip(selected_starts, extractor.extract_batch(crops)):
            batched.append((value, timestamp + start / 24000.0))
        batch_s = time.monotonic() - t0
        scalar_value = np.concatenate([x[0] for x in scalar])
        batch_value = np.concatenate([x[0] for x in batched])
        scalar_time = np.concatenate([x[1] for x in scalar])
        batch_time = np.concatenate([x[1] for x in batched])
        first_value, first_time = stitch_first(scalar)
        blend_value, blend_time = stitch_blend(scalar)
        grid_value, grid_time = stitch_overlap_grid(scalar)

        bus = extract_bus(y=mono.astype(np.float32), sr=sample_rate, fps_out=40, causal=True)
        mel = np.asarray(bus.mel, np.float32)
        float16_roundtrip = mel.astype(np.float16).astype(np.float32)
        quant = float16_roundtrip - mel
        rows.append({
            'path': str(path), 'wav_sha256': sha(path),
            'duration_s': float(len(mono) / sample_rate),
            'muq_chunks_tested': len(crops),
            'muq_batch_seconds': batch_s, 'muq_scalar_seconds': scalar_s,
            'muq_speedup': float(scalar_s / max(batch_s, 1e-9)),
            'muq_max_abs_batch_difference': float(np.max(np.abs(scalar_value - batch_value))),
            'muq_mean_abs_batch_difference': float(np.mean(np.abs(scalar_value - batch_value))),
            'muq_timestamps_identical': bool(np.array_equal(scalar_time, batch_time)),
            'muq_first_stitch_boundary': boundary_jumps(first_value, first_time),
            'muq_blend_stitch_boundary': boundary_jumps(blend_value, blend_time),
            'muq_overlap_grid_stitch_boundary': boundary_jumps(grid_value, grid_time),
            'base92_mel_float16_max_abs_error': float(np.max(np.abs(quant))),
            'base92_mel_float16_mean_abs_error': float(np.mean(np.abs(quant))),
            'base92_mel_float16_relative_l2': float(np.linalg.norm(quant) / max(np.linalg.norm(mel), 1e-12)),
            'base92_causal': True,
        })
    report = {'schema_version': 1, 'status': 'pilot_only', 'model': extractor.metadata,
              'recipe': {'muq_batch_candidate': True, 'muq_overlap_blend_candidate': True,
                         'base92_float32_candidate': True}, 'rows': rows,
              'active_cache_untouched': True}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
