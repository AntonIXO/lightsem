"""Authenticate exact per-track audio products before reuse in a new fit.

Original receipts and arrays remain in their original roots. Reuse is a new
reference, never a copied or rewritten receipt and never an aggregate matrix.
"""
from pathlib import Path

import numpy as np

from .full_fit_admission import sha, verify_muq
from ..dynamic_dance import audio
from ..dynamic_dance.contracts import MusicTimeline


def _within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Feature artifact outside declared cache root')
    return path


def verify_training_clock(time, target, domain):
    """Require the exact paired clock; FD may retain a measured audio tail."""
    time, target = np.asarray(time), np.asarray(target)
    if (time.ndim != 1 or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError('Invalid cached timeline clock')
    if domain == 'fd':
        if len(time) < len(target) or not np.array_equal(time[:len(target)], target):
            raise ValueError('Cached FineDance timeline and training clocks differ')
    elif not np.array_equal(time, target):
        raise ValueError('Cached timeline and training clocks differ')


def verify_cached_features(row, expected, cache_root, waveform_id):
    """Return original products only after waveform/model/clock/recipe checks."""
    root = Path(cache_root).resolve()
    key = row['feature_key']
    if '/' in key or key in ('.', '..'):
        raise ValueError('Unsafe feature key')
    receipt_path = root/'features/receipts'/f'{key}.json'
    receipt = audio.read_receipt(receipt_path)
    if (receipt['key'] != key or receipt['input_sha256'] != row['wav_sha256']
            or receipt.get('intervention') is not None):
        raise ValueError('Cached waveform/key/intervention mismatch')
    if (receipt['source_sha256'] != sha(audio.__file__) or
            receipt['contracts_sha256'] != sha(Path(audio.__file__).with_name('contracts.py'))):
        raise ValueError('Cached feature implementation mismatch')
    timeline_path = _within(receipt['timeline_path'], root)
    if (sha(timeline_path) != receipt['timeline_sha256'] or
            sha(timeline_path.with_suffix('.json')) != receipt['timeline_metadata_sha256']):
        raise ValueError('Cached timeline receipt mismatch')
    timeline = MusicTimeline.load(timeline_path)
    metadata = timeline.metadata
    if (metadata.get('key') != key or metadata.get('intervention') is not None
            or metadata.get('analyzed_waveform_sha256') != waveform_id
            or metadata.get('recipe') != audio.RECIPE):
        raise ValueError('Cached analyzed waveform or context recipe mismatch')
    audit = verify_muq(timeline, expected, row['wav_sha256'])
    target = np.arange(row['frames'], dtype=np.float64)/30
    verify_training_clock(timeline.time, target, row.get('domain'))
    base_path = _within(receipt['base92']['path'], root)
    base = audio.read_receipt(base_path.with_suffix('.json'))
    if (base != receipt['base92'] or base != metadata.get('base92') or
            sha(base_path) != base['npz_sha256'] or
            base['input_sha256'] != row['wav_sha256'] or
            base['analyzed_waveform_sha256'] != waveform_id or
            base.get('intervention') is not None or
            base['source_sha256'] != sha(audio.__file__) or
            base['extractor_sha256'] != sha(Path(audio.__file__).parents[2]/'audio/online.py') or
            base['recipe'] != 'existing extract_bus causalTrue40Hz; mel float32; exact30Hz interpolation'):
        raise ValueError('Cached Base92 binding or recipe mismatch')
    with np.load(base_path, allow_pickle=False) as arrays:
        if (not np.array_equal(arrays['time30'], target) or
                arrays['x30'].shape != (len(target), 92) or
                arrays['x30'].dtype != np.float32 or
                not np.isfinite(arrays['x30']).all()):
            raise ValueError('Cached Base92 values or clock mismatch')
    product = dict(feature_key=key, timeline_path=str(timeline_path),
        timeline_sha256=receipt['timeline_sha256'], base92_path=str(base_path),
        base92_sha256=base['npz_sha256'], seconds=receipt['elapsed_s'],
        feature_receipt_path=str(receipt_path), feature_receipt_sha256=sha(receipt_path),
        reused=True, cache_root=str(root))
    if row.get('domain') == 'fd':
        aligned = _within(root/'features/fd_aligned'/f"{row['key']}.npz", root)
        fd_manifest = root/'data/FD_MANIFEST.json'
        import json
        originals = {r['key']: r for r in json.loads(fd_manifest.read_text())['entries']}
        original = originals[row['key']]
        if (original['wav_sha256'] != row['wav_sha256'] or
                original['frames'] != row['frames'] or
                Path(original['audio_path']).resolve() != aligned or
                sha(aligned) != original['audio_sha256']):
            raise ValueError('Cached FineDance alignment mismatch')
        with np.load(aligned, allow_pickle=False) as arrays:
            if (arrays['x'].shape != (len(target),92) or
                    arrays['extra'].shape != (len(target),1061) or
                    not all(np.isfinite(arrays[k]).all() for k in ('x','extra'))):
                raise ValueError('Cached FineDance aligned values invalid')
        product['fd_aligned'] = str(aligned)
    binding = dict(key=key, waveform_sha256=waveform_id,
        wav_file_sha256=row['wav_sha256'], base92_receipt_sha256=sha(base_path.with_suffix('.json')),
        timeline_receipt_sha256=sha(receipt_path), muq_model=metadata['muq'],
        timestamps=metadata['timestamps'])
    return product, dict(key=key, **audit), binding
