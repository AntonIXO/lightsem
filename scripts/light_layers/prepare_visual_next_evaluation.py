#!/usr/bin/env python3
"""Freeze observer references and audio controls before any student evaluation.

CPU-only preparation. Reads historical TRAIN/DEV source data, never a student
checkpoint, TEST data, or predictions. Reference labels remain automatic proxies.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.model.light_layers.evaluation import (
    audio_window, donor_audit, donor_map, eligible_intervals, freeze_labels,
    shift_plan, verify_donor_tempos,
)
from lightsem.model.light_layers.training import AUDIO_SCALARS, CENTER10, SourceStore
from lightsem.model.light_layers.visual import sha256, write_arrays, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Preparation input changed: ' + path)
    root = Path(freeze['output'])
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = json.loads(Path(freeze['training_manifest']['path']).read_text())
    transforms = json.loads(Path(freeze['baseline_fit_report']).read_text())['frozen_transforms']
    store = SourceStore(manifest, max_ram_bytes=512 * 1024**2)
    train = [row for row in manifest['records'] if row['partition'] == 'train']
    dev = [row for row in manifest['records'] if row['partition'] == 'val']
    if len(train) != 91 or len(dev) != 31 or len(manifest['records']) != 122:
        raise RuntimeError('Original 91 TRAIN / 31 historical DEV required')

    positives = bins = 0
    train_references = []
    for row in train:
        labels = freeze_labels(store.visual(row['recording_id']), transforms)
        valid = labels.color_mask[CENTER10]
        count = int(valid.sum())
        hits = int(labels.color[CENTER10][valid].sum())
        positives += hits
        bins += count
        train_references.append({
            'canonical_id': row['canonical_id'], 'recording_id': row['recording_id'],
            'reference_hash': labels.content_hash(),
            'eligible_center_bins': count, 'positive_center_bins': hits,
        })
    if not bins:
        raise RuntimeError('No eligible TRAIN center bins for constant baseline')

    tempos = {}
    references = []
    for row in dev:
        rid = row['recording_id']
        record = store.visual(rid)
        labels = freeze_labels(record, transforms)
        period = store.audio(rid)[:, 64 + AUDIO_SCALARS.index('beat_period_s')]
        if not np.isfinite(period).all() or np.any(period <= 0):
            raise RuntimeError('Invalid source beat period')
        tempos[rid] = 60 / float(np.median(period))
        # Validate exact audio coverage before assigning any donor or predicting.
        window = audio_window(store, record, transforms)
        intervals = eligible_intervals(labels, 'color', center=False)
        plan = shift_plan(labels.requested_times10_s, intervals, 10)
        arrays = {key: getattr(labels, key) for key in (
            'requested_times10_s', 'source_times10_s', 'accent10', 'color',
            'accent_mask10', 'color_mask',
        )}
        path = root / (row['canonical_id'] + '.npz')
        write_arrays(path, arrays)
        references.append({
            'canonical_id': row['canonical_id'], 'recording_id': rid,
            'reference_hash': labels.content_hash(),
            'path': str(path), 'sha256': sha256(path),
            'source_coverage': float(np.diff(intervals, axis=1).sum() / 64),
            'eligible_intervals_s': intervals.tolist(), 'shift_plan': plan,
            'source_positive_bins': int(labels.color.sum()),
            'audio_window_shape': list(window.shape),
        })
    mapping = donor_map(dev, tempos)
    verify_donor_tempos(store, [row['recording_id'] for row in dev], tempos)
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Preparation input changed: ' + path)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    write_json(root / 'REPORT.json', {
        'status': 'source_references_and_audio_controls_frozen_before_predictions',
        'freeze_sha256': sha256(args.freeze), 'training_manifest': freeze['training_manifest'],
        'transforms': transforms, 'TRAIN_color_prevalence': positives / bins,
        'prevalence_support': 'Original TRAIN central 8 seconds, observed eligible bins; pooled bin mean.',
        'TRAIN_positive_bins': positives, 'TRAIN_eligible_bins': bins,
        'TRAIN_references': train_references, 'DEV_references': references,
        'donor_tempos_bpm': tempos, 'donors': mapping, 'donor_audit': donor_audit(dev, mapping),
        'student_predictions_observed': False, 'TEST_access': False, 'CUDA_initialized': False,
        'elapsed_s': time.monotonic() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'scope': 'Source-only automatic observer references; no independent human accuracy or model quality result.',
    })
    print('EVALUATION_INPUTS_FROZEN', len(references), positives, bins, flush=True)


if __name__ == '__main__':
    main()
