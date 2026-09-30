#!/usr/bin/env python3
"""Bind the actual additional TRAIN supervision recipes before any fitting."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.model.light_layers.visual import sha256, write_json
from scripts.light_layers.train_visual_next import prepared_data as baseline_data, main_target_digest
from scripts.light_layers.train_visual_next_demonstrations import ARMS, prepared_data


def counts(target, section):
    return {'color_positive': int(target['color'][section].sum()),
            'color_eligible': int(target['color_mask'][section].sum()),
            'palette_eligible': int(target['palette_mask'][section].sum()),
            'auxiliary_valid_coordinates': int(target['rich_mask'][section].sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen target-preparation input changed: ' + path)
    root = Path(freeze['output'])
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    _, originals, original_transforms, baseline = baseline_data(freeze, 'R0')
    if main_target_digest(baseline) != freeze['original_main_targets_sha256']:
        raise RuntimeError('Original targets changed')
    ids = {r.recording_id for r in originals}
    original_samples = [(r.recording_id, 0) for r in originals[:4]]
    original_audio = baseline.batch(original_samples, 'cpu')['audio40'].numpy()
    summaries = {}
    mask_targets = None
    for arm in ARMS:
        _, records, transforms, data = prepared_data(freeze, arm)
        if {r.recording_id for r in records} != ids or transforms != original_transforms:
            raise RuntimeError('TRAIN inventory or fixed transforms changed')
        if not np.array_equal(data.batch(original_samples, 'cpu')['audio40'].numpy(), original_audio):
            raise RuntimeError('Audio changed in a visual supervision intervention')
        per_family = []
        for record in records:
            target = data.targets[record.recording_id]
            old = baseline.targets[record.recording_id]
            for key in ('accent10', 'accent_mask10'):
                if not np.array_equal(target[key], old[key]):
                    raise RuntimeError('Original accent targets changed')
            if np.any(target['color'][~target['color_mask']] != 0):
                raise RuntimeError('Color positives outside observed support')
            if arm == 'cut_only':
                for key in ('color_mask', 'palette', 'palette_mask', 'rich', 'rich_mask'):
                    if not np.array_equal(target[key], old[key]):
                        raise RuntimeError('Cut-only changed more than positive color labels: ' + key)
                if np.any(target['color'] > old['color']):
                    raise RuntimeError('Cut-only added positive color labels')
            if arm == 'combined':
                other = mask_targets[record.recording_id]
                for key in ('color_mask', 'palette', 'palette_mask'):
                    if not np.array_equal(target[key], other[key]):
                        raise RuntimeError('Combined recipe changed mask-only main support or vocabulary')
                if np.any(target['color'] > other['color']):
                    raise RuntimeError('Combined cut veto added positive color labels')
                with np.load(Path(freeze['projected_root']) / (record.canonical_id + '.npz'), allow_pickle=False) as z:
                    if not np.array_equal(target['rich'], z['R3'].astype(np.float32)):
                        raise RuntimeError('Combined auxiliary is not the fixed original R3 target')
                    if not np.array_equal(target['rich_mask'], np.broadcast_to(z['valid'][:, None], (640, 96))):
                        raise RuntimeError('Combined auxiliary support changed')
            per_family.append({
                'canonical_id': record.canonical_id, 'recording_id': record.recording_id,
                'original_full64': counts(old, slice(None)), 'new_full64': counts(target, slice(None)),
                'original_supervised_center8': counts(old, slice(280, 360)),
                'new_supervised_center8': counts(target, slice(280, 360)),
                'changed_color_bins': int(np.count_nonzero(target['color'] != old['color'])),
                'changed_palette_bins': int(np.count_nonzero(target['palette'] != old['palette'])),
                'mask_auxiliary_float16_roundoff': getattr(record, 'visual_next_aux_roundoff', None),
            })
        if arm == 'mask_only':
            mask_targets = {rid: {key: target[key].copy() for key in ('color', 'color_mask', 'palette', 'palette_mask')}
                            for rid, target in data.targets.items()}
        summaries[arm] = {
            'main_targets_sha256': main_target_digest(data), 'families': per_family,
            'accent_targets_unchanged': True, 'audio_unchanged': True,
            'full64_color_positive': sum(row['new_full64']['color_positive'] for row in per_family),
            'supervised_center8_color_positive': sum(row['new_supervised_center8']['color_positive'] for row in per_family),
            'supervised_center8_color_eligible': sum(row['new_supervised_center8']['color_eligible'] for row in per_family),
        }
        print('DEMONSTRATION_TARGETS_BOUND', arm, summaries[arm]['main_targets_sha256'], flush=True)
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen target-preparation input changed: ' + path)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU target preparation initialized CUDA')
    write_json(root / 'REPORT.json', {
        'status': 'three_additional_TRAIN_supervision_recipes_bound_before_fitting',
        'freeze_sha256': sha256(args.freeze), 'original_main_targets_sha256': main_target_digest(baseline),
        'arms': summaries, 'elapsed_s': time.monotonic() - started,
        'training_performed': False, 'GPU_used': False, 'DEV_read': False, 'TEST_access': False,
        'scope': 'Target counts and intervention invariants only; no false-event, accuracy or lighting-quality claim.',
    })


if __name__ == '__main__':
    main()
