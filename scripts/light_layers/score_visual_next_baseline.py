#!/usr/bin/env python3
"""Reuse authenticated baseline-C predictions on the fixed full64s proxy.

No inference, refitting or new checkpoint selection. Beat-only predictions were
not cached in the historical run and remain explicitly unmeasured here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from lightsem.experiments.visual_next.scoring import score_color_window
from lightsem.model.light_layers.evaluation import FrozenLabels
from lightsem.model.light_layers.visual import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen baseline input changed: ' + path)
    prepared = json.loads(Path(freeze['reference_preparation']).read_text())
    historic = json.loads(Path(freeze['prediction_receipt']).read_text())
    if historic['status'] != 'complete' or historic['checkpoint']['sha256'] != freeze['baseline_C_sha256']:
        raise RuntimeError('Historical predictions are not the accepted C checkpoint')
    old = {row['recording_id']: row for row in historic['records']}
    refs = prepared['DEV_references']
    if len(old) != 31 or set(old) != {row['recording_id'] for row in refs}:
        raise RuntimeError('Fixed historical DEV membership changed')
    for key, saved in (('donors', 'donor_map'), ('donor_tempos_bpm', 'donor_tempos_bpm')):
        if prepared[key] != {rid: historic[saved][rid] for rid in old}:
            raise RuntimeError('Historical donor control is incompatible with the new frozen inputs')
    root = Path(freeze['output'])
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = []
    with threadpool_limits(limits=2):
        for row in refs:
            rid = row['recording_id']
            with np.load(row['path'], allow_pickle=False) as z:
                labels = FrozenLabels(rid, row['canonical_id'], 'val', **{k: z[k].copy() for k in z.files}).validate()
            if labels.content_hash() != row['reference_hash'] or row['reference_hash'] != old[rid]['target_sha256']:
                raise RuntimeError('Old and new reference labels differ')
            curves = {}
            with np.load(old[rid]['artifact']['path'], allow_pickle=False) as z:
                for condition in ('model', 'donor'):
                    curve = z[condition + '__color'].copy()
                    if curve.shape != (640,) or not np.isfinite(curve).all() or np.any((curve < 0) | (curve > 1)):
                        raise RuntimeError('Invalid saved color probabilities')
                    for key in ('requested_times10_s', 'source_times10_s'):
                        if not np.array_equal(z[condition + '__' + key], getattr(labels, key)):
                            raise RuntimeError('Saved prediction/source clock changed')
                    curves[condition] = curve
                if not np.array_equal(z['target__color_mask10'], labels.color_mask):
                    raise RuntimeError('Saved target support differs')
                curves['constant'] = np.full(640, prepared['TRAIN_color_prevalence'])
                # Historical baseline took a float32 label-array mean. The new
                # preparation stores the exact Python ratio of the same counts.
                historical_constant = np.full(640, float(np.float32(prepared['TRAIN_color_prevalence'])))
                if not np.array_equal(z['prevalence__color'], historical_constant):
                    raise RuntimeError('Original TRAIN constant differs')
            begin = time.monotonic()
            result = score_color_window(labels, curves)
            result.update(predictions=old[rid]['artifact'], scoring_seconds=time.monotonic() - begin,
                          checkpoint_role='accepted_baseline_C', beat_only='unmeasured_saved_prediction_absent')
            write_json(root / (row['canonical_id'] + '.json'), result)
            rows.append(result)
            print('C_RESCORED', len(rows), 31, round(result['scoring_seconds'], 3), flush=True)
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen baseline input changed: ' + path)
    eligible = [r for r in rows if r['source_coverage'] >= .8]
    event_rows = [r for r in eligible if r['scores']['model']['average_precision'] is not None]
    write_json(root / 'REPORT.json', {
        'status': 'baseline_C_saved_model_donor_predictions_rescored',
        'freeze_sha256': sha256(args.freeze), 'checkpoint': historic['checkpoint'],
        'rows': rows, 'source_eligible_families': len(eligible), 'eligible_event_families': len(event_rows),
        'macro_AP': {name: float(np.mean([r['scores'][name]['average_precision'] for r in event_rows]))
                     if event_rows else None for name in ('model', 'donor', 'constant')},
        'elapsed_s': time.monotonic() - started, 'new_inference': False, 'GPU_used': False,
        'beat_only': 'unmeasured_saved_prediction_absent', 'TEST_access': False,
        'constant_rounding': 'Same101/6927 TRAIN bins; historical float32 mean authenticated, current full64s scores use the frozen float64 ratio.',
        'scope': 'Accepted selected-seed C anchor on exposed historicalDEV, full64s automatic observer targets. Not a paired final6000 H3 cell or independent content accuracy. Original center8 results are unchanged.',
    })


if __name__ == '__main__':
    main()
