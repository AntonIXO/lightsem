#!/usr/bin/env python3
"""Recompute every fixed evaluation score from saved predictions and references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.visual_next.scoring import score_color_window
from lightsem.model.light_layers.evaluation import FrozenLabels
from lightsem.model.light_layers.training import _cue_indices
from lightsem.model.light_layers.visual import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous audit')
    f = json.loads(args.freeze.read_text())
    for path, digest in f['files'].items():
        assert sha256(path) == digest, path
    prepared = json.loads(Path(f['reference_preparation']['path']).read_text())
    references = {}
    for item in prepared['DEV_references']:
        assert sha256(item['path']) == item['sha256']
        with np.load(item['path'], allow_pickle=False) as z:
            labels = FrozenLabels(item['recording_id'], item['canonical_id'], 'val', **{k: z[k] for k in z.files}).validate()
        assert labels.content_hash() == item['reference_hash']
        references[item['canonical_id']] = labels
    assert len(references) == 31 and len(f['checkpoints']) == 27
    torch.set_num_threads(2)
    summaries, report_hashes = [], {}
    for checkpoint in f['checkpoints']:
        directory = Path(f['output']) / checkpoint['id']
        report_path = directory / 'REPORT.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'all31historicalDEV_conditions_scored'
        assert report['checkpoint'] == checkpoint and report['freeze_sha256'] == sha256(args.freeze)
        assert report['original_representation_fit_gate_pass'] is False
        assert report['reference_preparation'] == f['reference_preparation']
        rows = {row['canonical_id']: row for row in report['rows']}
        assert len(report['rows']) == 31 and set(rows) == set(references)
        for cid, labels in references.items():
            row = rows[cid]
            assert row['reference_hash'] == labels.content_hash()
            assert row['donor_recording_id'] == f['donors'][labels.recording_id]
            assert json.loads((directory / (cid + '.json')).read_text()) == row
            path = directory / (cid + '.npz')
            assert sha256(path) == row['predictions_sha256']
            curves = {'constant': np.full(640, f['TRAIN_color_prevalence'])}
            with np.load(path, allow_pickle=False) as values:
                assert all(np.isfinite(values[key]).all() for key in values.files)
                for condition in ('model', 'donor', 'beat_only'):
                    for key in ('requested_times10_s', 'source_times10_s'):
                        assert np.array_equal(values[condition+'_'+key], getattr(labels, key))
                    curve = values[condition+'_color']
                    assert curve.shape == (640,) and np.all((curve >= 0) & (curve <= 1))
                    logit = values[condition+'_color_logit']
                    np.testing.assert_allclose(torch.sigmoid(torch.from_numpy(logit)).numpy(), curve, rtol=2e-6, atol=1e-7)
                    cues = values[condition+'_generated_color_cues']
                    assert cues.dtype == bool and np.array_equal(np.flatnonzero(cues), _cue_indices(curve[None])[0])
                    ids = values[condition+'_generated_palette_ids']
                    previous = values[condition+'_previous_ids']
                    assert ids.shape == previous.shape == (640,)
                    assert np.all((ids >= 0) & (ids < 32)) and previous[0] == 32
                    assert np.array_equal(previous[1:], ids[:-1])
                    assert not np.any((ids[1:] != ids[:-1]) & ~cues[1:])
                    curves[condition] = curve.copy()
            recalculated = score_color_window(labels, curves)
            assert recalculated == {key: row[key] for key in recalculated}, (checkpoint['id'], cid)
        eligible = [r for r in rows.values() if r['source_coverage'] >= .8]
        events = [r for r in eligible if r['scores']['model']['average_precision'] is not None]
        summary = {'checkpoint': checkpoint, 'source_eligible_families': len(eligible), 'event_bearing_families': len(events),
                   'macro_AP': {condition: float(np.mean([r['scores'][condition]['average_precision'] for r in events]))
                                for condition in ('model', 'donor', 'beat_only', 'constant')}}
        summaries.append(summary)
        report_hashes[str(report_path)] = sha256(report_path)
        print('EVALUATION_INDEPENDENTLY_RECOMPUTED', checkpoint['id'], len(rows), flush=True)
    for path, digest in f['files'].items():
        assert sha256(path) == digest, path
    write_json(args.output, {'status': 'all27_checkpoints_and837_family_scores_independently_recomputed',
                            'freeze_sha256': sha256(args.freeze), 'report_hashes': report_hashes,
                            'checkpoint_summaries': summaries, 'family_scores': 27*31,
                            'conditions_per_family': ['model', 'donor', 'beat_only', 'constant', '20_shifts_when_qualified', '20_random'],
                            'GPU_used': False, 'formal_original_hypothesis_confirmed': False,
                            'scope': 'Recomputed frozen scoring instrument from saved finite predictions, clocks and independent pre-inference reference artifacts; automatic observer endpoint only.'})


if __name__ == '__main__':
    main()
