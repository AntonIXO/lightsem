#!/usr/bin/env python3
"""Apply already frozen TRAIN transforms in independent TRAIN/DEV phases.

There is no fitting path. This permits preparing TRAIN inputs while the shared
extractor finishes DEV. Completed projected arrays are never replaced.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from lightsem.experiments.visual_next.data import arm_features
from lightsem.experiments.visual_next.features import ARMS
from lightsem.experiments.visual_next.projection import project
from lightsem.model.light_layers.visual import sha256, write_arrays, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen projection input changed: ' + path)
    phase = freeze['phase']
    if phase not in ('train', 'val'):
        raise RuntimeError('Only original TRAIN or historical DEV application is supported')
    rows = freeze['records']
    expected = 91 if phase == 'train' else 31
    if len(rows) != expected or len({r['canonical_id'] for r in rows}) != expected or any(r['partition'] != phase for r in rows):
        raise RuntimeError('Original phase-specific cohort required')
    root = Path(freeze['output'])
    report_path = root / ('TRAIN_APPLY_REPORT.json' if phase == 'train' else 'DEV_APPLY_REPORT.json')
    if report_path.exists() or any((root / (r['canonical_id'] + '.npz')).exists() for r in rows):
        raise RuntimeError('Refuse to overwrite an application phase or completed target array')
    fitted = json.loads((root / 'FIT_REPORT.json').read_text())
    if fitted['status'] != 'all7_TRAIN_projections_frozen' or fitted['DEV_read'] or fitted['dev_fitted']:
        raise RuntimeError('TRAIN-only fitting receipt required')
    states = {}
    for arm in ARMS:
        metadata = json.loads((root / (arm + '_PCA.json')).read_text())
        if metadata != fitted['states'][arm] or metadata['arrays_sha256'] != sha256(root / (arm + '_PCA.npz')):
            raise RuntimeError('Immutable PCA state or metadata changed')
        if metadata['fit_families'] != freeze['fit_training_families'] or metadata['fit_partition'] != 'train':
            raise RuntimeError('PCA fit inventory drift')
        with np.load(root / (arm + '_PCA.npz'), allow_pickle=False) as z:
            states[arm] = {key: z[key].copy() for key in z.files}
    started = time.monotonic()
    artifacts = []
    with threadpool_limits(limits=2):
        for row in rows:
            arrays = {}
            support = None
            errors = {}
            for arm in ARMS:
                x, valid = arm_features(row, freeze['shared_root'], arm)
                if support is None:
                    support = valid
                elif not np.array_equal(support, valid):
                    raise RuntimeError('Auxiliary support differs between arms')
                target = np.zeros((len(x), 96), np.float32)
                target[valid] = project(x[valid], states[arm])
                arrays[arm] = target.astype(np.float16)
                if not np.isfinite(arrays[arm]).all():
                    raise RuntimeError('Invalid float16 target')
                errors[arm] = float(np.max(np.abs(arrays[arm].astype(np.float32) - target)))
            arrays['valid'] = support
            path = root / (row['canonical_id'] + '.npz')
            write_arrays(path, arrays)
            artifacts.append({'canonical_id': row['canonical_id'], 'partition': phase,
                              'path': str(path), 'sha256': sha256(path),
                              'common_valid_frames': int(support.sum()), 'max_float16_roundoff_by_arm': errors})
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3:
                raise RuntimeError('Application RAM cap')
            if time.monotonic() - started > freeze['max_cpu_seconds']:
                raise RuntimeError('Application CPU allowance')
            print('PCA_APPLIED', phase, row['canonical_id'], flush=True)
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen projection input changed: ' + path)
    report = {'status': 'frozen_TRAIN_transforms_applied_to_' + phase, 'phase': phase,
              'freeze_sha256': sha256(args.freeze), 'fit_report_sha256': sha256(root / 'FIT_REPORT.json'),
              'artifacts': artifacts, 'elapsed_s': time.monotonic() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'refitted': False, 'GPU_used': False, 'TEST_access': False}
    write_json(report_path, report)
    if phase == 'val':
        train = json.loads((root / 'TRAIN_APPLY_REPORT.json').read_text())
        if train['fit_report_sha256'] != report['fit_report_sha256'] or len(train['artifacts']) != 91:
            raise RuntimeError('Prior TRAIN application does not share the same transforms')
        for item in train['artifacts']:
            if sha256(item['path']) != item['sha256']:
                raise RuntimeError('Completed TRAIN targets changed')
        if (root / 'REPORT.json').exists():
            raise RuntimeError('Refuse to replace the consolidated application report')
        write_json(root / 'REPORT.json', {
            'status': 'train_projections_frozen_and_applied', 'states': fitted['states'],
            'artifacts': train['artifacts'] + artifacts, 'dev_fitted': False, 'test_access': False,
            'phase_reports': {str(path): sha256(path) for path in (root / 'TRAIN_APPLY_REPORT.json', report_path)},
            'scope': 'Same frozen TRAIN transforms applied independently to91TRAIN and31historicalDEV; no refit or replacement.',
        })


if __name__ == '__main__':
    main()
