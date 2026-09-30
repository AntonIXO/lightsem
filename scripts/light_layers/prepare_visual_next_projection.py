#!/usr/bin/env python3
"""Authenticate completed feature units and bind a TRAIN-only PCA phase.

This prepares an execution manifest; it never fits a projection itself. Missing
families fail before any manifest is written, allowing the shared extractor to
continue in its separate process.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from lightsem.experiments.visual_next.data import common_history_support
from lightsem.experiments.visual_next.features import ARMS
from lightsem.model.light_layers.visual import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-freeze', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('fit', 'apply'))
    parser.add_argument('--max-cpu-seconds', type=int, default=7200)
    args = parser.parse_args()
    if args.manifest.exists():
        raise RuntimeError('Refuse to replace a frozen execution manifest')
    source = json.loads(args.shared_freeze.read_text())
    rows = source['records']
    if args.mode == 'fit':
        rows = [row for row in rows if row['partition'] == 'train']
    expected = 91 if args.mode == 'fit' else 122
    if len(rows) != expected or len({r['canonical_id'] for r in rows}) != expected:
        raise RuntimeError('Original phase-specific cohort required')
    shared = Path(source['output'])
    missing = [r['canonical_id'] for r in rows if not (shared / r['canonical_id'] / 'REPORT.json').is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} required families are not complete; first: {missing[0]}')
    files = {}

    def bind(path, expected_digest=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError('Bound source or artifact changed: ' + str(path))
        files[str(path)] = digest

    bind(args.shared_freeze)
    support = []
    for row in rows:
        directory = shared / row['canonical_id']
        report_path = directory / 'REPORT.json'
        report = json.loads(report_path.read_text())
        if any(report[key] != row[key] for key in ('canonical_id', 'recording_id', 'partition')):
            raise RuntimeError('Extracted identity mismatch')
        if report['status'] != 'extracted' or report['frames'] != 640:
            raise RuntimeError('Incomplete feature extraction')
        if report['source_sha256'] != row['source_video']['sha256']:
            raise RuntimeError('Video identity mismatch')
        if report['baseline_replay_max_abs_rich_error'] > source['cached_baseline_max_abs_error']:
            raise RuntimeError('Source-observer replay tolerance failed')
        bind(report_path)
        for name, digest in report['artifacts'].items():
            # Authenticate the completed unit, including controls that will be
            # reused by later stages rather than fitted into this projection.
            bind(directory / name, digest)
        old_path = Path(row['capture_receipt']['path']).parent / 'sam2.npz'
        bind(old_path, source['files'][str(old_path)])
        bind(row['capture_receipt']['path'], row['capture_receipt']['sha256'])
        with np.load(directory / 'teacher_raw.npz', allow_pickle=False) as teacher, np.load(old_path, allow_pickle=False) as old:
            for key, shape in (('R1', (640, 1200)), ('dino_pretrained', (640, 16, 384)), ('dino_random_seed17', (640, 16, 384))):
                x = teacher[key]
                if x.dtype != np.float16 or x.shape != shape or not np.isfinite(x).all():
                    raise RuntimeError('Teacher representation schema mismatch')
                if np.any(x[~teacher['frame_valid']] != 0):
                    raise RuntimeError('Invalid teacher rows are not explicit zero placeholders')
            for key in ('frame_valid', 'requested_times_s', 'source_times_s'):
                if not np.array_equal(teacher[key], old[key]):
                    raise RuntimeError('Source clock or validity changed')
            valid = common_history_support(teacher['frame_valid'])
            if not valid.any() or int(teacher['frame_valid'].sum()) != report['original_valid_frames']:
                raise RuntimeError('Empty or inconsistent common feature support')
            support.append({'canonical_id': row['canonical_id'], 'partition': row['partition'],
                            'common_valid_rows': int(valid.sum())})

    root = Path(__file__).resolve().parents[2]
    for relative in (
        'scripts/light_layers/prepare_visual_next_projection.py',
        'scripts/light_layers/project_visual_next.py', 'run_torch.sh',
        'lightsem/__init__.py', 'lightsem/model/__init__.py',
        'lightsem/model/light_layers/__init__.py', 'lightsem/model/light_layers/visual.py',
        'lightsem/model/light_layers/targets.py', 'lightsem/experiments/__init__.py',
        'lightsem/experiments/visual_next/__init__.py',
        'lightsem/experiments/visual_next/features.py', 'lightsem/experiments/visual_next/data.py',
        'lightsem/experiments/visual_next/projection.py',
        'experiments/2026-09-29-visual-next/records/PREREGISTRATION_V1.md',
        'experiments/2026-09-29-visual-next/records/execution/IMPLEMENTATION_DETAILS_V2.md',
    ):
        bind(root / relative)
    if args.mode == 'apply':
        fitted = json.loads((args.output / 'FIT_REPORT.json').read_text())
        if fitted['status'] != 'all7_TRAIN_projections_frozen' or fitted['DEV_read']:
            raise RuntimeError('TRAIN-only fitting receipt required before application')
        bind(args.output / 'FIT_REPORT.json')
        bind(args.output / 'FROZEN_TRAIN_TRANSFORMS.json')
        for arm in ARMS:
            metadata = json.loads((args.output / (arm + '_PCA.json')).read_text())
            bind(args.output / (arm + '_PCA.json'))
            bind(args.output / (arm + '_PCA.npz'), metadata['arrays_sha256'])
    elif args.output.exists():
        raise RuntimeError('TRAIN projection output already exists')

    freeze = {
        'mode': args.mode, 'output': str(args.output.resolve()), 'shared_root': str(shared),
        'records': rows, 'files': files, 'authenticated_support': support,
        'max_cpu_seconds': args.max_cpu_seconds,
        'software_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'threadpoolctl')},
        'GPU': False, 'TEST_access': False, 'DEV_fit': False,
        'scope': 'Fixed seven-arm TRAIN-only PCA; source agreement is not accuracy. Apply phase never refits.',
        'command': 'TORCH_ENV=/root/lightsem/.venv/torch-access OPENBLAS_NUM_THREADS=2 bash run_torch.sh -u -m scripts.light_layers.project_visual_next --freeze ' + str(args.manifest.resolve()),
    }
    with args.manifest.open('x') as handle:
        json.dump(freeze, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('PROJECTION_PHASE_BOUND', args.mode, len(rows), len(files), sha256(args.manifest), flush=True)


if __name__ == '__main__':
    main()
