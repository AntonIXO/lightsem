#!/usr/bin/env python3
"""Bind the complete trained matrix to fixed DEV evaluation and Shpongle delivery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightsem.model.light_layers.visual import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / 'experiments/2026-09-29-visual-next/records/execution'
RUNS = ROOT / 'runs/visual_next_20260929'


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def bind(files, path, digest=None):
    ref = reference(path)
    if digest is not None and ref['sha256'] != digest:
        raise RuntimeError('Frozen input changed: ' + ref['path'])
    if ref['path'] in files and files[ref['path']] != ref['sha256']:
        raise RuntimeError('Conflicting input identities')
    files[ref['path']] = ref['sha256']
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-fit-audit', type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.primary_fit_audit)
    if audit['status'] != 'all24_full_fits_independently_verified':
        raise RuntimeError('Complete independent fit audit is required')
    queue_path = RECORDS / 'STUDENT_FIT_QUEUE_V1.json'
    queue = read(queue_path)
    if audit['queue_sha256'] != sha256(queue_path):
        raise RuntimeError('Audit belongs to a different training queue')
    if len(queue['cells']) != 24 or len(audit['cells']) != 24:
        raise RuntimeError('All21 study and3 demonstration cells are required')
    audited = {(r['arm'], r['seed']): r for r in audit['cells']}
    if len(audited) != 24:
        raise RuntimeError('Duplicate audited cell')
    checkpoints = []
    common = {}
    bind(common, args.primary_fit_audit)
    bind(common, queue_path)
    bind(common, __file__)
    bind(common, RECORDS / 'REVIEW_INVENTORY_V1.json')
    bind(common, RECORDS / 'STORAGE_AMENDMENT_V4.md')
    for cell in queue['cells']:
        directory = Path(cell['output'])
        report_ref = bind(common, directory / 'REPORT.json', audited[(cell['arm'], cell['seed'])]['REPORT_sha256'])
        report = read(report_ref['path'])
        checkpoint = bind(common, directory / 'final.pt', report['final_weights_sha256'])
        if checkpoint['sha256'] != audited[(cell['arm'], cell['seed'])]['checkpoint_sha256']:
            raise RuntimeError('Checkpoint differs from independently verified weights')
        if report['steps'] != 6000 or report['status'] != 'complete':
            raise RuntimeError('Incomplete fixed-step model')
        if report['main_targets_sha256'] != cell['main_targets_sha256']:
            raise RuntimeError('Training recipe identity changed')
        checkpoints.append(dict(checkpoint, id=directory.name, arm=cell['arm'], seed=cell['seed'],
                                role='final' if len(checkpoints) < 21 else 'demonstration',
                                main_targets_sha256=cell['main_targets_sha256']))

    preparation_path = RUNS / 'evaluation_inputs_v1/REPORT.json'
    prepared = read(preparation_path)
    preparation_freeze_path = RECORDS / 'EVALUATION_PREPARATION_FREEZE_V1.json'
    preparation_freeze = read(preparation_freeze_path)
    if prepared['freeze_sha256'] != sha256(preparation_freeze_path):
        raise RuntimeError('Reference preparation identity changed')
    files = common.copy()
    for path, digest in preparation_freeze['files'].items():
        bind(files, path, digest)
    bind(files, preparation_freeze_path)
    prep_ref = bind(files, preparation_path)
    qualification = bind(files, RUNS / 'retention_v1/REPORT.json')
    init_path = RUNS / 'initializations_v1/REPORT.json'
    initial = read(init_path)
    bind(files, init_path)
    for seed in (0, 1, 2):
        ref = initial['untrained_checkpoints'][str(seed)]
        bind(files, ref['path'], ref['sha256'])
        checkpoints.append(dict(ref, id=f'untrained_seed{seed}', role='untrained', arm='untrained', seed=seed))
    for relative in ('scripts/light_layers/evaluate_visual_next.py',
                     'lightsem/experiments/visual_next/scoring.py',
                     'lightsem/experiments/visual_next/shifts.py',
                     'lightsem/experiments/visual_next/instruments.py',
                     'lightsem/model/light_trunk.py',
                     'lightsem/experiments/stage11/stage11_common.py'):
        bind(files, ROOT / relative)
    evaluation = {key: prepared[key] for key in ('training_manifest', 'transforms', 'donors',
                                                'donor_tempos_bpm', 'TRAIN_color_prevalence')}
    evaluation.update(files=files, checkpoints=checkpoints, reference_preparation=prep_ref,
                      representation_qualification=qualification, output=str(RUNS / 'evaluations_v1'),
                      per_checkpoint_max_seconds=180, study_root=str(RUNS), max_study_bytes=5*2**30,
                      scope='27 fixed checkpoints; all31 historicalDEV; automatic full64s observer endpoint. No TEST, model selection or original hypothesis confirmation.')

    files = common.copy()
    palette_plan = read(RECORDS / 'PALETTE_PREVIEW_PLAN_V1.json')
    bind(files, qualification['path'], qualification['sha256'])
    for path, digest in palette_plan['input_hashes'].items():
        bind(files, path, digest)
    audio = bind(files, '/mnt/lightsem-data/experiments/lightsem/shpongle_layers_demo_20260922T161917Z/audio94.npz',
                 '224db0481c77677de7159e7c6f88cc42d8d861221c57e20e2b2072760dab6a9f')
    baseline = palette_plan['variants'][0]
    base = bind(files, baseline['streams'], baseline['streams_sha256'])
    companion_path = '/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z/streams.npz'
    companion = bind(files, companion_path, palette_plan['input_hashes'][companion_path])
    bind(files, preparation_path)
    for relative in ('scripts/light_layers/prepare_visual_next_demo.py',
                     'lightsem/experiments/visual_next/delivery.py',
                     'lightsem/experiments/visual_next/palettes.py',
                     'lightsem/experiments/visual_next/oracle.py',
                     'lightsem/model/light_layers/training.py',
                     'lightsem/model/light_layers/model.py',
                     'lightsem/model/light_layers/visual.py',
                     'lightsem/model/light_layers/transitions.py',
                     'lightsem/model/light_layers/overlay.py',
                     'lightsem/model/light_trunk.py',
                     'lightsem/conductor/rig.py', 'lightsem/dmx/fixtures.py',
                     'lightsem/experiments/stage11/stage11_common.py'):
        bind(files, ROOT / relative)
    candidates = []
    for checkpoint in checkpoints:
        if checkpoint['seed'] == 0 and checkpoint['role'] != 'untrained':
            candidates.append({key: checkpoint[key] for key in ('id', 'arm', 'seed', 'role', 'main_targets_sha256')})
            candidates[-1].update(checkpoint={key: checkpoint[key] for key in ('path', 'sha256')},
                                  layouts=['global', 'four_groups'] if checkpoint['arm'] == 'combined' else ['global'])
    if len(candidates) != 10 or sum(len(c['layouts']) for c in candidates) != 11:
        raise RuntimeError('All10 fixed seed0 models and11 learned displays required')
    inference = dict(files=files, candidates=candidates, audio94=audio, baseline_C=base,
                     companion=companion, transforms=prepared['transforms'],
                     representation_qualification=qualification,
                     output=str(RUNS / 'shpongle_learned_streams_v1'),
                     per_candidate_max_seconds=180, study_root=str(RUNS), max_study_bytes=5*2**30,
                     scope='Fixed seed0; whole277.3s FP32 inference, own history, global and combined4group displays. Personal review; no model selection or baseline promotion.')
    for name, value in [('EVALUATION_FREEZE_V1.json', evaluation), ('LEARNED_DEMO_FREEZE_V1.json', inference)]:
        destination = RECORDS / name
        if destination.exists():
            raise RuntimeError('Refuse to replace a frozen execution manifest')
        write_json(destination, value)
        print('POSTFIT_BOUND', name, sha256(destination), len(value['files']), flush=True)


if __name__ == '__main__':
    main()
