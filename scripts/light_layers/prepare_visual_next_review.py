#!/usr/bin/env python3
"""Assemble all19 fixed review variants after actual model inference completes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from lightsem.model.light_layers.visual import sha256, write_json
from scripts.light_layers.prepare_visual_next_postfit import ROOT, RECORDS, RUNS, bind, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h3-seconds-remaining', type=float, required=True)
    parser.add_argument('--h4-seconds-remaining', type=float, required=True)
    parser.add_argument('--budget-ledger', type=Path, required=True)
    args = parser.parse_args()
    ledger = read(args.budget_ledger)
    for stage, remaining in [('H3', args.h3_seconds_remaining), ('H4', args.h4_seconds_remaining)]:
        if remaining <= 0 or ledger['remaining_seconds'][stage] != remaining:
            raise RuntimeError('Unverified remaining delivery budget')
    inventory_path = RECORDS / 'REVIEW_INVENTORY_V1.json'
    inventory = read(inventory_path)
    old_manifest_path = RUNS / 'shpongle_palette_previews_v1/MANIFEST.json'
    old = read(old_manifest_path)
    old_delivery_path = RUNS / 'shpongle_palette_previews_v1/DELIVERY.json'
    if read(old_delivery_path)['manifest_sha256'] != sha256(old_manifest_path):
        raise RuntimeError('Palette-preview delivery changed')
    oracle_report_path = RUNS / 'oracle_review_streams_v1/REPORT.json'
    oracle = read(oracle_report_path)
    oracle_audit_path = RECORDS / 'ORACLE_DEMO_PRIMARY_AUDIT_V1.json'
    if read(oracle_audit_path)['report_sha256'] != sha256(oracle_report_path):
        raise RuntimeError('Oracle streams differ from the independent audit')
    learned_audit_path = RECORDS / 'LEARNED_DEMO_PRIMARY_AUDIT_V1.json'
    learned_audit = read(learned_audit_path)
    if learned_audit['status'] != 'all10_inferences_and11_streams_independently_verified':
        raise RuntimeError('Complete learned-stream audit required')
    verified_reports = learned_audit['report_hashes']
    root = RUNS / 'shpongle_full_review_v1'
    if root.exists():
        raise RuntimeError('Preserve any previous review preparation')
    files = {}
    for path in (inventory_path, old_manifest_path, old_delivery_path, oracle_report_path,
                 oracle_audit_path, learned_audit_path, args.budget_ledger, Path(__file__)):
        bind(files, path)
    variants = []
    for item in inventory['variants']:
        row = copy.deepcopy(item)
        row.update(duration_s=277.3, start_s=0.)
        if row['kind'] == 'existing':
            previous = next(v for v in old['variants'] if v['letter'] == row['arm'])
            if previous['status'] not in ('verified_existing', 'rendered_and_verified'):
                raise RuntimeError('Requested existing video is not complete')
            for key in ('path', 'sha256', 'streams', 'streams_sha256', 'audit'):
                row[key] = previous[key]
            row['status'] = 'verified_existing'
            bind(files, row['path'], row['sha256'])
            bind(files, row['audit'])
        elif row['kind'] == 'learned':
            report_path = RUNS / 'shpongle_learned_streams_v1' / (row['arm'] + '_seed0') / 'REPORT.json'
            bind(files, report_path, verified_reports[str(report_path)])
            report = read(report_path)
            stream = next(v for v in report['variants'] if v['layout'] == row['layout'])
            row.update(streams=stream['streams'], streams_sha256=stream['sha256'],
                       delivery_invariants=stream['invariants'], budget_stage='H3')
            if not report['learned_color_active']:
                row['notice'] = 'This model emitted no color-change cues on Shpongle. The video therefore retains the original companion colors; there is no active learned color contribution.'
            row['checkpoint'] = report['candidate']['checkpoint']
        else:
            stream = next(v for v in oracle['variants'] if v['arm'] == row['arm'])
            row.update(streams=stream['streams'], streams_sha256=stream['sha256'],
                       delivery_invariants=stream['invariants'], budget_stage='H4',
                       source_reference=oracle['reference'])
        if row['kind'] != 'existing':
            row.update(status='planned', path=str(root / 'media' / row['id'] / (row['letter'] + '_full_1080p_av1.mp4')),
                       caption=row['letter'] + ' - ' + row['arm'] + ' - ' + ('reference oracle' if row['kind']=='oracle' else row['layout']) + ' - 300ms')
        bind(files, row['streams'], row['streams_sha256'])
        variants.append(row)
    if len(variants) != 19 or len({v['id'] for v in variants}) != 19:
        raise RuntimeError('Incomplete review inventory')
    # Preserve every experiment and filename while avoiding repeated rendering
    # of byte-identical, independently audited full-track outputs.
    duplicate_groups = {}
    for row in variants:
        if row['kind'] != 'learned':
            continue
        with np.load(row['streams'], allow_pickle=False) as values:
            digest = hashlib.sha256(values['candidate'].tobytes() + values['time'].tobytes()).hexdigest()
        duplicate_groups.setdefault(digest, []).append(row)
    exact_duplicates = []
    for digest, members in duplicate_groups.items():
        if len(members) < 2:
            continue
        first = members[0]
        first['caption'] = 'Shared color output - no learned color cues - 300ms'
        if any(row['delivery_invariants']['color_cue_count'] != 0 for row in members):
            first['caption'] = 'Shared identical color output - 300ms'
        with np.load(first['streams'], allow_pickle=False) as original:
            for row in members[1:]:
                with np.load(row['streams'], allow_pickle=False) as other:
                    assert np.array_equal(original['candidate'], other['candidate'])
                    assert np.array_equal(original['time'], other['time'])
                row['same_render_as'] = first['id']
                row['caption'] = first['caption']
        exact_duplicates.append({'frame_and_clock_sha256': digest, 'representative_id': first['id'],
                                 'variant_ids': [row['id'] for row in members],
                                 'rule': 'Exact full-track DMX and clock equality; all experiments remain available under separate filenames.'})
    bind(files, RECORDS / 'EXACT_OUTPUT_RENDER_REUSE_V1.md')
    page = ROOT / 'scripts/light_layers/visual_next_full_review.html'
    bind(files, page)
    for relative in ('scripts/light_layers/render_visual_next_candidates.py',
                     'scripts/rendering/render_light_layers_review.py',
                     'scripts/rendering/serve_light_review.py',
                     'scripts/motion/run_rhythm_companion.py',
                     'lightsem/conductor/rig.py', 'lightsem/dmx/fixtures.py', 'lightsem/dmx/__init__.py',
                     'lightsem/sim/renderer.py', 'lightsem/sim/shaders.py',
                     'lightsem/experiments/stage11/stage11_common.py'):
        bind(files, ROOT / relative)
    bind(files, old['source_audio'])
    root.mkdir()
    shutil.copyfile(page, root / 'index.html')
    bind(files, root / 'index.html')
    plan = dict(status='all_streams_verified_media_prepared', output_root=str(root),
                source_audio=old['source_audio'], expected_pcm_sha256=old['expected_pcm_sha256'],
                groups=inventory['groups'], variants=variants, exact_duplicate_groups=exact_duplicates,
                encoding='Native1080p30 AV1p7UHQ1100k/max1200k/AAC128k; <=50MB each',
                baseline_promoted=False, scope=inventory['scope'])
    plan_path = root / 'PLAN.json'
    write_json(plan_path, plan)
    bind(files, plan_path)
    freeze_path = RECORDS / 'FULL_REVIEW_RENDER_FREEZE_V1.json'
    if freeze_path.exists():
        raise RuntimeError('Preserve prior render freeze')
    write_json(freeze_path, dict(files=files, plan=str(plan_path), study_root=str(RUNS), max_study_bytes=5*2**30,
                                stage_seconds_remaining={'H3': args.h3_seconds_remaining, 'H4': args.h4_seconds_remaining},
                                budget_ledger=str(args.budget_ledger), scope='Complete19 inventory:3existing files and16new named files; exact duplicate outputs may share one encoded render via authenticated hard links.'))
    print('COMPLETE_REVIEW_PREPARED', sha256(freeze_path), len(files), flush=True)


if __name__ == '__main__':
    main()
