#!/usr/bin/env python3
"""Create full-track oracle review streams and verify the scored-context replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.experiments.visual_next.oracle_delivery import oracle_color_delivery
from lightsem.experiments.visual_next.oracle import GROUP_ORDER
from lightsem.model.light_layers.visual import sha256, write_arrays, write_json
from scripts.light_layers.run_visual_next_oracle import setup, state_schedule

SALT = 'visual-next-oracle-demo-v1:'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen oracle review input changed: ' + path)
    oracle_freeze = json.loads(Path(freeze['oracle_freeze']).read_text())
    report = json.loads(Path(freeze['oracle_report']).read_text())
    family = min((row['canonical_id'] for row in report['rows']),
                 key=lambda cid: hashlib.sha256((SALT + cid).encode()).hexdigest())
    if family != freeze['display_family']:
        raise RuntimeError('Display family differs from the fixed hash rule')
    base, _, ids, palette, groups, cues, initial = setup(oracle_freeze)
    output = Path(freeze['output'])
    output.mkdir(exist_ok=False)
    unchanged, _ = oracle_color_delivery(base, ids, cues, np.zeros((len(cues), 4), np.int64), palette)
    if not np.array_equal(unchanged, base):
        raise RuntimeError('Global-mode presentation does not reproduce baseline C')
    variants = []
    with np.load(Path(oracle_freeze['output']) / (family + '_SCORING.npz'), allow_pickle=False) as data:
        for arm in ('O0', 'O2', 'O4', 'permuted', 'donor'):
            choices = data[arm + '_choices']
            stream, invariants = oracle_color_delivery(base, ids, cues, choices, palette)
            states = state_schedule(choices, cues, initial, ids, palette)
            for index in range(110 * 40, 174 * 40):
                event = int(np.searchsorted(cues, index, side='right') - 1)
                colors = np.floor(np.clip(states[event].at([index / 40])[0], 0, 1) * 255 + .5).astype(np.uint8)
                for fixture in groups:
                    if not np.array_equal(stream[index, fixture['rgb_slots_zero_based']], colors[GROUP_ORDER.index(fixture['o4'])]):
                        raise RuntimeError('Full-track presentation differs from the actual oracle in its scored context')
            path = output / (arm + '.npz')
            write_arrays(path, {'candidate': stream, 'time': np.arange(len(base)) / 40})
            variants.append({'arm': arm, 'streams': str(path), 'sha256': sha256(path), 'invariants': invariants})
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen oracle review input changed: ' + path)
    write_json(output / 'REPORT.json', {
        'status': 'five_oracle_review_streams_verified', 'freeze_sha256': sha256(args.freeze),
        'display_family': family, 'selection_rule': SALT + 'canonical_id, minimum SHA256',
        'selection_scope': 'Post-study illustration chosen by identity hash, never by error/gain; all31 statistical results retained.',
        'reference': freeze['reference'], 'variants': variants,
        'scored_context_frames_verified_per_variant': 2560,
        'global_mode_full_C_replay_exact': True, 'GPU_used': False, 'videos_rendered': 0,
        'scope': 'Reference-oracle examples with frozen source information; not learned audio-spatial models or evidence of aesthetic quality.',
    })


if __name__ == '__main__':
    main()
