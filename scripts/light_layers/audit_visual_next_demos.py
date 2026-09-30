#!/usr/bin/env python3
"""CPU audit of saved own-history inference and every delivered RGB frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.visual_next.delivery import global_palette_rgb
from lightsem.experiments.visual_next.oracle import GROUP_ORDER
from lightsem.experiments.visual_next.palettes import fixture_groups, group_rgb_mask, prototype_representatives
from lightsem.model.light_layers.training import _cue_indices, load_checkpoint
from lightsem.model.light_layers.visual import sha256, write_json


def analytical_fade(target, times, active):
    """Piecewise segment reconstruction, independent of the delivery fade loop."""
    expected = np.floor(np.clip(target, 0, 1)*255+.5).astype(np.uint8)
    if not active.any():
        return expected
    first = int(np.flatnonzero(active)[0])
    origin = target[max(0, first-1)].copy()
    destination = origin.copy()
    began = times[first]-.3
    changes = np.flatnonzero(np.r_[False, np.any(target[1:] != target[:-1], axis=1)] & active)
    for start, stop in zip(changes, np.r_[changes[1:], len(times)]):
        u = min(1., max(0., (times[start]-began)/.3))
        origin = origin+(destination-origin)*(u*u*(3-2*u))
        destination = target[start]
        began = times[start]
        u = np.clip((times[start:stop]-began)/.3, 0, 1)
        values = origin+(destination-origin)*(u*u*(3-2*u))[:, None]
        expected[start:stop] = np.floor(np.clip(values, 0, 1)*255+.5).astype(np.uint8)
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve prior independent audit')
    f = json.loads(args.freeze.read_text())
    for path, digest in f['files'].items():
        assert sha256(path) == digest, path
    with np.load(f['baseline_C']['path'], allow_pickle=False) as z:
        baseline = z['candidate']
    with np.load(f['companion']['path'], allow_pickle=False) as z:
        companion = z['companion']
    times = np.arange(11092)/40
    fixtures = fixture_groups(canonical_rig())
    rgb_mask = group_rgb_mask(fixtures)
    prototypes = f['transforms']['palette_prototypes']
    palette = np.floor(global_palette_rgb(prototypes)*255+.5).astype(np.uint8)[:, None, :]
    representatives = np.asarray([[m['rgb'] for m in prototype_representatives(p)] for p in prototypes])
    palette = np.concatenate([palette, np.floor(representatives*255+.5).astype(np.uint8)], axis=1)
    report_hashes, rows = {}, []
    torch.set_num_threads(2)
    for candidate in f['candidates']:
        root = Path(f['output']) / candidate['id']
        report = json.loads((root / 'REPORT.json').read_text())
        assert report['status'] == 'actual_candidate_inference_and_RGB_streams_ready'
        assert report['freeze_sha256'] == sha256(args.freeze) and report['candidate'] == candidate
        assert sha256(root / 'model_outputs.npz') == report['model_outputs_sha256']
        with np.load(root / 'model_outputs.npz', allow_pickle=False) as z:
            values = {key: z[key] for key in z.files}
        assert all(np.isfinite(x).all() for x in values.values())
        assert np.array_equal(values['time40_s'], times)
        assert np.array_equal(values['time10_s'], np.arange(2773)/10)
        assert values['context10'].shape == (2773, 76)
        assert values['accent_logit'].shape == values['accent_score'].shape == (11092,)
        assert values['color_score'].shape == (2773,)
        cues = values['color_cues']
        assert cues.dtype == bool and cues.shape == (2773,)
        assert np.array_equal(np.flatnonzero(cues), _cue_indices(values['color_score'][None])[0])
        for logit, score in [('accent_logit', 'accent_score'), ('chromatic_change_logit', 'color_score')]:
            reference = torch.sigmoid(torch.from_numpy(values[logit])).numpy()
            np.testing.assert_allclose(reference, values[score], rtol=2e-6, atol=1e-7)
        actions = np.unique(np.r_[0, np.flatnonzero(cues)]).astype(int)
        assert np.array_equal(actions, values['action_indices'])
        # Replay all2773 own-state decisions using the actual saved action table.
        lookup = dict(zip(actions.tolist(), values['action_argmax_choices']))
        state = 32
        for index in range(2773):
            assert values['previous_ids'][index] == state
            if index in lookup:
                state = int(lookup[index][state])
            assert values['generated_palette_ids'][index] == state
        assert int(values['final_ids']) == state
        model, metadata = load_checkpoint(candidate['checkpoint']['path'], device='cpu')
        assert (metadata['arm'], metadata['seed'], metadata['step']) == (candidate['arm'], 0, 6000)
        with torch.no_grad():
            contexts = torch.from_numpy(values['context10'][actions])[:, None].expand(-1, 33, -1)
            previous = torch.arange(33).expand(len(actions), -1)
            tables = model.palette_logits(contexts, previous).numpy()
        actual_previous = values['previous_ids'][actions]
        assert np.array_equal(actual_previous, values['action_previous_ids'])
        selected = tables[np.arange(len(actions)), actual_previous]
        np.testing.assert_allclose(selected, values['action_logits'], rtol=2e-5, atol=1e-4)
        # Backend roundoff may change a near-tied unused state. Record all-state
        # agreement, while actual selected actions must reproduce the rollout.
        assert np.array_equal(selected.argmax(-1), values['generated_palette_ids'][actions])
        all_state_agreement = float(np.mean(tables.argmax(-1) == values['action_argmax_choices']))
        assert report['learned_color_active'] == bool(cues.any())
        active = np.zeros(11092, bool) if not cues.any() else times >= np.flatnonzero(cues)[0]/10
        ids40 = np.repeat(values['generated_palette_ids'], 4)
        assert {v['layout'] for v in report['variants']} == set(candidate['layouts'])
        for variant in report['variants']:
            assert sha256(variant['streams']) == variant['sha256']
            with np.load(variant['streams'], allow_pickle=False) as z:
                stream = z['candidate']
                assert np.array_equal(z['time'], times)
            assert stream.shape == baseline.shape and stream.dtype == np.uint8
            assert np.array_equal(stream[:, ~rgb_mask], baseline[:, ~rgb_mask])
            assert np.array_equal(stream[~active][:, rgb_mask], companion[~active][:, rgb_mask])
            assignment = {'global': [0, 0, 0, 0], 'four_groups': [1, 2, 0, 3]}[variant['layout']]
            for fixture in fixtures:
                slots = fixture['rgb_slots_zero_based']
                target = companion[:, slots].astype(float)/255
                mode = assignment[GROUP_ORDER.index(fixture['o4'])]
                target[active] = palette[ids40[active], mode]/255
                assert np.array_equal(analytical_fade(target, times, active), stream[:, slots])
            rows.append({'candidate': candidate['id'], 'layout': variant['layout'], 'stream_sha256': variant['sha256'],
                         'cue_count': int(cues.sum()), 'palette_changes': variant['invariants']['subsequent_palette_changes'],
                         'learned_color_active': bool(cues.any()), 'all_frames_and479_protected_slots_verified': True,
                         'CPU_palette_all_state_argmax_agreement': all_state_agreement})
        report_hashes[str(root / 'REPORT.json')] = sha256(root / 'REPORT.json')
        print('LEARNED_DEMO_INDEPENDENTLY_VERIFIED', candidate['id'], int(cues.sum()), flush=True)
    assert len(report_hashes) == 10 and len(rows) == 11
    for path, digest in f['files'].items():
        assert sha256(path) == digest, path
    write_json(args.output, {'status': 'all10_inferences_and11_streams_independently_verified',
                            'freeze_sha256': sha256(args.freeze), 'report_hashes': report_hashes, 'variants': rows,
                            'GPU_used': False, 'videos_rendered': False,
                            'scope': 'All own-state recurrences, saved-head CPU recomputation and independent analytical RGB fades; not an aesthetic or scientific efficacy verdict.'})


if __name__ == '__main__':
    main()
