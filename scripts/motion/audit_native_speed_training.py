"""Descriptive TRAIN-only inventory/slow-target audit; no fitting or model selection."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from lightsem.experiments.native_dynamics.kinematics import (
    observed_dynamics, HOLD_THRESHOLD, HOLD_TEMPERATURE, TORSO_METRES,
)
from lightsem.experiments.native_dynamics.data import SMPL

REGIONS = ('lower_body', 'torso', 'head', 'left_arm', 'right_arm', 'whole_articulation')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root, output):
    root, output = Path(root), Path(output)
    if shutil.disk_usage(root).free < 20 * 2 ** 30 + 2 * 2 ** 20:
        raise ValueError('20GiB reserve reached')
    torch.set_num_threads(1)
    cache = Path('/mnt/lightsem-data/cache')
    jd_path = root / 'data/MANIFEST.json'
    fd_path = root.parent / 'dynamic_dance_20261005_v1/learning/cache/manifest.json'
    sup_path = root / 'data/SUPERVISION.json'
    jd, fd, supervision = read(jd_path), read(fd_path), read(sup_path)
    inputs = {str(path): sha(path) for path in (jd_path, fd_path, sup_path)}
    targets = {}
    for domain in ('fd', 'jd'):
        records = [r for r in supervision['entries'] if r['domain'] == domain and r['partition'] == 'train']
        values, masks = [], []
        for record in records:
            if sha(record['path']) != record['sha256']:
                raise ValueError('Supervision changed')
            inputs[record['path']] = record['sha256']
            with np.load(record['path'], allow_pickle=False) as source:
                values.append(source['plan'].reshape(-1, 6, 8))
                masks.append(source['plan_valid'])
        plan, valid = np.concatenate(values), np.concatenate(masks)
        targets[domain] = {'training_recordings': len(records), 'half_second_bins': len(plan), 'regions': {}}
        for i, region in enumerate(REGIONS):
            good = valid[:, i, 1]
            targets[domain]['regions'][region] = {
                'valid_half_second_bins_fraction': float(good.mean()),
                'mean_quiet_fraction_in_valid_bins': float(plan[good, i, 3].mean()),
                'fully_quiet_valid_bins_fraction': float((plan[good, i, 3] >= .99).mean()),
            }
    valid_count, quiet_count, frames = np.zeros(6, int), np.zeros(6, int), 0
    durations = {region: [] for region in REGIONS}
    for row in jd['entries']:
        if row['partition'] != 'train':
            continue
        if sha(row['pose_cache']) != row['pose_cache_sha256']:
            raise ValueError('Pose supervision changed')
        inputs[row['pose_cache']] = row['pose_cache_sha256']
        with np.load(row['pose_cache'], allow_pickle=False) as source:
            pose, mask = torch.tensor(source['pose'])[None], torch.tensor(source['mask'])[None]
        dyn = observed_dynamics(pose, mask)
        valid, quiet = dyn['low_valid'][0].numpy(), dyn['low'][0].numpy()
        valid_count += valid.sum(0)
        quiet_count += (quiet & valid).sum(0)
        frames += pose.shape[1]
        for i, region in enumerate(REGIONS):
            state = quiet[:, i] & valid[:, i]
            edges = np.diff(np.r_[False, state, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if start > 0 and end < len(valid) and valid[start - 1, i] and valid[end, i]:
                    durations[region].append((end - start) / 30)
    frame_targets = {
        region: {'valid_frames': int(valid_count[i]), 'quiet_frames': int(quiet_count[i]),
                 'valid_frame_fraction': float(valid_count[i] / frames),
                 'quiet_fraction_of_valid_frames': float(quiet_count[i] / max(valid_count[i], 1)),
                 'quiet_seconds_observed': float(quiet_count[i] / 30),
                 'complete_bouts': len(durations[region]),
                 'complete_bouts_ge1s': int(sum(v >= 1 for v in durations[region])),
                 'complete_bouts_ge2s': int(sum(v >= 2 for v in durations[region]))}
        for i, region in enumerate(REGIONS)
    }
    cutoff = HOLD_THRESHOLD + HOLD_TEMPERATURE * math.log((1 - 1e-6) / 1e-6)
    loss_probe = []
    for speed in (1., 10.):
        old = torch.tensor(speed, requires_grad=True)
        p = torch.sigmoid((HOLD_THRESHOLD - old) / HOLD_TEMPERATURE)
        old_loss = F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.tensor(1.))
        old_loss.backward()
        fixed = torch.tensor(speed, requires_grad=True)
        fixed_loss = F.binary_cross_entropy_with_logits((HOLD_THRESHOLD - fixed) / HOLD_TEMPERATURE, torch.tensor(1.))
        fixed_loss.backward()
        loss_probe.append({'predicted_projected_peak_speed': speed, 'quiet_target': True,
                           'original_loss': float(old_loss.detach()), 'original_gradient': float(old.grad),
                           'corrected_loss': float(fixed_loss.detach()), 'corrected_gradient': float(fixed.grad)})
    showcase = []
    for song in ('shpongle', 'cure_for_me'):
        for method in ('D7_reference', 'C4'):
            path = root / 'repair_review_v1/bodies' / method / song / 'body.npz'
            inputs[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as source:
                world = torch.tensor(source['world'])[None]
            pose = (world[:, :, SMPL, :2] - world[:, :, [1, 2], :2].mean(-2, keepdim=True)) / TORSO_METRES
            dyn = observed_dynamics(pose, torch.ones(pose.shape[:3], dtype=torch.bool))
            valid = dyn['low_valid'][0, :, 5]
            speed = dyn['peak_profile'][0, :, 5][valid]
            showcase.append({'song': song, 'method': method,
                             'whole_quiet_fraction_by_observation_frontend': float(dyn['low'][0, :, 5][valid].float().mean()),
                             'fraction_above_original_quiet_gradient_cutoff': float((speed > cutoff).float().mean()),
                             'scope': 'Generated-motion measurement, not a ground-truth dance target for this song; observational smoothing is not a raw3D stability check.'})
    inventory = {}
    for name in ('justdance_wav', 'justdance_pose', 'justdance_motion', 'justdance_audio', 'justdance_structure', 'justdance_stem_feats', 'justdance_mert'):
        files = [p for p in (cache / name).iterdir() if p.is_file()]
        inventory[name] = {'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}
    def cohort(entries):
        return {partition: {'recordings': len(rows := [r for r in entries if r['partition'] == partition]),
                            'groups': len({r.get('canonical_id', r.get('group')) for r in rows}),
                            'hours': sum(r.get('duration_s', r.get('n_frames', 0) / r.get('fps', 30)) for r in rows) / 3600}
                for partition in ('train', 'val')}
    catalog_path = cache / 'justdance_manifest.json'
    catalog = read(catalog_path)
    result = {
        'scope': 'Descriptive audit. Only admitted TRAIN motion payloads and exposed showcase outputs were opened; development/test motion payloads were not opened.',
        'source_inputs': inputs, 'auditor_sha256': sha(Path(__file__)),
        'current_training': {'FineDance': cohort(fd['entries']), 'JustDance': cohort(jd['entries']),
                             'domain_order': ['fd', 'fd', 'jd'], 'FD_batch': 16, 'JD_batch': 8,
                             'sampler': 'Uniform canonical songgroup, then recording/start; no slow-state stratification.'},
        'JD_catalog': {'recordings': len(catalog), 'hours': sum(r['duration_s'] for r in catalog) / 3600,
                       'catalog_sha256': sha(catalog_path), 'cached_assets': inventory,
                       'previously_QA_qualified_groups': jd['qualified_groups_before_bounded_selection'],
                       'bounded_selected_groups': len(jd['entries']),
                       'extra_QA_qualified_groups_outside_selected_cohort': jd['qualified_groups_before_bounded_selection'] - len(jd['entries'])},
        'quiet_definition': {'peak_projected_joint_speed_threshold_torso_lengths_per_s': HOLD_THRESHOLD,
                             'measurement': 'Existing sigma50ms observer, then200ms mean of maximum regional joint speed; same masks as original training. This is a strict quiet proxy, not all very-slow gestures or semantic pause labels.'},
        'existing_coarse_targets': targets, 'JD_TRAIN_exact_frame_targets': frame_targets,
        'loss_bug': {'original_clipping': '[1e-6,1-1e-6] after sigmoid before BCE',
                     'gradient_cutoff_projected_speed': cutoff, 'probe': loss_probe,
                     'fix': 'Stable BCEWithLogits; observation masks/threshold/temperature/duration/physical terms unchanged.',
                     'limitation': 'Only this quiet-loss term loses gradient; other speed objectives still train. Causal contribution to observed nonstop movement is not isolated.'},
        'showcase_observer_measurements': showcase,
        'checkpoint_status': 'Current displayed models and prior recipes unchanged. Gradient correction is code-tested, not retrained or musically qualified.',
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x') as stream:
        stream.write(payload)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
    print(args.output)
