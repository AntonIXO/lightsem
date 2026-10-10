"""Build strict/max Just Dance admission manifests from the sealed v1 audit.

This package builder reads the prior coach-level audit and Stage11 metadata. It
opens only max-TRAIN pose caches for the descriptive slow-target report; TEST
payloads are metadata-only and remain protected. No training or baseline
selection happens here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from lightsem.experiments.native_dynamics.admission import group_balanced_schedule


STAGE = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
CACHE = Path('/mnt/lightsem-data/cache')
OLD = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1')
FD = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/learning/cache/manifest.json')
PRIOR = Path('/mnt/lightsem-data/experiments/lightsem/native_admission_audit_20261007_v1')
COCO = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
REGIONS = {
    'lower_body': [7, 8, 9, 10, 11, 12], 'torso': [1, 2, 7, 8], 'head': [0],
    'left_arm': [1, 3, 5], 'right_arm': [2, 4, 6], 'whole_articulation': list(range(13)),
}
REGION_NAMES = tuple(REGIONS)
FPS = 30.0
HOLD_THRESHOLD = .10


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _audio_state(stage_id):
    path = STAGE/'data/audio_state'/f'{stage_id}.json'
    if not path.exists():
        return {}
    return read(path)


def _mask_pose(path: Path, coach_index: int):
    """Recreate fixed finite-transition masks without filling missing motion."""
    with np.load(path, allow_pickle=False) as z:
        k, c, box, times = z['K'].astype(float), z['C'].astype(float), z['box'].astype(float), z['t'].astype(float)
        meta = json.loads(str(z['meta']))
    if k.ndim == 3:
        k, c, box = k[:, None], c[:, None], box[:, None]
    k, c, box = k[:, coach_index], c[:, coach_index], box[:, coach_index]
    height = float(meta['height']); finite = np.isfinite(k).all(-1)
    confidence = np.isfinite(c) & (c >= .60)
    visible = np.isfinite(box).all(-1) & ((box[:, 3]-box[:, 1]) >= .30*height)
    valid = finite & confidence & visible[:, None]
    torso_ids = [5, 6, 11, 12]
    torso = valid[:, torso_ids].all(1) & (c[:, torso_ids] >= .70).all(1)
    shoulder = (k[:, 5] + k[:, 6])/2; hip = (k[:, 11] + k[:, 12])/2
    length = np.linalg.norm(shoulder-hip, axis=-1)
    eligible = torso & np.isfinite(length) & (length > .05*height)
    if eligible.sum() < 30:
        return np.zeros((0, 13, 2), np.float32), np.zeros((0, 13), bool), np.zeros(0), {'status': 'insufficient_torso'}
    scale = float(np.median(length[eligible]))
    clean = np.where(np.isfinite(k), k, 0.)
    steps = np.linalg.norm(np.diff(clean, axis=0), axis=-1)/scale
    pair = np.isfinite(k[:-1]).all(-1) & np.isfinite(k[1:]).all(-1) & np.isfinite(c[:-1]) & np.isfinite(c[1:])
    jumps = np.any((steps > 1.25) & pair, axis=1)
    valid[1:] &= ~jumps[:, None]; valid[:-1] &= ~jumps[:, None]
    valid &= valid[:, [11, 12]].all(1)[:, None]
    q = (clean-hip[:, None])/scale; q[..., 1] *= -1
    q, valid = q[:, COCO], valid[:, COCO]
    target = np.arange(int(np.ceil(len(times)/float(meta['fps_eff'])*FPS)))/FPS
    hi = np.searchsorted(times, target, side='left'); hi = np.clip(hi, 1, len(times)-1); lo = hi-1
    ok = ((target >= times[0]) & (target <= times[-1]) &
          ((times[hi]-times[lo]) <= .060))
    mask = valid[lo] & valid[hi] & ok[:, None]
    value = (1-((target-times[lo])/np.maximum(times[hi]-times[lo], 1e-12))[:, None, None])*q[lo]
    value += (((target-times[lo])/np.maximum(times[hi]-times[lo], 1e-12))[:, None, None])*q[hi]
    exact = np.isclose(target, times[hi], atol=1e-8, rtol=0); value[exact] = q[hi[exact]]; mask[exact] = valid[hi[exact]] & ok[exact, None]
    exact0 = np.isclose(target, times[0], atol=1e-8, rtol=0); value[exact0] = q[0]; mask[exact0] = valid[0]
    return np.where(mask[..., None], value, 0.).astype(np.float32), mask, target, {
        'status': 'ok', 'torso_scale_px': scale, 'raw_frames': len(times), 'raw_duration_s': float(times[-1]),
        'fixed_jump_edges': int(jumps.sum()), 'mask_contract': 'confidence>=.60; torso>=.70; box>=.30H; finite-to-finite jump<1.25; hip-valid; interpolation<=.060s; no forward fill'}


def _box_average(x, valid, width=6):
    pad_left, pad_right = width//2, (width-1)//2
    xp = np.pad(np.where(valid, x, 0.), ((pad_left, pad_right), (0, 0)))
    vp = np.pad(valid.astype(float), ((pad_left, pad_right), (0, 0)))
    total = np.stack([xp[i:i+len(x)] for i in range(width)], 0).sum(0)
    count = np.stack([vp[i:i+len(x)] for i in range(width)], 0).sum(0)
    return total/np.maximum(count, 1), count >= .8*width


def _slow_metrics(pose, mask):
    if len(pose) < 12:
        return {'status': 'too_short'}
    grid = np.arange(-4, 5); kernel = np.exp(-.5*(grid/1.5)**2); kernel /= kernel.sum()
    smooth = np.zeros_like(pose)
    reliable = np.zeros(mask.shape, bool)
    for j in range(pose.shape[1]):
        for d in range(2):
            smooth[:, j, d] = np.convolve(np.where(mask[:, j], pose[:, j, d], 0.), kernel, mode='same')
        reliable[:, j] = np.convolve(mask[:, j].astype(float), np.ones(9), mode='same') == 9
    velocity = np.diff(smooth, axis=0)*FPS
    edge = reliable[:-1] & reliable[1:]
    speed = np.sqrt((velocity**2).sum(-1)); speed = np.where(edge, speed, 0.)
    per_region = []
    for ids in REGIONS.values():
        per_region.append((speed[:, ids].mean(-1), speed[:, ids].max(-1), edge[:, ids].all(-1)))
    group_speed = np.stack([v[0] for v in per_region], -1)
    group_peak = np.stack([v[1] for v in per_region], -1)
    speed_valid = np.stack([v[2] for v in per_region], -1)
    peak_profile, peak_valid = _box_average(np.pad(group_peak, ((0, 1), (0, 0))), np.pad(speed_valid, ((0, 1), (0, 0))), 6)
    low = (peak_profile < HOLD_THRESHOLD) & peak_valid
    results = {}
    for i, region in enumerate(REGION_NAMES):
        state = low[:, i]; valid = peak_valid[:, i]
        starts = np.flatnonzero(np.diff(np.r_[False, state, False].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, state, False].astype(np.int8)) == -1)
        complete, censored = [], []
        for a, b in zip(starts, ends):
            if a > 0 and b < len(valid) and valid[a-1] and valid[b]: complete.append((b-a)/FPS)
            else: censored.append((b-a)/FPS)
        results[region] = {'valid_frames': int(valid.sum()), 'low_frames': int((state & valid).sum()),
                           'quiet_fraction_of_valid_frames': float((state & valid).sum()/max(valid.sum(), 1)),
                           'quiet_seconds_observed': float((state & valid).sum()/FPS),
                           'complete_bouts': len(complete), 'complete_bouts_ge1s': int(sum(v >= 1 for v in complete)),
                           'complete_bouts_ge2s': int(sum(v >= 2 for v in complete)),
                           'censored_bouts': len(censored), 'censored_seconds': float(sum(censored))}
    return {'status': 'ok', 'frames': len(pose), 'regions': results}


def _entry(row, stage_by_id, identity_by_id, *, whole_record=False):
    rid = row['key']; ident = identity_by_id[rid]; st = stage_by_id[rid]
    state = _audio_state(rid)
    return {'domain': 'jd', 'canonical_id': row['canonical_id'], 'recording_id': rid,
            'coach_index': row['coach_index'], 'coach_id': f'{rid}_c{row["coach_index"]}',
            'partition': 'train', 'identity_status': row['identity_status'], 'is_old': row['is_old'],
            'title_normalized': st['title_normalized'], 'legacy_song_key': ident.get('legacy_song_key'),
            'pose_path': st['pose_path'], 'pose_sha256': st['pose_sha256'], 'source_path': st['source_path'],
            'source_sha256': st['source_sha256'], 'wav_path': st['wav_path'],
            'wav_sha256': state.get('artifacts', {}).get(st['wav_path']),
            'audio_path': st['audio_path'], 'audio_sha256': state.get('artifacts', {}).get(st['audio_path']),
            'native_timestamp_field': 't', 'mask_contract': row['fixed_gate'].get('mask_contract', 'derived from raw K/C/box/t; no forward fill'),
            'window_rule': {'frames': 480, 'fps': 30, 'minimum_window_coverage': .80, 'start_stride_native_frames': 15},
            'native_duration_s': row['video_frame_duration_s'], 'audio_tail_s': row['audio_tail_s'],
            'fixed_gate': {'coverage': row['fixed_gate'].get('coverage'), 'eligible_windows': row['fixed_gate'].get('eligible_windows'),
                           'torso_scale_px': row['fixed_gate'].get('torso_scale_px'), 'jump_edges': row['fixed_gate'].get('tracking_jump_edges'),
                           'timestamp_gap_count': row['fixed_gate'].get('timestamp_gap_count')},
            'whole_record_qualified_80pct': bool(row['fixed_gate'].get('coverage', 0) >= .80),
            'preserves_native_timestamps_and_masks': True}


def build(output: Path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    stage = read(STAGE/'data/manifest.json'); stage_by_id = {r['recording_id']: r for r in stage['recordings']}
    songs = read(CACHE/'justdance_songs.json'); legacy = {r['yt_id']: r for r in songs['rows']}
    canonical = read(STAGE/'data/canonical_map.json'); old = read(OLD/'data/MANIFEST.json')
    audit_rows = read(PRIOR/'COACH_AUDIT.json'); prior_report = read(PRIOR/'REPORT.json')
    prior_muq = read(PRIOR/'MUQ_CACHE_AUDIT.json'); fd = read(FD)
    old_val_ids = {r['key'] for r in old['entries'] if r.get('partition') != 'train'}
    old_val_cids = {stage_by_id[k]['canonical_id'] for k in old_val_ids if k in stage_by_id}
    stage_holdout_cids = {r['canonical_id'] for r in stage['recordings'] if r['partition'] in ('val', 'test')}
    protected_cids = old_val_cids | stage_holdout_cids
    components = {c['canonical_id']: c for c in canonical['components']}
    identity_by_id = {}
    identity_rows = []
    for rid, st in sorted(stage_by_id.items()):
        legacy_row = legacy.get(rid, {})
        comp = components.get(st['canonical_id'], {})
        state = _audio_state(rid)
        identity_by_id[rid] = {'legacy_song_key': legacy_row.get('song_key')}
        identity_rows.append({'recording_id': rid, 'canonical_id': st['canonical_id'], 'identity_status': st['identity_status'],
                              'identity_reasons': st.get('holdout_exclusion_reasons', []), 'partition': st['partition'],
                              'protected_partition': ('test' if st['canonical_id'] in {r['canonical_id'] for r in stage['recordings'] if r['partition']=='test'} else
                                                      'val' if st['canonical_id'] in protected_cids else None),
                              'is_old': st['is_old'], 'legacy_map_present': rid in legacy, 'legacy_song_key': legacy_row.get('song_key'),
                              'title_normalized': st['title_normalized'], 'duplicate_members': comp.get('recording_ids', []),
                              'duplicate_edges': [e for e in canonical['edges'] if e['a'] in comp.get('recording_ids', []) and e['b'] in comp.get('recording_ids', [])],
                              'cross_corpus_links': [], 'cross_corpus_link_policy': 'No FineDance↔JustDance identity inferred from style labels; namespaces remain disjoint.',
                              'coach_identity': [{'coach_id': c['coach_id'], 'coach_index': c['coach_index'], 'dancer': c['dancer'], 'coverage': c['coverage'], 'coach_exclusions': st.get('coach_exclusions', [])} for c in st.get('coaches', [])],
                              'source_hashes': {'video': st['source_sha256'], 'pose': st['pose_sha256'],
                                               'wav': state.get('artifacts', {}).get(st['wav_path']), 'audio_features': state.get('artifacts', {}).get(st['audio_path']),
                                               'fingerprint': state.get('artifacts', {}).get(st['fingerprint_path'])},
                              'payload_opened_this_run': False if st['canonical_id'] in stage_holdout_cids and st['partition'] == 'test' else None})
    write(output/'RECORDING_IDENTITY.json', {'schema_version': 1, 'status': 'identity_reconciled_metadata_only_for_TEST',
                                             'legacy_map_sha256': digest(CACHE/'justdance_songs.json'), 'stage_manifest_sha256': digest(STAGE/'data/manifest.json'),
                                             'canonical_map_sha256': digest(STAGE/'data/canonical_map.json'), 'rows': identity_rows})

    def usable(row):
        return row.get('fixed_gate', {}).get('eligible_windows', 0) > 0 and row.get('admission_partition') == 'train' and row.get('admission_status') == 'candidate'
    strict_rows = [r for r in audit_rows if usable(r) and r['fixed_gate'].get('coverage', 0) >= .80]
    max_rows = [r for r in audit_rows if usable(r)]
    strict_entries = [_entry(r, stage_by_id, identity_by_id, whole_record=True) for r in strict_rows]
    max_entries = [_entry(r, stage_by_id, identity_by_id) for r in max_rows]
    strict = {'schema_version': 1, 'status': 'strict_qualification_manifest_not_trained',
              'selection': 'all duplicate-safe TRAIN coach tracks with finite-to-finite jump fix, an eligible 16-second masked window, whole-record coverage >=80%, and all existing quality gates; group-balanced sampler required',
              'entries': strict_entries, 'groups': sorted({r['canonical_id'] for r in strict_entries}),
              'counts': {'tracks': len(strict_entries), 'recordings': len({r['recording_id'] for r in strict_entries}), 'groups': len({r['canonical_id'] for r in strict_entries}),
                         'attached_minutes': sum(r['native_duration_s'] for r in strict_entries)/60,
                         'one_best_coach_per_group_minutes': prior_report['admission']['whole_record_qualified_train_minutes']}}
    max_manifest = {'schema_version': 1, 'status': 'max_masked_window_train_manifest_unqualified',
                    'selection': 'all duplicate-safe TRAIN coach tracks with >=1 valid masked window; no forward fill; native timestamps/masks derived at read time; group-balanced sampler required',
                    'entries': max_entries, 'groups': sorted({r['canonical_id'] for r in max_entries}),
                    'counts': {'tracks': len(max_entries), 'recordings': len({r['recording_id'] for r in max_entries}), 'groups': len({r['canonical_id'] for r in max_entries}),
                               'attached_minutes': sum(r['native_duration_s'] for r in max_entries)/60,
                               'one_best_coach_per_group_minutes': prior_report['admission']['window_recoverable_train_minutes'],
                               'whole_record_qualified_tracks': sum(r['whole_record_qualified_80pct'] for r in max_entries),
                               'window_only_tracks': sum(not r['whole_record_qualified_80pct'] for r in max_entries)}}
    write(output/'strict_qualification_manifest.json', strict); write(output/'max_masked_window_train_manifest.json', max_manifest)

    test_cids = {r['canonical_id'] for r in stage['recordings'] if r['partition'] == 'test'}
    protected_entries = []
    for rid, st in sorted(stage_by_id.items()):
        if st['canonical_id'] not in protected_cids:
            continue
        protected_entries.append({'recording_id': rid, 'canonical_id': st['canonical_id'], 'stage_partition': st['partition'],
                                 'protected_partition': 'test' if st['canonical_id'] in test_cids else 'val', 'identity_status': st['identity_status'],
                                 'is_old': st['is_old'], 'source_path': st['source_path'], 'source_sha256': st['source_sha256'],
                                 'pose_path': st['pose_path'], 'pose_sha256': st['pose_sha256'], 'coaches': st.get('coaches', []),
                                 'coach_exclusions': st.get('coach_exclusions', []), 'payload_opened_this_run': False})
    protected = {'schema_version': 1, 'status': 'protected_val_test_metadata_only', 'test_payload_opened_this_run': False,
                 'protected_group_count': len(protected_cids), 'test_group_count': len(test_cids), 'old_val_group_count': len(old_val_cids),
                 'entries': protected_entries}
    write(output/'protected_val_test_manifest.json', protected)
    leakage = {'strict_train_groups': len(set(strict['groups'])), 'max_train_groups': len(set(max_manifest['groups'])),
               'protected_groups': len(protected_cids), 'strict_overlap_protected': sorted(set(strict['groups']) & protected_cids),
               'max_overlap_protected': sorted(set(max_manifest['groups']) & protected_cids),
               'strict_test_overlap': sorted(set(strict['groups']) & test_cids), 'max_test_overlap': sorted(set(max_manifest['groups']) & test_cids),
               'test_payload_opened_this_run': False, 'status': 'pass'}
    if leakage['strict_overlap_protected'] or leakage['max_overlap_protected'] or leakage['strict_test_overlap'] or leakage['max_test_overlap']:
        leakage['status'] = 'fail'
    write(output/'LEAKAGE_AUDIT.json', leakage)

    # Group-balanced sampling receipt; do not sample TEST and do not write poses.
    schedule = group_balanced_schedule(max_entries, steps=8, batch_size=min(16, len(max_manifest['groups'])), seed=7001)
    write(output/'GROUP_BALANCED_SAMPLING.json', {'schema_version': 1, 'policy': 'uniform canonical group, then uniform recording/coach; at most one coach track per group per batch; no averaging',
                                                   'group_count': len(max_manifest['groups']), 'batch_size': len(schedule[0]), 'steps_preview': [[{'group': x['canonical_id'], 'recording_id': x['recording_id'], 'coach_index': x['coach_index']} for x in batch] for batch in schedule],
                                                   'train_only': True})

    # Slow/quiet target audit opens max TRAIN pose caches only. TEST is never read.
    per_track = []; grouped = defaultdict(list)
    for row in max_rows:
        pose, mask, target, meta = _mask_pose(Path(stage_by_id[row['key']]['pose_path']), int(row['coach_index']))
        if meta['status'] != 'ok': continue
        stats = _slow_metrics(pose, mask); stats.update({'canonical_id': row['canonical_id'], 'recording_id': row['key'], 'coach_index': row['coach_index'], 'coverage': row['fixed_gate']['coverage'], 'whole_record_qualified_80pct': row['fixed_gate']['coverage'] >= .80})
        per_track.append(stats); grouped[row['canonical_id']].append(stats)
    macro = {}
    for region in REGION_NAMES:
        vals = [s['regions'][region]['quiet_fraction_of_valid_frames'] for s in per_track if s['status']=='ok']
        gvals = [float(np.mean([s['regions'][region]['quiet_fraction_of_valid_frames'] for s in rs])) for rs in grouped.values() if rs]
        macro[region] = {'track_weighted_quiet_fraction': float(np.mean(vals)) if vals else None, 'canonical_group_macro_quiet_fraction': float(np.mean(gvals)) if gvals else None,
                         'track_count': len(vals), 'group_count': len(gvals),
                         'valid_frames': int(sum(s['regions'][region]['valid_frames'] for s in per_track if s['status']=='ok')),
                         'quiet_frames': int(sum(s['regions'][region]['low_frames'] for s in per_track if s['status']=='ok')),
                         'quiet_seconds_observed': float(sum(s['regions'][region]['quiet_seconds_observed'] for s in per_track if s['status']=='ok')),
                         'complete_bouts_ge1s': int(sum(s['regions'][region]['complete_bouts_ge1s'] for s in per_track if s['status']=='ok')),
                         'complete_bouts_ge2s': int(sum(s['regions'][region]['complete_bouts_ge2s'] for s in per_track if s['status']=='ok')),
                         'censored_bouts': int(sum(s['regions'][region]['censored_bouts'] for s in per_track if s['status']=='ok')),
                         'censored_seconds': float(sum(s['regions'][region]['censored_seconds'] for s in per_track if s['status']=='ok'))}
    write(output/'SLOW_TARGET_REPORT.json', {'schema_version': 1, 'status': 'descriptive_train_only', 'test_payload_opened_this_run': False,
                                             'quiet_definition': {'peak_projected_joint_speed_threshold_torso_lengths_per_s': HOLD_THRESHOLD, 'frontend': '9-tap sigma~50ms observer; 6-frame peak average; masks require valid endpoints', 'interpretation': 'strict observed quiet proxy, not all slow gestures and not a semantic rest label'},
                                             'target_censoring': 'bout durations are labeled only when valid observations exist immediately before and after; other runs are censored',
                                             'tracks_audited': len(per_track), 'groups_audited': len(grouped), 'regional': macro, 'per_track': per_track,
                                             'group_weighting': 'macro group summary is separate from track-weighted summary; no group/coach averaging is used for pose values'})

    muq_rows = {r['recording_id'] for r in prior_muq['rows']}; strict_missing = sorted({r['recording_id'] for r in strict_entries if r['recording_id'] not in muq_rows}); max_missing = sorted({r['recording_id'] for r in max_entries if r['recording_id'] not in muq_rows})
    mean_stream_bytes = np.mean([sum(l.get('bytes', 0) for l in r['layers']) for r in prior_muq['rows']])
    missing_stage = [r for r in stage['recordings'] if r['recording_id'] not in muq_rows]
    write(output/'MUQ_ADMISSION_REPORT.json', {'schema_version': 1, 'model_receipt': prior_muq['model'], 'verified_cache_receipt': str(PRIOR/'MUQ_CACHE_AUDIT.json'),
                                                'verified_streams': len(muq_rows), 'stage_missing_streams': len(missing_stage),
                                                'stage_missing_by_partition': dict(Counter(r['partition'] for r in missing_stage)),
                                                'strict_missing_recordings': strict_missing, 'max_missing_recordings': max_missing,
                                                'max_missing_estimated_bytes': int(mean_stream_bytes*len(max_missing)),
                                                'aggregate_matrix_rejected_for_reuse': True,
                                                'missing_extraction': {'status': 'blocked_storage_floor', 'reason': 'Estimated new per-track native streams would reduce /mnt free space below the required 20 GiB floor.', 'next_command': 'bash run_torch.sh scripts/motion/extract_missing_muq_native_20261007.py --package /mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2 --output /mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2/features/muq_native', 'no_TEST': True}})

    all_coach_minutes = sum(r.get('video_frame_duration_s', 0.) for r in audit_rows)/60
    fixed_window_rows = [r for r in audit_rows if r.get('fixed_gate', {}).get('eligible_windows', 0) > 0]
    fixed_window_minutes = sum(r.get('video_frame_duration_s', 0.) for r in fixed_window_rows)/60
    fresh_audio_minutes = sum(x['source'].get('duration_s', 0.) for x in prior_report.get('audio_exclusions', [])
                              if x.get('fresh_policy') in ('usable_source_audio_but_pose_not_extracted', 'usable_after_explicit_gain_normalization'))/60
    write(output/'COVERAGE_REPORT.json', {'schema_version': 1, 'source_stage11': {'recordings': len(stage['recordings']), 'partitions': dict(Counter(r['partition'] for r in stage['recordings']))},
                                         'gate_table': [
                                             {'gate': 'Stage11_non_silent_recordings', 'recordings': len(stage['recordings']), 'minutes': sum(r['duration_s'] for r in stage['recordings'])/60},
                                             {'gate': 'coach_tracks_in_saved_audit', 'tracks': len(audit_rows), 'minutes': all_coach_minutes},
                                             {'gate': 'finite_transition_masked_16s_window', 'tracks': len(fixed_window_rows), 'minutes': fixed_window_minutes},
                                             {'gate': 'max_duplicate_safe_TRAIN', 'tracks': len(max_entries), 'groups': len(max_manifest['groups']), 'minutes_attached': max_manifest['counts']['attached_minutes'], 'minutes_one_best_group': max_manifest['counts']['one_best_coach_per_group_minutes']},
                                             {'gate': 'strict_whole_record_80pct_TRAIN', 'tracks': len(strict_entries), 'groups': len(strict['groups']), 'minutes_attached': strict['counts']['attached_minutes'], 'minutes_one_best_group': strict['counts']['one_best_coach_per_group_minutes']},
                                             {'gate': 'fresh_audio_sources_without_pose_cache', 'recordings': sum(x.get('fresh_policy') in ('usable_source_audio_but_pose_not_extracted', 'usable_after_explicit_gain_normalization') for x in prior_report.get('audio_exclusions', [])), 'minutes': fresh_audio_minutes},
                                         ],
                                         'legacy_saved_audit': prior_report['coach_audit']['whole_record_vs_window']['saved_native_audit_comparison'],
                                         'strict': strict['counts'], 'max_masked_window': max_manifest['counts'],
                                         'protected': {'groups': len(protected_cids), 'records': len(protected_entries), 'test_payload_opened_this_run': False},
                                         'gate_order': ['Stage11 source/identity inventory', 'finite-to-finite pose gates', 'masked 16-second window', 'whole-record 80% strict gate', 'duplicate-safe canonical selection', 'protected-partition leakage check']})
    # Preserve visual evidence by copying the already authenticated v1 frames.
    if (PRIOR/'evidence').exists(): shutil.copytree(PRIOR/'evidence', output/'evidence', dirs_exist_ok=True)
    commands = [
        'cat AGENTS.md', 'cat /root/.codex/skills/native-dance-light-pipeline/SKILL.md',
        'bash run.sh scripts/motion/audit_justdance_admission_20261007.py --output /mnt/lightsem-data/experiments/lightsem/native_admission_audit_20261007_v1',
        'PYTHONPATH=/root/lightsem uv run --with pytest --with numpy --with scipy --with soundfile --with av pytest -q tests/test_stage11_data.py tests/test_stage11_coach_loop.py tests/test_justdance_audio_probe.py tests/test_justdance_admission.py tests/test_native_full_data.py tests/test_native_dynamics_data.py',
        'NEXT (storage-gated, not run): bash run_torch.sh scripts/motion/extract_missing_muq_native_20261007.py --package /mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2 --output /mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2/features/muq_native',
    ]
    (output/'COMMANDS.log').write_text('\n'.join(commands)+'\n')
    manifest = {'status': 'admission_package_complete_not_trained', 'strict_manifest': str(output/'strict_qualification_manifest.json'), 'max_manifest': str(output/'max_masked_window_train_manifest.json'), 'protected_manifest': str(output/'protected_val_test_manifest.json'), 'test_payload_opened_this_run': False, 'source_v1_report_sha256': digest(PRIOR/'REPORT.json')}
    write(output/'PACKAGE_STATUS.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args(); print(json.dumps(build(a.output), indent=2))
