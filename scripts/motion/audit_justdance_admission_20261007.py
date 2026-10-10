"""Audit Just Dance admission without training or changing frozen splits.

This audit joins the legacy cache, Stage11's duplicate-safe identities, the
native masked-window gate, source-audio probes, and the reusable MuQ cache.  It
writes a new, explicitly unqualified manifest under ``--output``.  It does not
open or modify model checkpoints and never treats an observed 2-D pose as a
metric 3-D label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf

from lightsem.experiments.native_dynamics.data import Quality, sha


CACHE = Path('/mnt/lightsem-data/cache')
STAGE = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
OLD_NATIVE = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1')
MUQ_NATIVE = Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/features/muq_native')
MUQ_MODEL = MUQ_NATIVE.parent / 'muq_model.json'
MUQ_WEIGHTS = MUQ_NATIVE.parent / 'muq_weight_audit.json'
SOURCE = Path('/mnt/lightsem-data/justdance')
COCO = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _finite_hash(path: Path, expected_bytes: int, expected_sha: str) -> dict:
    result = {'path': str(path), 'exists': path.exists()}
    if not path.exists():
        return result | {'bytes_ok': False, 'sha256_ok': False}
    result['bytes'] = path.stat().st_size
    result['bytes_ok'] = result['bytes'] == expected_bytes
    result['sha256'] = _sha(path)
    result['sha256_ok'] = result['sha256'] == expected_sha
    return result


def _source_audio_stats(path: Path) -> dict:
    """Decode one source audio stream to floats for an admission diagnostic."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(path), '-vn',
           '-ac', '1', '-ar', '44100', '-f', 'f32le', 'pipe:1']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        return {'decode_ok': False, 'error': proc.stderr.decode(errors='replace')[:240]}
    x = np.frombuffer(proc.stdout, dtype='<f4')
    if not len(x):
        return {'decode_ok': True, 'samples': 0, 'usable_signal': False}
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    peak = float(np.max(np.abs(x)))
    nonzero = float(np.mean(np.abs(x) > 1e-7))
    mean_db = 20 * math.log10(max(rms, 1e-12))
    peak_db = 20 * math.log10(max(peak, 1e-12))
    probe = subprocess.run(
        ['ffmpeg', '-hide_banner', '-nostats', '-v', 'info', '-i', str(path), '-vn',
         '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, text=True, check=False)
    measured = {}
    for line in (probe.stderr or '').splitlines():
        if 'mean_volume' in line or 'max_volume' in line:
            key, value = line.split(':', 1)
            measured['fresh_' + ('mean_db' if 'mean_volume' in key else 'peak_db')] = float(value.strip().split()[0])
    return {'decode_ok': True, 'samples': int(len(x)), 'duration_s': float(len(x) / 44100),
            'rms_db': mean_db, 'peak_db': peak_db, 'crest_db': peak_db-mean_db,
            'nonzero_fraction': nonzero, 'usable_signal': bool(nonzero > .01 and peak > 1e-4),
            'gain_to_minus24_db': -24.0-mean_db,
            'peak_after_minus24_db': peak_db + (-24.0-mean_db), **measured}


def _apply_jump_gate(k, c, box, meta, times=None, *, finite_transition_only: bool) -> dict:
    """Reproduce native thresholds, with an explicit missing-transition variant."""
    cfg = Quality()
    height = float(meta['height'])
    finite = np.isfinite(k).all(-1)
    confidence = np.isfinite(c) & (c >= cfg.confidence)
    visible = np.isfinite(box).all(-1) & ((box[:, 3]-box[:, 1]) >= cfg.minimum_box_height_fraction*height)
    valid = finite & confidence & visible[:, None]
    torso_ids = [5, 6, 11, 12]
    torso = valid[:, torso_ids].all(1) & (c[:, torso_ids] >= cfg.torso_confidence).all(1)
    shoulder = (k[:, 5] + k[:, 6])/2
    hip = (k[:, 11] + k[:, 12])/2
    length = np.linalg.norm(shoulder-hip, axis=-1)
    eligible = torso & np.isfinite(length) & (length > .05*height)
    if eligible.sum() < 30:
        return {'status': 'insufficient_reliable_torso', 'finite_keypoint_frame_frac': float(finite.all(1).mean()),
                'confidence_joint_frac': float(confidence.mean()), 'box_ok_frac': float(visible.mean()),
                'torso_gate_frac': float(torso.mean()), 'eligible_torso_frames': int(eligible.sum()),
                'tracking_jump_edges': None, 'missing_transition_flags': None,
                'coverage': 0.0, 'eligible_windows': 0, 'torso_scale_px': None}
    scale = float(np.median(length[eligible]))
    clean = np.where(np.isfinite(k), k, 0.)
    steps = np.linalg.norm(np.diff(clean, axis=0), axis=-1)/scale
    raw_jump = np.any(steps > cfg.maximum_joint_step_torso, axis=1)
    finite_pair = (np.isfinite(k[:-1]).all(-1) & np.isfinite(k[1:]).all(-1) &
                   np.isfinite(c[:-1]) & np.isfinite(c[1:]))
    jump = np.any((steps > cfg.maximum_joint_step_torso) & finite_pair, axis=1)
    if not finite_transition_only:
        jump = raw_jump
    valid[1:] &= ~jump[:, None]
    valid[:-1] &= ~jump[:, None]
    valid &= valid[:, [11, 12]].all(1)[:, None]
    cm = valid[:, COCO]
    # The native window audit is bounded by the recorded pose/video endpoint.
    time = np.asarray(times, dtype=float) if times is not None else np.arange(len(k))/float(meta['fps_eff'])
    duration = len(k)/float(meta['fps_eff'])
    target = np.arange(int(np.ceil(duration*cfg.fps)))/cfg.fps
    hi = np.searchsorted(time, target, side='left')
    hi = np.clip(hi, 1, len(time)-1)
    lo = hi-1
    ok = ((target >= time[0]) & (target <= time[-1]) &
          ((time[hi]-time[lo]) <= cfg.maximum_interpolation_gap_s))
    mask = cm[lo] & cm[hi] & ok[:, None]
    exact_hi = np.isclose(target, time[hi], atol=1e-8, rtol=0)
    exact_lo = np.isclose(target, time[lo], atol=1e-8, rtol=0)
    mask[exact_hi] = cm[hi[exact_hi]]
    mask[exact_lo] = cm[lo[exact_lo]]
    good = mask.sum(1) >= 10
    window = np.convolve(good.astype(float), np.ones(cfg.frames)/cfg.frames, mode='valid')
    return {'status': 'ok', 'finite_keypoint_frame_frac': float(finite.all(1).mean()),
            'confidence_joint_frac': float(confidence.mean()), 'box_ok_frac': float(visible.mean()),
            'torso_gate_frac': float(torso.mean()), 'eligible_torso_frames': int(eligible.sum()),
            'tracking_jump_edges': int(jump.sum()), 'raw_tracking_jump_edges': int(raw_jump.sum()),
            'missing_transition_flags': int((raw_jump & ~jump).sum()),
            'coverage': float(good.mean()), 'eligible_windows': int((window >= cfg.minimum_coverage).sum()),
            'torso_scale_px': scale, 'native_duration_s': float(time[-1]),
            'video_frame_duration_s': duration, 'timestamp_gap_count': int((np.diff(time) > cfg.maximum_interpolation_gap_s).sum()),
            'maximum_timestamp_gap_s': float(np.max(np.diff(time))) if len(time) > 1 else 0.0}


def _audit_pose_coach(path: Path, coach_index: int, audio_duration_s: float, reasons: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        k, c, box, times = z['K'].astype(float), z['C'].astype(float), z['box'].astype(float), z['t'].astype(float)
        meta = json.loads(str(z['meta']))
    if k.ndim == 3:
        k, c, box = k[:, None], c[:, None], box[:, None]
    k, c, box = k[:, coach_index], c[:, coach_index], box[:, coach_index]
    old = _apply_jump_gate(k, c, box, meta, times, finite_transition_only=False)
    fixed = _apply_jump_gate(k, c, box, meta, times, finite_transition_only=True)
    native_end = float(times[-1])
    video_end = float(len(times)/float(meta['fps_eff']))
    audio_tail = max(0.0, float(audio_duration_s)-video_end)
    return {'key': path.stem, 'coach_index': int(coach_index), 'reasons': reasons,
            'audio_duration_s': float(audio_duration_s), 'native_duration_s': native_end,
            'video_frame_duration_s': video_end, 'audio_tail_s': audio_tail,
            'audio_pose_diff_s': float(audio_duration_s-native_end),
            'old_gate': old, 'fixed_gate': fixed,
            'duration_binding': ('audio_tail_only' if audio_tail <= 1.0 and audio_duration_s >= video_end-1.0
                                 else 'unresolved_pose_audio_binding')}


def _muq_audit(stage_ids: set[str], stage_partitions: dict[str, str]) -> dict:
    metadata = [p for p in MUQ_NATIVE.glob('*.json') if not p.name.endswith('.sha256.json') and p.stem != 'report']
    model = json.loads(MUQ_MODEL.read_text()) if MUQ_MODEL.exists() else {}
    weights = json.loads(MUQ_WEIGHTS.read_text()) if MUQ_WEIGHTS.exists() else {}
    rows, failures = [], []
    for meta_path in sorted(metadata):
        m = json.loads(meta_path.read_text())
        issues = []
        for key, expected in [('model_revision', model.get('revision')), ('native_hz', model.get('native_hz'))]:
            if m.get(key) != expected:
                issues.append(f'{key}_mismatch')
        if m.get('context_s') != 30 or m.get('core_s') != 20 or m.get('time_origin_s') != 0.0:
            issues.append('context_or_origin_mismatch')
        rid = m.get('recording_id')
        wav = Path(m.get('source_waveform', ''))
        if not wav.exists():
            issues.append('waveform_missing')
        else:
            if _sha(wav) != m.get('waveform_sha256'):
                issues.append('waveform_hash_mismatch')
        layer_checks = []
        for layer in model.get('layers', [3, 6, 9, 12]):
            q = MUQ_NATIVE/f'{rid}_layer{layer}.npy'
            expected_artifact = m.get('artifacts', {}).get(str(q), {})
            if not expected_artifact:
                issues.append(f'layer{layer}_metadata_missing'); continue
            check = _finite_hash(q, int(expected_artifact.get('bytes', -1)), expected_artifact.get('sha256'))
            check['layer'] = layer
            try:
                arr = np.load(q, mmap_mode='r')
                check['shape'] = list(arr.shape); check['dtype'] = str(arr.dtype)
                check['shape_ok'] = arr.ndim == 2 and arr.shape == (int(m['frames']), 1024)
                # Full-array finite checks are deliberate: this is a cache reuse gate.
                check['finite_ok'] = bool(np.isfinite(arr).all())
            except Exception as error:  # noqa: BLE001
                check.update({'shape_ok': False, 'finite_ok': False, 'error': f'{type(error).__name__}: {error}'})
            if not check.get('bytes_ok') or not check.get('sha256_ok') or not check.get('shape_ok') or not check.get('finite_ok'):
                issues.append(f'layer{layer}_integrity_failure')
            layer_checks.append(check)
        row = {'recording_id': rid, 'partition': m.get('partition'), 'in_stage11': rid in stage_ids,
               'stage_partition': stage_partitions.get(rid), 'model_revision': m.get('model_revision'),
               'native_hz': m.get('native_hz'), 'context_s': m.get('context_s'), 'core_s': m.get('core_s'),
               'frames': m.get('frames'), 'timestamp_contract': 'time_origin=0; implied t=i/25; no explicit timestamp array',
               'issues': issues, 'layers': layer_checks}
        rows.append(row)
        if issues: failures.append({'recording_id': rid, 'issues': issues})
    return {'status': 'verified' if not failures else 'diagnostic_failures',
            'model': model, 'weights': weights, 'metadata_files': len(metadata),
            'layer_file_count': len(list(MUQ_NATIVE.glob('*_layer*.npy'))),
            'stage11_overlap': sum(r['in_stage11'] for r in rows), 'stage11_missing_cache': len(stage_ids-{r['recording_id'] for r in rows}),
            'partitions': dict(Counter(r['partition'] for r in rows)), 'failures': failures, 'rows': rows}


def audit(output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stage = json.loads((STAGE/'data/manifest.json').read_text())
    songs = json.loads((CACHE/'justdance_songs.json').read_text())
    catalog = json.loads((CACHE/'justdance_manifest.json').read_text())
    old_native = json.loads((OLD_NATIVE/'data/MANIFEST.json').read_text())
    saved_audit = json.loads((Path('/root/lightsem/docs/research/2026-10-07-native-full-data/DATA_AUDIT.json')).read_text())
    stage_by_id = {r['recording_id']: r for r in stage['recordings']}
    stage_ids = set(stage_by_id)
    legacy_ids = {r['yt_id'] for r in songs['rows']}
    new_rows = [r for r in stage['recordings'] if not r['is_old']]
    protected_old_val_ids = {r['key'] for r in old_native['entries'] if r.get('partition') != 'train'}
    protected_old_val_cids = {stage_by_id[k]['canonical_id'] for k in protected_old_val_ids if k in stage_by_id}
    stage_holdout_cids = {r['canonical_id'] for r in stage['recordings'] if r['partition'] in ('val', 'test')}
    protected_cids = protected_old_val_cids | stage_holdout_cids
    reason_by = defaultdict(list)
    for r in saved_audit['rejected']:
        if 'coach_index' in r:
            reason_by[(r['key'], r['coach_index'])].append(r['reason'])

    coaches = []
    for rec in stage['recordings']:
        pose = CACHE/'justdance_pose'/f"{rec['recording_id']}.npz"
        wav = STAGE/'data/wav'/f"{rec['recording_id']}.wav"
        if not pose.exists() or not wav.exists():
            continue
        duration = sf.info(wav).duration
        with np.load(pose, allow_pickle=False) as z:
            n_coaches = z['K'].shape[1] if z['K'].ndim == 4 else 1
        for ci in range(n_coaches):
            coaches.append(_audit_pose_coach(pose, ci, duration, reason_by.get((rec['recording_id'], ci), [])))

    def has_window(row):
        return row['fixed_gate'].get('status') == 'ok' and row['fixed_gate'].get('eligible_windows', 0) > 0
    # One independently tracked coach per canonical family; no averaging.
    best = {}
    for row in coaches:
        rec = stage_by_id[row['key']]
        row['canonical_id'] = rec['canonical_id']; row['stage_partition'] = rec['partition']
        row['identity_status'] = rec['identity_status']; row['is_old'] = rec['is_old']
        row['eligible_core_stage11'] = rec['eligible_core']
        if not has_window(row):
            continue
        if rec['canonical_id'] in protected_cids:
            part = 'test' if rec['partition'] == 'test' else 'val'
        elif rec['partition'] == 'train':
            part = 'train'
        else:
            continue
        row['admission_partition'] = part
        # New conservative identities stay visible in the audit but are not
        # promoted to a training group until identity ambiguity is resolved.
        if part == 'train' and row['identity_status'] != 'verified' and not row['is_old']:
            row['admission_status'] = 'identity_review'
            continue
        row['admission_status'] = 'candidate'
        rank = (-row['fixed_gate'].get('coverage', 0.), row['key'], row['coach_index'])
        key = (part, rec['canonical_id'])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, row)

    entries = []
    for (part, cid), (_, row) in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
        rec = stage_by_id[row['key']]
        entries.append({'domain': 'jd', 'partition': part, 'canonical_id': cid,
                        'recording_id': row['key'], 'coach_index': row['coach_index'],
                        'pose_path': str(CACHE/'justdance_pose'/f"{row['key']}.npz"),
                        'wav_path': str(STAGE/'data/wav'/f"{row['key']}.wav"),
                        'coverage': row['fixed_gate']['coverage'],
                        'whole_record_qualified_80pct': bool(row['fixed_gate']['coverage'] >= Quality().minimum_coverage),
                        'eligible_windows': row['fixed_gate']['eligible_windows'],
                        'duration_s': row['video_frame_duration_s'],
                        'audio_tail_s': row['audio_tail_s'],
                        'identity_status': rec['identity_status'], 'is_old': rec['is_old'],
                        'pose_sha256': sha(CACHE/'justdance_pose'/f"{row['key']}.npz"),
                        'wav_sha256': sha(STAGE/'data/wav'/f"{row['key']}.wav")})
    entry_cids = {e['canonical_id'] for e in entries}
    whole_entries = [e for e in entries if e['whole_record_qualified_80pct']]
    audio_rows = []
    for exclusion in stage['source_exclusions']:
        src = SOURCE/exclusion['source']['file']
        stats = _source_audio_stats(src) if src.exists() else {'decode_ok': False, 'error': 'source_missing'}
        manifest_mean = exclusion['source'].get('mean_db')
        fresh_mean = stats.get('fresh_mean_db')
        audio_rows.append({'recording_id': exclusion['recording_id'], 'reason': exclusion['reason'],
                           'source': exclusion['source'], 'audio_stats': stats,
                           'manifest_mean_db': manifest_mean, 'fresh_mean_db': fresh_mean,
                           'manifest_level_stale': bool(fresh_mean is not None and manifest_mean is not None and abs(fresh_mean-manifest_mean) > 1.0),
                           'fresh_policy': ('usable_after_explicit_gain_normalization' if exclusion['recording_id'] == '-G-6DNPGEfk'
                                            else 'usable_source_audio_but_pose_not_extracted' if fresh_mean is not None and fresh_mean > -40
                                            else 'silent_or_unusable')})

    stage_partitions = {rid: r['partition'] for rid, r in stage_by_id.items()}
    prior_muq = output/'MUQ_CACHE_AUDIT.json'
    if prior_muq.exists():
        muq = json.loads(prior_muq.read_text())
        muq['reused_verified_receipt'] = str(prior_muq)
    else:
        muq = _muq_audit(stage_ids, stage_partitions)
    old_window = [r for r in coaches if r['old_gate'].get('status') == 'ok' and r['old_gate'].get('eligible_windows', 0) > 0]
    fixed_window = [r for r in coaches if has_window(r)]
    fixed_whole_pass = [r for r in fixed_window if r['fixed_gate'].get('coverage', 0.) >= Quality().minimum_coverage]
    fixed_whole_loss = [r for r in fixed_window if r['fixed_gate'].get('coverage', 0.) < Quality().minimum_coverage]
    best_window_by_group = {}
    for row in fixed_window:
        key = (row['stage_partition'], row.get('canonical_id'))
        rank = (-row['fixed_gate'].get('coverage', 0.), row['key'], row['coach_index'])
        if key not in best_window_by_group or rank < best_window_by_group[key][0]:
            best_window_by_group[key] = (rank, row)
    best_whole_loss = [row for _, row in best_window_by_group.values()
                       if row['fixed_gate'].get('coverage', 0.) < Quality().minimum_coverage]
    mismatch = [r for r in coaches if r['old_gate'].get('status') == 'ok' and r['old_gate'].get('eligible_windows', 0) > 0 and abs(r['audio_pose_diff_s']) > Quality().maximum_av_duration_difference_s]
    saved_best = {}
    for row in saved_audit['candidates']:
        key = row.get('known_group') or row['key']
        rank = (-row['quality']['coverage'], row['key'], row['coach_index'])
        if key not in saved_best or rank < saved_best[key][0]:
            saved_best[key] = (rank, row)
    saved_best_window_loss_minutes = sum(row['duration_s'] for _, row in saved_best.values()
                                         if not row['strict_original_record_QA'])/60
    groups = lambda rows: len({(r['stage_partition'], r.get('canonical_id')) for r in rows if r.get('canonical_id')})
    report = {
        'schema_version': 1, 'status': 'admission_audit_not_training',
        'inputs': {str(p): sha(p) for p in [STAGE/'data/manifest.json', CACHE/'justdance_songs.json', CACHE/'justdance_manifest.json', OLD_NATIVE/'data/MANIFEST.json']},
        'inventory': {'legacy_song_rows': len(songs['rows']), 'legacy_song_excluded': sum(bool(r.get('excluded')) for r in songs['rows']),
                      'catalog_recordings': len(catalog), 'stage_recordings': len(stage['recordings']),
                      'stage_partitions': dict(Counter(r['partition'] for r in stage['recordings'])),
                      'stage_new_not_legacy_by_id': len(stage_ids-legacy_ids),
                      'stage_new_partition_by_id': dict(Counter(r['partition'] for r in new_rows)),
                      'stage_new_identity_status': dict(Counter(r['identity_status'] for r in new_rows)),
                      'stage_new_eligible_core': {f'{part}:{eligible}': count for (part, eligible), count in
                                                  Counter((r['partition'], r['eligible_core']) for r in new_rows).items()},
                      'known_finding_316_reproduced': len(stage_ids-legacy_ids) == 316,
                      'known_finding_316_note': 'The supplied files produce 337 new recording IDs (232 TRAIN, 54 VAL, 51 TEST), not 316 (214/53/49).',
                      'natural_316_subset': '337 minus 19 no_qualified_coach and 2 missing_or_short_audio_fingerprint; this subset is 211/54/51, so it does not reproduce 214/53/49.'},
        'held_out_protection': {'stage11_val_test_canonical_groups': len(stage_holdout_cids),
                                'old_native_val_groups': len(protected_old_val_cids),
                                'union_protected_canonical_groups': len(protected_cids),
                                'protected_groups_with_eligible_window': len(entry_cids & protected_cids),
                                'protected_groups_without_eligible_window': len(protected_cids-entry_cids),
                                'protected_groups_without_window_ids': sorted(protected_cids-entry_cids),
                                'stage11_val_test_ids': sorted(stage_holdout_cids),
                                'old_native_val_ids_mapped': sorted(protected_old_val_cids)},
        'coach_audit': {'rows': len(coaches), 'old_window_tracks': len(old_window), 'fixed_window_tracks': len(fixed_window),
                        'old_window_recordings': len({r['key'] for r in old_window}), 'fixed_window_recordings': len({r['key'] for r in fixed_window}),
                        'old_window_minutes': sum(r['video_frame_duration_s'] for r in old_window)/60,
                        'fixed_window_minutes': sum(r['video_frame_duration_s'] for r in fixed_window)/60,
                        'missing_transition_flags': sum(r['fixed_gate'].get('missing_transition_flags', 0) or 0 for r in coaches),
                        'raw_jump_edges': sum(r['old_gate'].get('raw_tracking_jump_edges', 0) or 0 for r in coaches),
                        'finite_pair_jump_edges': sum(r['fixed_gate'].get('tracking_jump_edges', 0) or 0 for r in coaches),
                        'duration_mismatch_window_tracks': len(mismatch),
                        'duration_mismatch_window_minutes': sum(r['video_frame_duration_s'] for r in mismatch)/60,
                        'timestamp_gap_tracks': sum(r['fixed_gate'].get('timestamp_gap_count', 0) > 0 for r in coaches),
                        'timestamp_gap_events': sum(r['fixed_gate'].get('timestamp_gap_count', 0) or 0 for r in coaches),
                        'maximum_timestamp_gap_s': max((r['fixed_gate'].get('maximum_timestamp_gap_s', 0.) or 0.) for r in coaches),
                        'whole_record_vs_window': {
                            'fixed_window_tracks': len(fixed_window),
                            'fixed_window_tracks_passing_whole_record_80pct': len(fixed_whole_pass),
                            'fixed_window_tracks_lost_to_whole_record_gate': len(fixed_whole_loss),
                            'fixed_window_minutes_lost_to_whole_record_gate': sum(r['video_frame_duration_s'] for r in fixed_whole_loss)/60,
                            'best_coach_per_group_loss_groups': len(best_whole_loss),
                            'best_coach_per_group_loss_minutes': sum(r['video_frame_duration_s'] for r in best_whole_loss)/60,
                            'best_coach_per_group_train_loss_groups': sum(r['stage_partition'] == 'train' for r in best_whole_loss),
                            'best_coach_per_group_train_loss_minutes': sum(r['video_frame_duration_s'] for r in best_whole_loss if r['stage_partition'] == 'train')/60,
                            'saved_native_audit_comparison': {
                                'candidate_tracks': len(saved_audit['candidates']),
                                'candidate_recordings': saved_audit['candidate_recordings'],
                                'strict_whole_record_tracks': saved_audit['strict_original_record_QA_tracks'],
                                'masked_window_only_tracks': sum(not r['strict_original_record_QA'] for r in saved_audit['candidates']),
                                'masked_window_only_minutes': sum(r['duration_s'] for r in saved_audit['candidates'] if not r['strict_original_record_QA'])/60,
                                'masked_window_only_best_group_minutes': saved_best_window_loss_minutes,
                                'note': 'The saved audit is the supplied legacy-map audit; its best-group minutes are retained as an auditable comparison value.'},
                            'definition': 'source video minutes attached to a coach with at least one eligible 16-second masked window but <80% whole-record valid frames; group totals choose the highest-coverage coach and never average people.',
                        },
                        'whole_record_vs_window_loss_note': 'Minutes are source video duration attached to tracks with a valid 16-second masked window; they are not independent song hours.'},
        'admission': {'candidate_rows': len([r for r in coaches if r.get('admission_status') == 'candidate']),
                      'identity_review_rows': len([r for r in coaches if r.get('admission_status') == 'identity_review']),
                      'window_recoverable_train_groups': len({e['canonical_id'] for e in entries if e['partition'] == 'train'}),
                      'window_recoverable_train_minutes': sum(e['duration_s'] for e in entries if e['partition'] == 'train')/60,
                      'whole_record_qualified_train_groups': len({e['canonical_id'] for e in whole_entries if e['partition'] == 'train'}),
                      'whole_record_qualified_train_minutes': sum(e['duration_s'] for e in whole_entries if e['partition'] == 'train')/60,
                      'window_only_train_groups': len({e['canonical_id'] for e in entries if e['partition'] == 'train' and not e['whole_record_qualified_80pct']}),
                      'window_only_train_minutes': sum(e['duration_s'] for e in entries if e['partition'] == 'train' and not e['whole_record_qualified_80pct'])/60,
                      'protected_val_groups_with_window': len({e['canonical_id'] for e in entries if e['partition'] == 'val'}),
                      'protected_test_groups_with_window': len({e['canonical_id'] for e in entries if e['partition'] == 'test'}),
                      'partition_entries': dict(Counter(e['partition'] for e in entries)),
                      'manifest_policy': 'one tracked coach per canonical group; finite original-time masked window; fixed missing-transition jump audit; audio tail recorded explicitly; conservative new identities remain review-only; held-out groups never moved to TRAIN'},
        'audio_exclusions': audio_rows,
        'muq_cache': {'path': str(MUQ_NATIVE),
                      'cache_bytes': sum(p.stat().st_size for p in MUQ_NATIVE.rglob('*') if p.is_file()),
                      'audit': muq},
        'quality_vs_generator': {'extraction_quality': 'This report audits pose/audio/identity/feature admission only.',
                                 'generator_quality': 'Not measured here; no checkpoint, training, native body, DMX, or baseline was changed or evaluated.'},
        'evidence': {'directory': str(output/'evidence'), 'contact_sheet': str(output/'evidence/contact_sheet.jpg'),
                     'frames': sorted(str(p) for p in (output/'evidence').glob('*.jpg'))},
    }
    write_json(output/'REPORT.json', report)
    write_json(output/'COACH_AUDIT.json', coaches)
    write_json(output/'TRAIN_MANIFEST.json', {'schema_version': 1, 'status': 'admission_manifest_not_trained',
                                              'entries': entries, 'source_report_sha256': sha(output/'REPORT.json'),
                                              'selection': report['admission']['manifest_policy']})
    write_json(output/'AUDIO_EXCLUSIONS.json', audio_rows)
    write_json(output/'MUQ_CACHE_AUDIT.json', muq)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps({'status': report['status'], 'coach_audit': report['coach_audit'], 'admission': report['admission'],
                      'muq': {k: report['muq_cache']['audit'][k] for k in ('status', 'metadata_files', 'stage11_overlap', 'stage11_missing_cache')}}, indent=2))


if __name__ == '__main__':
    main()
