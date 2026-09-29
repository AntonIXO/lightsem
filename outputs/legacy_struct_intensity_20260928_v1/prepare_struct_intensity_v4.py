#!/usr/bin/env python3
"""Preparation only. No model construction, forward, scoring or GPU stage exists.

Payload access is unreachable without a future root authorization that binds the
exact plan, source, tests, native test receipt, command and scientific allowlist.
The CPU checkpoint reader uses restricted torch.load; no unsafe fallback exists.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import stat
import sys
import time
import zipfile

VERSION = 4
GLOBAL_CAP = 1020054732
PACKET_CAP = 12582912
COMPARISON_RESERVE = 33554432
FREE_FLOOR = 1073741824
ORIGINAL = Path('/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z')
WORKSPACE = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1')
COMPARISON = WORKSPACE / 'comparison_demos_v1'
PACKET = WORKSPACE / 'legacy_struct_intensity_v4'
SOURCE_DIR = Path('/root/lightsem/outputs/legacy_struct_intensity_20260928_v1')
PLAN_DIR = Path('/root/lightsem/docs/research/2026-09-28-end-to-end')
PLAN_PATH = PLAN_DIR / 'PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V4.json'
MD_PATH = PLAN_DIR / 'PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V4.md'
AUDIO_ROOT = Path('/mnt/lightsem-data/cache/justdance_audio')
ROLE_ROOT = Path('/mnt/lightsem-data/cache/justdance_structure')
ARMS = ('none', 'struct', 'shuf')
ROLES = ('intro', 'breakdown', 'build-up', 'drop', 'outro')
# Frozen producer contract: meta.duration_s is the last centered STFT frame.
# Only the final Segment.end_s uses waveform len(y)/SR, up to one hop later.
STRUCTURE_FEATURE_SR = 22050
STRUCTURE_FEATURE_HOP = 512
STRUCTURE_FINAL_TAIL_BOUND_S = STRUCTURE_FEATURE_HOP / STRUCTURE_FEATURE_SR
STRUCTURE_ENDPOINT_ROUNDOFF_S = 1e-9
PRESERVED_PROTOCOLS = (
    PLAN_DIR / 'PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V3.json',
    PLAN_DIR / 'PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V3.md',
)
A_NAMES = ('metric_level', 'vert_period_s', 'vert_bounce_amp', 'com_speed')
C_NAMES = ('qom', 'head_speed', 'foot_speed', 'weight', 'time_eff', 'spread',
           'section_energy', 'hand_speed', 'hand_spread', 'lateral_pos',
           'travel_speed', 'symmetry', 'laterality', 'space_eff', 'flow_eff',
           'jerk', 'vert_bounce')
AUDIO_SCALARS = ('rms', 'rms_db', 'flux', 'flux_lf', 'flux_lmid', 'flux_mid',
 'flux_hmid', 'flux_hf', 'energy_lf', 'energy_lmid', 'energy_mid', 'energy_hmid',
 'energy_hf', 'onset', 'onset_ema', 'flux_ema', 'flux_lf_ema', 'flux_hf_ema',
 'centroid', 'flatness', 'rolloff', 'percussive_ratio', 'pulse_clarity',
 'beat_phase_sin', 'beat_phase_cos', 'beat_accent', 'bar_phase', 'beat_period_s')


class PreparationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PreparationError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise PreparationError('nonfinite JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=bad_constant)


def checked_path(value):
    p = Path(value)
    require(p.is_absolute() and p == p.resolve(strict=True), 'noncanonical or symlink path: ' + str(p))
    require(stat.S_ISREG(p.stat().st_mode), 'not a regular file: ' + str(p))
    return p


def read_ref(ref, max_bytes=134217728):
    require(isinstance(ref, dict), 'file reference must be an object')
    p = checked_path(ref['path'])
    expected = ref['sha256']
    require(isinstance(expected, str) and len(expected) == 64 and
            all(c in '0123456789abcdef' for c in expected), 'invalid SHA256')
    size = ref.get('size_bytes', p.stat().st_size)
    require(type(size) is int and 0 <= size <= max_bytes, 'reference size outside cap')
    require(p.stat().st_size == size, 'size mismatch: ' + str(p))
    with p.open('rb') as stream:
        data = stream.read(max_bytes + 1)
    require(len(data) == size and digest(data) == expected, 'hash mismatch: ' + str(p))
    return data


def local_ref(path, max_bytes=2097152):
    p = checked_path(path)
    require(p.stat().st_size <= max_bytes, 'control file exceeds bound')
    data = p.read_bytes()
    return {'path': str(p), 'size_bytes': len(data), 'sha256': digest(data)}, data


def validate_plan(plan):
    require(plan.get('document_revision') == VERSION, 'wrong protocol revision')
    stage = plan['preparation_v4']
    require(stage['scope'] == 'PREPARATION_ONLY_NO_MODEL_OR_SCORE', 'wrong preparation scope')
    require(stage['packet_root'] == str(PACKET), 'wrong packet root')
    require(stage['packet_cap_bytes'] == PACKET_CAP, 'packet cap changed')
    require(stage['global_cap_bytes'] == GLOBAL_CAP, 'global cap changed')
    require(stage['comparison_reserve_bytes'] == COMPARISON_RESERVE, 'comparison reserve changed')
    require(stage['original_root'] == str(ORIGINAL) and stage['workspace_root'] == str(WORKSPACE), 'accounting roots changed')
    rows = plan['cohorts']['clip_index']
    require(len(rows) == 40 and len({r['clip'] for r in rows}) == 40, 'must preserve exact40 unique clips')
    require(sum(r['T_metadata'] for r in rows) == 371393, 'frozen metadata frame total changed')
    recipients = {r['recording_id'] for r in rows}
    donors = {r['original_shuffled_donor_recording_id'] for r in rows}
    require(len(recipients) == 31 and donors <= recipients, 'recording/donor union changed')
    for row in rows:
        require(row['current_partition'] == 'train' and row['current_Stage11_TEST_aliases'] == [], 'known TEST alias or nonTRAIN row')
        require(type(row['T_metadata']) is int and row['T_metadata'] > 0, 'invalid metadata T')
        require(math.isfinite(row['start_s_metadata']) and row['start_s_metadata'] >= 0, 'invalid start_s')
        require(row['original_shuffled_donor_canonical_id'] != row['canonical_id'], 'canonical-self donor')
    expected_allowlist = {(str(AUDIO_ROOT / (mid + '.npz')), 'audio', mid) for mid in recipients}
    expected_allowlist |= {(str(ROLE_ROOT / (mid + '.npz')), 'structure', mid) for mid in recipients | donors}
    entries = stage['input_slots']
    require(len(entries) == 62 and {(r['path'], r['kind'], r['recording_id']) for r in entries} == expected_allowlist,
            'input slots not exact frozen62 file union')
    require(plan['qualified_gate_scope']['original_G_struct_contrast_completion_claim_permitted'] is False,
            'forbidden full-gate claim')


def validate_authorization(auth, plan, plan_ref, driver_ref, test_refs, argv):
    require(auth.get('scope') == 'PREPARATION_ONLY_NO_MODEL_OR_SCORE' and auth.get('authorized_by') == 'root', 'missing root preparation authority')
    require(auth.get('plan_ref') == plan_ref and auth.get('driver_ref') == driver_ref and auth.get('test_refs') == test_refs, 'authorization hash binding mismatch')
    require(auth.get('argv') == argv, 'exact command argv mismatch')
    for key in ('allow_model_construction', 'allow_model_forward', 'allow_scoring', 'allow_GPU', 'allow_TEST_access', 'allow_raw_manifest_access'):
        require(auth.get(key) is False, 'forbidden permission: ' + key)
    require(auth.get('allow_cpu_checkpoint_deserialization') is True, 'CPU checkpoint read not authorized')
    require(auth.get('accept_historical_descriptive_unknown_cross_corpus_aliases') is True,
            'descriptive identity limitation not prospectively accepted')
    require(auth.get('native_registration_receipt_verified') is True and isinstance(auth.get('native_registration_receipt_ref'), dict), 'missing native preparation registration receipt')
    receipt = auth['native_synthetic_test_result']
    require(receipt.get('exit_code') == 0 and receipt.get('failures') == 0 and receipt.get('errors') == 0, 'native tests not successful')
    require(all(receipt.get(key) == 0 for key in ('skipped', 'expected_failures', 'unexpected_successes')), 'native tests skipped or nonstandard success')
    require(type(receipt.get('tests_run')) is int and receipt['tests_run'] >= plan['preparation_v4']['minimum_synthetic_tests'], 'native tests missing required count')
    require(receipt['tests_run'] == receipt.get('tests_expected') == plan['preparation_v4']['expected_synthetic_tests'], 'native test count mismatch')
    require(receipt.get('plan_ref') == plan_ref and receipt.get('driver_ref') == driver_ref and receipt.get('test_refs') == test_refs, 'native test source bindings mismatch')
    require(receipt.get('real_model_forwards') == 0 and receipt.get('sealed_TEST_reads') == 0 and receipt.get('checkpoint_deserializations') == 0, 'native tests exceeded synthetic scope')
    require(receipt.get('command') == plan['preparation_v4']['native_synthetic_test_command'], 'native test command differs from exact published command')
    require(isinstance(receipt.get('log_ref'), dict), 'native test log not hash-bound')
    require(type(auth.get('other_reserved_bytes')) is int and auth['other_reserved_bytes'] >= 0, 'other reservation ledger missing')
    require(auth.get('preparation_external_limits_bound') == {'wall_seconds': 900, 'host_RAM_bytes': 4294967296, 'GPU_access': False}, 'CPU preparation process-tree limits not bound')
    require(auth.get('preparation_log') == {'path': str(SOURCE_DIR / 'preparation_v4.log'), 'max_bytes': 65536}, 'bounded internal preparation log not bound')
    require(auth.get('original_inventory_sha256') and auth.get('remaining_project_GPU_seconds', -1) >= 0, 'resource/original inventory binding missing')
    source_refs = auth['source_refs']
    require(source_refs == plan['preparation_v4']['source_refs'], 'source dependency bindings changed')
    require(auth['checkpoint_refs'] == plan['preparation_v4']['checkpoint_refs'], 'checkpoint hash map changed')
    slots = {r['path']: r for r in plan['preparation_v4']['input_slots']}
    approved = auth['input_refs']
    require(len(approved) == 62 and {r['path'] for r in approved} == set(slots), 'input allowlist is not exact62 paths')
    for entry in approved:
        slot = slots[entry['path']]
        require(entry['recording_id'] == slot['recording_id'] and entry['kind'] == slot['kind'], 'input identity binding changed')
        require(entry.get('current_Stage11_partition') == 'train' and entry.get('known_current_TEST_aliases') == [], 'unapproved partition/known TEST alias')
        require(entry.get('identity_projection_ref') == plan['preparation_v4']['identity_projection_ref'], 'identity evidence not bound')
        require(entry.get('cross_corpus_identity') == 'UNRESOLVED_NOT_INDEPENDENT', 'unknown identity limitation concealed')
        require(entry.get('approval') == 'ROOT_APPROVED_HISTORICAL_DESCRIPTIVE_ONLY', 'explicit file approval missing')
        require(type(entry.get('size_bytes')) is int and isinstance(entry.get('sha256'), str), 'payload bytes not prospectively bound')


def validate_native_test_log(data, expected):
    text = data.decode('utf-8')
    summaries = re.findall(r'^Ran (\d+) tests? in [0-9.]+s$', text, re.MULTILINE)
    successes = re.findall(r'^test_[^\n]+ \.\.\. ok$', text, re.MULTILINE)
    require(summaries == [str(expected)] and len(successes) == expected and text.rstrip().endswith('\nOK'),
            'native verbose log does not show exactly all required tests passing')
    require(not re.search(r'\.\.\. (skipped|expected failure|unexpected success|FAIL|ERROR)', text),
            'native verbose log contains nonpass result')


def expected_state_shapes(cfg):
    d, m = cfg['d_model'], cfg['d_model'] * cfg['ff_mult']
    shapes = {'inp.0.weight': (cfg['in_dim'],), 'inp.0.bias': (cfg['in_dim'],),
              'inp.1.weight': (d, cfg['in_dim']), 'inp.1.bias': (d,)}
    def norm(prefix):
        shapes[prefix + '.weight'] = (d,)
        shapes[prefix + '.bias'] = (d,)
    def linear(prefix, out_, in_):
        shapes[prefix + '.weight'] = (out_, in_)
        shapes[prefix + '.bias'] = (out_,)
    for i in range(cfg['n_blocks']):
        p = f'blocks.{i}.'
        for ff in ('ff1', 'ff2'):
            norm(p + ff + '.net.0')
            linear(p + ff + '.net.1', m, d)
            linear(p + ff + '.net.4', d, m)
        norm(p + 'attn.norm'); linear(p + 'attn.qkv', 3 * d, d); linear(p + 'attn.out', d, d)
        norm(p + 'conv.norm'); norm(p + 'norm')
        for suffix, shape in (('dw', (d, 1, cfg['conv_kernel'])), ('pw1', (2 * d, d, 1)), ('pw2', (d, d, 1))):
            shapes[p + 'conv.' + suffix + '.weight'] = shape
            shapes[p + 'conv.' + suffix + '.bias'] = (shape[0],)
    linear('head_a', cfg['n_a'], d); linear('head_b', cfg['n_b'], d)
    linear('head_c.net.0', 320, d + 2 * cfg['n_c'] + 1)
    linear('head_c.net.2', 320, 320); linear('head_c.net.4', cfg['n_c'], 320)
    return shapes


def validate_checkpoint_metadata(blob, arm, plan, np_module):
    np = np_module
    require(arm in ARMS and isinstance(blob, dict), 'invalid checkpoint metadata')
    require(set(blob) == {'model', 'cfg', 'mu', 'sd', 'block_names', 'event_thr', 'smooth_tau'}, 'unexpected or missing checkpoint fields')
    cfg = dict(plan['generation_and_mapper']['model_configuration'])
    cfg.pop('expected_dim_none'); cfg.pop('expected_dim_struct_shuf')
    cfg.update(in_dim=92 if arm == 'none' else 100, ff_mult=4, dropout=0.1)
    require(blob['cfg'] == cfg, 'checkpoint config differs from exact frozen contract')
    require(blob['block_names'] == {'a': list(A_NAMES), 'c': list(C_NAMES)}, 'checkpoint block names/order mismatch')
    for name in ('mu', 'sd'):
        value = np.asarray(blob[name])
        require(value.shape == (cfg['in_dim'],) and value.dtype == np.float32 and np.isfinite(value).all(), 'invalid checkpoint normalizer ' + name)
        if name == 'sd':
            require((value > 0).all(), 'nonpositive checkpoint sd')
    calibration = plan['lineage']['calibration_reports'][arm]
    require(blob['event_thr'] == calibration['event_thresholds']['thresholds'], 'event thresholds do not match frozen report')
    require(blob['smooth_tau'] == calibration['smoothing']['smooth_tau'], 'smoothing does not match frozen report')
    shapes = expected_state_shapes(cfg)
    require(set(blob['model']) == set(shapes), 'checkpoint state keys differ')
    for key, shape in shapes.items():
        tensor = blob['model'][key]
        require(tuple(tensor.shape) == shape, 'checkpoint tensor shape mismatch: ' + key)
        require(str(tensor.dtype) == 'torch.float32', 'checkpoint state is not float32: ' + key)
        require(str(tensor.device) == 'cpu', 'checkpoint state is not CPU: ' + key)
    return {'arm': arm, 'config': cfg, 'block_names': blob['block_names'], 'state_tensor_count': len(shapes),
            'normalizers_validated': True, 'calibration_exact': True, 'model_constructed': False,
            'state_values_not_scored_or_used_in_forward': True}


def load_checkpoint_cpu(data):
    # This function is called ONLY after root authorization and byte checks.
    import numpy as np
    import torch
    core = getattr(np, '_core', None)
    if core is None:
        core = np.core
    allowed = [core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)


def scalar_meta(value):
    if hasattr(value, 'shape'):
        require(value.shape == (), 'metadata must be scalar JSON')
        value = value.item()
    require(isinstance(value, (str, bytes)), 'metadata is not JSON text')
    result = parse_json(value)
    require(isinstance(result, dict), 'metadata is not an object')
    return result


def finite_array(value, shape, name, np):
    arr = np.asarray(value)
    require(arr.shape == shape and arr.dtype.kind in 'fiu' and np.isfinite(arr).all(), 'invalid shape/dtype/finites: ' + name)
    require(np.all(np.abs(arr.astype(np.float64)) <= np.finfo(np.float32).max), 'float32 feature overflow: ' + name)
    return arr


def clock_array(value, np, name):
    times = np.asarray(value)
    require(times.ndim == 1 and 0 < len(times) <= 100000 and times.dtype.kind == 'f', 'invalid time vector: ' + name)
    require(np.isfinite(times).all(), 'nonfinite clock: ' + name)
    expected = np.arange(len(times), dtype=np.float64) / 40.0
    require(np.all(np.abs(times.astype(np.float64) - expected) <= 1e-4), 'clock origin/grid mismatch: ' + name)
    deltas = np.diff(times.astype(np.float64))
    require(len(times) == 1 or np.all(deltas > 0), 'nonmonotone clock: ' + name)
    require(len(times) == 1 or np.all(np.abs(deltas - 0.025) <= 1e-4), 'adjacent clock step mismatch: ' + name)
    return times


def validate_bus(bus_dict, np_module):
    np = np_module
    meta = scalar_meta(bus_dict['meta'])
    require(meta.get('fps') == 40.0, 'bus fps must be40')
    times = clock_array(bus_dict['times'], np, 'bus.times')
    n = len(times)
    finite_array(bus_dict['mel'], (64, n), 'mel', np)
    for name in AUDIO_SCALARS:
        finite_array(bus_dict['ch__' + name], (n,), name, np)
    for name in ('beats', 'downbeats'):
        vals = np.asarray(bus_dict[name])
        require(vals.ndim == 1 and vals.dtype.kind in 'fi' and np.isfinite(vals).all(), 'invalid beat array')
        require(len(vals) < 2 or np.all(np.diff(vals) > 0), 'beat times not strictly increasing')
    require(math.isfinite(float(meta['duration_s'])) and meta['duration_s'] > 0, 'invalid bus duration')
    require(abs(float(meta['duration_s']) - n / 40.0) <= 0.025 + 1e-4, 'bus duration/grid mismatch')
    require(math.isfinite(float(meta['tempo'])), 'invalid bus tempo')
    flags = meta.get('flags', {})
    require(isinstance(flags, dict), 'bus flags must be an object when present')
    return {'frames': n, 'fps': 40.0, 'oracle_beat': flags.get('oracle_beat', 'UNKNOWN')}


def validate_structure(struct_dict, np_module):
    np = np_module
    meta = scalar_meta(struct_dict['meta'])
    times = clock_array(struct_dict['t'], np, 'structure.t')
    n = len(times)
    require(tuple(meta['roles']) == ROLES, 'structure role order differs')
    duration = float(meta['duration_s'])
    require(math.isfinite(duration) and duration > 0 and abs(n / 40.0 - duration) <= 0.0125 + 1e-4, 'structure duration/grid mismatch')
    role = np.asarray(struct_dict['role'])
    require(role.shape == (n,) and role.dtype.kind in 'iu' and np.all((role >= 0) & (role <= 4)), 'invalid role values; do not clip')
    missing = {}
    for key in ('energy', 'to_rise_s', 'to_drop_s'):
        arr = np.asarray(struct_dict[key])
        require(arr.shape == (n,) and arr.dtype.kind == 'f' and not np.isinf(arr).any(), 'invalid structure feature: ' + key)
        missing[key] = int(np.isnan(arr).sum())
        transformed = np.nan_to_num(arr, nan=0.0)
        if key != 'energy':
            transformed = np.tanh(transformed / 30.0)
        require(np.isfinite(transformed).all(), 'nonfinite transformed conditioning')
        require(np.all(np.abs(transformed.astype(np.float64)) <= np.finfo(np.float32).max), 'float32 conditioning overflow: ' + key)
    valid = np.zeros(n, dtype=bool)
    sections = np.full(n, 65535, dtype=np.uint16)
    segments = meta['segments']
    require(isinstance(segments, list) and len(segments) < 65535, 'invalid section table size')
    previous_end = 0.0
    # Mathematical sample clock, not rounded float32 cache timestamps.
    sample_times = np.arange(n, dtype=np.float64) / 40.0
    for sid, seg in enumerate(segments):
        a, b, label = float(seg['start_s']), float(seg['end_s']), seg['role']
        final = sid == len(segments) - 1
        # Retain the original start/intermediate endpoint/overlap constraints.
        # A final-endpoint exception must not become a blanket clock tolerance.
        require(math.isfinite(a) and math.isfinite(b) and 0 <= a < b and
                a <= duration + 1e-4 and a >= previous_end,
                'invalid/overlapping/out-of-duration cached segment')
        if final:
            require(b <= duration + STRUCTURE_FINAL_TAIL_BOUND_S + STRUCTURE_ENDPOINT_ROUNDOFF_S,
                    'final segment exceeds producer waveform-tail bound')
        else:
            require(b <= duration + 1e-4, 'intermediate segment exceeds last STFT-center duration')
        require(label in ROLES, 'unknown cached section label')
        mask = (sample_times >= a) & (sample_times < b)
        require(np.all(role[mask] == ROLES.index(label)), 'cached role/segment disagreement')
        valid[mask] = True
        sections[mask] = sid
        previous_end = b
    return {'frames': n, 'roles': role, 'valid': valid, 'section_ids': sections,
            'segments': segments, 'missing_counts': missing,
            'cached_duration_last_STFT_center_s': duration,
            'final_endpoint_minus_STFT_center_s': (float(segments[-1]['end_s']) - duration) if segments else None,
            'final_endpoint_tail_bound_s': STRUCTURE_FINAL_TAIL_BOUND_S,
            'cache_arrays_not_extended': True,
            'endpoint_check_scope': 'METADATA_CONSISTENCY_ONLY_NOT_WAVEFORM_END_VERIFICATION'}


def qualify_clip(row, bus_dict, structure_info, donor_info, np_module):
    np = np_module
    bus = validate_bus(bus_dict, np)
    t = row['T_metadata']; raw = row['start_s_metadata']
    require(type(t) is int and t > 0 and type(raw) in (int, float) and math.isfinite(raw) and raw >= 0, 'invalid frozen clip clock')
    i0 = int(round(raw * 40.0))
    require(i0 >= 0 and i0 + t <= bus['frames'], 'recipient bus shorter than exact i0:i0+T; no clamp/truncate/pad')
    require(abs(i0 / 40.0 - raw) <= 0.0125 + 1e-9, 'start rounding outside half-frame')
    source_times = np.asarray(bus_dict['times'][i0:i0+t], dtype=np.float64)
    require(len(source_times) == t and np.all(np.abs(source_times - (i0 + np.arange(t)) / 40.0) <= 1e-4), 'recipient slice clock mismatch')
    n = max(0, min(t, structure_info['frames'] - i0))
    valid = np.zeros(t, dtype=bool); roles = np.full(t, 255, dtype=np.uint8); section_ids = np.full(t, 65535, dtype=np.uint16)
    if n:
        valid[:n] = structure_info['valid'][i0:i0+n]
        roles[:n] = np.where(valid[:n], structure_info['roles'][i0:i0+n], 255)
        section_ids[:n] = np.where(valid[:n], structure_info['section_ids'][i0:i0+n], 65535)
    sections = []
    counts = {name: 0 for name in ROLES}
    for sid in np.unique(section_ids[valid]):
        mask = valid & (section_ids == sid)
        indices = np.flatnonzero(mask)
        label = structure_info['segments'][int(sid)]['role']
        retained = int(mask.sum()) >= 80
        if retained:
            counts[label] += int(mask.sum())
        sections.append({'section_id': int(sid), 'role': label, 'start_frame': int(indices[0]),
                         'end_frame_exclusive': int(indices[-1]) + 1, 'intersection_frames': int(mask.sum()),
                         'retained': retained, 'reason': None if retained else 'INTERSECTION_LT80_FRAMES'})
    valid_count = int(valid.sum())
    reasons = []
    if valid_count < math.ceil(0.95 * t):
        reasons.append('TARGET_VALID_COVERAGE_LT95_PERCENT')
    for label in ('drop', 'breakdown'):
        if counts[label] < 80:
            reasons.append(label.upper() + '_RETAINED_LT80_FRAMES')
    donor_n = min(t, donor_info['frames'])
    return {'clip': row['clip'], 'recording_id': row['recording_id'],
            'donor_recording_id': row['original_shuffled_donor_recording_id'],
            'T': t, 'fps': 40, 'raw_start_s': raw, 'i0': i0, 'rounded_origin_s': i0 / 40.0,
            'roundoff_s': i0 / 40.0 - raw, 'recipient_bus_frames_full': bus['frames'],
            'target_conditioning_observed_frames': n, 'target_conditioning_zero_tail_frames': t - n,
            'donor_conditioning_observed_frames': donor_n, 'donor_conditioning_zero_tail_frames': t - donor_n,
            'target_valid_role_frames': valid_count, 'target_invalid_role_frames': t - valid_count,
            'donor_valid_role_frames_within_conditioning': int(donor_info['valid'][:donor_n].sum()),
            'retained_role_frames': counts, 'sections': sections,
            'target_missing_feature_counts_full_recording': structure_info['missing_counts'],
            'donor_missing_feature_counts_full_recording': donor_info['missing_counts'],
            'structure_endpoint_metadata': {
                'target_last_STFT_center_s': structure_info['cached_duration_last_STFT_center_s'],
                'target_final_endpoint_minus_center_s': structure_info['final_endpoint_minus_STFT_center_s'],
                'donor_last_STFT_center_s': donor_info['cached_duration_last_STFT_center_s'],
                'donor_final_endpoint_minus_center_s': donor_info['final_endpoint_minus_STFT_center_s'],
                'scope': 'METADATA_CONSISTENCY_ONLY_NOT_WAVEFORM_END_VERIFICATION',
                'hop_bound_s': STRUCTURE_FINAL_TAIL_BOUND_S},
            'oracle_beat': bus['oracle_beat'], 'eligible': not reasons, 'reasons': reasons,
            'status': 'INPUT_QUALIFIED' if not reasons else 'ROLE_SUPPORT_INELIGIBLE',
            'ratio_or_prediction_computed': False}


def read_npz_ref(ref, kind, np):
    data = read_ref(ref, 33554432)
    required = ({'times', 'mel', 'beats', 'downbeats', 'meta'} | {'ch__' + k for k in AUDIO_SCALARS}) if kind == 'audio' else {'t', 'role', 'energy', 'to_rise_s', 'to_drop_s', 'meta'}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(len(entries) <= 128 and len({e.filename for e in entries}) == len(entries), 'duplicate/too many NPZ entries')
        require(sum(e.file_size for e in entries) <= 67108864, 'NPZ expanded bytes exceed cap')
        require(all(not e.is_dir() and '/' not in e.filename and not e.flag_bits & 1 for e in entries), 'unsafe NPZ member')
        require({key + '.npy' for key in required} <= {e.filename for e in entries}, 'missing required NPZ member')
    with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=16384) as archive:
        return {key: archive[key] for key in required}


def inventory(root):
    if not root.exists():
        return [], 0
    require(root.resolve() == root and root.is_dir(), 'noncanonical accounting root')
    rows = []
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), 'symlink in accounting tree: ' + str(path))
        if stat.S_ISREG(info.st_mode):
            require(info.st_nlink == 1, 'hardlink in accounting tree: ' + str(path))
            rows.append([str(path.relative_to(root)), info.st_size, info.st_mtime_ns])
        else:
            require(stat.S_ISDIR(info.st_mode), 'special file in accounting tree')
    return rows, sum(row[1] for row in rows)


def resource_values(original_bytes, workspace_bytes, comparison_bytes, external_packet_bytes,
                    packet_workspace_bytes, other_reserved_bytes, free_mnt_bytes, free_root_bytes):
    values = (original_bytes, workspace_bytes, comparison_bytes, external_packet_bytes,
              packet_workspace_bytes, other_reserved_bytes, free_mnt_bytes, free_root_bytes)
    require(all(type(v) is int and v >= 0 for v in values), 'resource counts must be nonnegative integers')
    require(comparison_bytes <= workspace_bytes and packet_workspace_bytes <= workspace_bytes and comparison_bytes + packet_workspace_bytes <= workspace_bytes, 'overlapping/invalid workspace accounting')
    packet_used = external_packet_bytes + packet_workspace_bytes
    require(packet_used <= PACKET_CAP and comparison_bytes <= COMPARISON_RESERVE, 'packet/comparison cap exceeded')
    remaining = PACKET_CAP - packet_used
    committed = original_bytes + workspace_bytes + external_packet_bytes + remaining + (COMPARISON_RESERVE - comparison_bytes) + other_reserved_bytes
    require(committed <= GLOBAL_CAP, 'combined campaign plus reservations exceeds cap')
    require(free_mnt_bytes >= FREE_FLOOR + remaining and free_root_bytes >= FREE_FLOOR + remaining,
            'free floor plus all remaining peak packet writes not available')
    return {'packet_used_bytes': packet_used, 'packet_remaining_bytes': remaining,
            'combined_committed_bytes': committed, 'global_cap_bytes': GLOBAL_CAP,
            'free_mnt_bytes': free_mnt_bytes, 'free_root_bytes': free_root_bytes}


def guard(auth):
    require(ORIGINAL.is_dir() and WORKSPACE.is_dir() and SOURCE_DIR.is_dir(), 'required accounting root missing')
    original_rows, original_bytes = inventory(ORIGINAL)
    _, workspace_bytes = inventory(WORKSPACE)
    _, comparison_bytes = inventory(COMPARISON)
    _, packet_bytes = inventory(PACKET)
    _, source_bytes = inventory(SOURCE_DIR)
    # V4 protocol and all preparer/tests/logs/historical sources count, even on root.
    external = source_bytes + PLAN_PATH.stat().st_size + MD_PATH.stat().st_size
    external += sum(path.stat().st_size for path in PRESERVED_PROTOCOLS)
    extra_control_paths = {Path(ref['path']) for ref in (auth['authorization_ref'], auth['native_registration_receipt_ref'])}
    for path in extra_control_paths:
        require(not path.is_relative_to(ORIGINAL) and not path.is_relative_to(COMPARISON),
                'new control receipt cannot be in original campaign or comparison allocation')
        if path.is_relative_to(SOURCE_DIR) or path.is_relative_to(PACKET) or path in (PLAN_PATH, MD_PATH) or path in PRESERVED_PROTOCOLS:
            continue
        if path.is_relative_to(WORKSPACE):
            packet_bytes += path.stat().st_size  # already included once in whole workspace
        else:
            external += path.stat().st_size
    native_log = Path(auth['native_synthetic_test_result']['log_ref']['path'])
    require(native_log.is_relative_to(SOURCE_DIR), 'synthetic log must be inside counted source packet')
    result = resource_values(original_bytes, workspace_bytes, comparison_bytes, external, packet_bytes,
                             auth['other_reserved_bytes'], shutil.disk_usage(WORKSPACE).free,
                             shutil.disk_usage(SOURCE_DIR).free)
    result['original_inventory_sha256'] = digest(canonical_bytes(original_rows))
    require(result['original_inventory_sha256'] == auth['original_inventory_sha256'], 'original campaign inventory changed')
    return result


def bounded_receipt_write(path, value, auth):
    require(path.parent == PACKET and not path.exists(), 'receipt must be new and exclusive')
    payload = canonical_bytes(value) + b'\n'
    require(len(payload) <= 2097152, 'readiness receipt exceeds2MiB')
    current = guard(auth)
    require(len(payload) <= current['packet_remaining_bytes'], 'readiness receipt exceeds remaining packet cap')
    temp = path.with_suffix('.json.tmp')
    require(not temp.exists(), 'stale temporary receipt; root review required')
    with temp.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # No overwrite of an old receipt. Single admitted process owns PACKET.
    require(not path.exists(), 'receipt appeared concurrently')
    os.rename(temp, path)
    guard(auth)


class CappedLog(io.TextIOBase):
    """Authenticated internal stdout/stderr sink; every append is cap-checked."""
    def __init__(self, auth):
        self.auth = auth
        self.path = Path(auth['preparation_log']['path'])
        require(self.path.parent == SOURCE_DIR and not self.path.exists(), 'preparation log must be new/exclusive')
        self.stream = self.path.open('xb')
        self.used = 0

    def write(self, text):
        payload = text.encode('utf-8')
        require(self.used + len(payload) <= 65536, 'preparation log cap reached')
        require(len(payload) <= guard(self.auth)['packet_remaining_bytes'], 'log would exceed packet cap')
        self.stream.write(payload); self.stream.flush()
        self.used += len(payload)
        guard(self.auth)
        return len(text)

    def flush(self):
        if hasattr(self, 'stream') and not self.stream.closed:
            self.stream.flush()

    def close(self):
        self.flush()
        if hasattr(self, 'stream'):
            self.stream.close()
        super().close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args(argv)
    actual_argv = [str(Path(__file__).resolve())] + (list(argv) if argv is not None else sys.argv[1:])
    require(Path(args.plan) == PLAN_PATH, 'wrong immutable V4 protocol path')
    plan_ref, plan_bytes = local_ref(args.plan)
    plan = parse_json(plan_bytes); validate_plan(plan)
    driver_ref, _ = local_ref(Path(__file__).resolve())
    test_refs = [local_ref(SOURCE_DIR / 'test_prepare_struct_intensity_v4.py')[0]]
    auth_ref, auth_bytes = local_ref(args.authorization)
    auth = parse_json(auth_bytes)
    validate_authorization(auth, plan, plan_ref, driver_ref, test_refs, actual_argv)
    auth['authorization_ref'] = auth_ref
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1',
            'CUDA must be hidden and bytecode writes disabled')
    read_ref(auth['native_registration_receipt_ref'], 2097152)
    native_log_bytes = read_ref(auth['native_synthetic_test_result']['log_ref'], 2097152)
    validate_native_test_log(native_log_bytes, plan['preparation_v4']['expected_synthetic_tests'])
    read_ref(plan['preparation_v4']['identity_projection_ref'], 2097152)
    for ref in auth['source_refs']:
        read_ref(ref, 2097152)
    before = guard(auth)
    require(not PACKET.exists(), 'packet workspace already exists; no overwrite/resume without new root review')
    PACKET.mkdir(parents=False)
    with CappedLog(auth) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            prepare_authenticated(plan, plan_ref, driver_ref, test_refs, auth, auth_ref, before)


def prepare_authenticated(plan, plan_ref, driver_ref, test_refs, auth, auth_ref, before):
    start = time.monotonic()
    # Only now may native numerical runtimes and explicitly approved bytes enter.
    import numpy as np
    checkpoint_receipts = []
    for arm in ARMS:
        require(time.monotonic() - start <= 900, 'CPU preparation wall ceiling reached')
        guard(auth)
        checkpoint_ref = auth['checkpoint_refs'][arm]
        blob = load_checkpoint_cpu(read_ref(checkpoint_ref))
        checkpoint_receipts.append(validate_checkpoint_metadata(blob, arm, plan, np))
        del blob
    refs = {(r['recording_id'], r['kind']): r for r in auth['input_refs']}
    rows = []
    for row in plan['cohorts']['clip_index']:
        require(time.monotonic() - start <= 900, 'CPU preparation wall ceiling reached')
        guard(auth)
        try:
            bus = read_npz_ref(refs[(row['recording_id'], 'audio')], 'audio', np)
            target = validate_structure(read_npz_ref(refs[(row['recording_id'], 'structure')], 'structure', np), np)
            donor = validate_structure(read_npz_ref(refs[(row['original_shuffled_donor_recording_id'], 'structure')], 'structure', np), np)
            result = qualify_clip(row, bus, target, donor, np)
            del bus, target, donor
        except (PreparationError, KeyError, ValueError, TypeError, OSError, zipfile.BadZipFile) as exc:
            result = {'clip': row['clip'], 'recording_id': row['recording_id'],
                      'donor_recording_id': row['original_shuffled_donor_recording_id'], 'eligible': False,
                      'status': 'MATERIAL_QUALIFICATION_FAILED', 'reasons': [type(exc).__name__ + ': ' + str(exc)],
                      'ratio_or_prediction_computed': False}
        rows.append(result)
    require(len(rows) == 40, 'all40 decisions required')
    after = guard(auth)
    receipt = {'schema_version': VERSION, 'status': 'PREPARATION_COMPLETE_NO_SCIENTIFIC_EVALUATION',
               'plan_ref': plan_ref, 'driver_ref': driver_ref, 'test_refs': test_refs,
               'authorization_ref': auth_ref, 'source_refs': auth['source_refs'],
               'checkpoint_refs': auth['checkpoint_refs'], 'input_refs': auth['input_refs'],
               'checkpoint_metadata': checkpoint_receipts, 'all40_eligibility': rows,
               'eligible_clip_ids': [r['clip'] for r in rows if r['eligible']],
               'original40_inventory_preserved': True, 'replacement_clips': [],
               'science_scores_computed': 0, 'model_constructions': 0, 'model_forwards': 0,
               'GPU_jobs': 0, 'TEST_payload_reads': 0, 'raw_manifest_reads': 0,
               'GPU_or_study_admission_granted': False, 'original_gate_completed': False,
               'eligibility_is_not_an_execution_grant': True,
               'resource_before': before, 'resource_after_before_receipt': after,
               'elapsed_seconds': time.monotonic() - start,
               'limitations': plan['preparation_v4']['descriptive_identity_limitations']}
    print(json.dumps({'status': 'PREPARED_AWAITING_READINESS_PUBLICATION_AND_NATIVE_EXIT0',
                      'receipt': str(PACKET / 'preparation_readiness_v4.json'),
                      'eligible_clips': len(receipt['eligible_clip_ids']), 'model_forwards': 0,
                      'scientific_scores': 0, 'GPU_admission': False}, sort_keys=True))
    receipt['preparation_log_ref'] = local_ref(SOURCE_DIR / 'preparation_v4.log', 65536)[0]
    receipt['requires_parent_verified_native_exit0'] = True
    bounded_receipt_write(PACKET / 'preparation_readiness_v4.json', receipt, auth)
    # No logging after publication; parent still must verify this process exited0.


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('PREPARATION_STOP: ' + type(exc).__name__ + ': ' + str(exc)[:1024], file=sys.stderr)
        raise SystemExit(2)
