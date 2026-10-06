"""CPU evidence for source timing when a fixed body drives 1, 63 or 117 outputs.

This audit never generates demo motion. Real-song comparisons use the accepted
180-second bodies; synthetic reaches are the existing regional intervention
with an appended frozen tail. Offline actuator anticipation is reported as
anticipation, separately from autonomous motion on an entirely frozen body.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.experiments.dynamic_dance.campaign import ROOT, array_sha, read, sha, verify_files, write_new
from lightsem.experiments.dynamic_dance.mapping import baseline_body, compile_body, implementation_pins, lighting_body, p2_colors

COUNT_CASES = {
    '1': {'beams': 1, 'lasers': 0, 'bulbs': 0},
    '63': {'beams': 8, 'lasers': 5, 'bulbs': 50},
    '117': {'beams': 23, 'lasers': 11, 'bulbs': 83},
}
SYNTHETIC_TAIL_FRAMES = 80
SETTLING_WINDOW_FRAMES = 40
PROTOCOL_SHA256 = 'b6180d599cabbed62a8261cc5c4c5664effd1bee37795395567acc4d1f033e29'


def _frame_magnitude(value):
    value = np.asarray(value, dtype=float)
    return np.max(np.abs(value).reshape(len(value), -1), axis=1)


def _first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def compare_stream(left, right, times):
    """Exact same-time comparison: shifting a stream cannot rescue a mismatch."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.shape[0] != len(times):
        raise ValueError('Compared stream dimensions/timestamps differ')
    mismatch = np.any((a != b).reshape(len(a), -1), axis=1)
    changed_a = np.any((a[1:] != a[:-1]).reshape(len(a)-1, -1), axis=1)
    changed_b = np.any((b[1:] != b[:-1]).reshape(len(b)-1, -1), axis=1)
    first = _first(mismatch)
    first_change = _first(changed_a)
    return {
        'exact': bool(not mismatch.any()), 'mismatched_frames': int(mismatch.sum()),
        'first_mismatch_s': None if first is None else float(times[first]),
        'maximum_absolute_difference': float(np.max(np.abs(a.astype(float)-b.astype(float)))),
        'change_timestamp_sets_exact': bool(np.array_equal(changed_a, changed_b)),
        'changed_intervals_left': int(changed_a.sum()), 'changed_intervals_right': int(changed_b.sum()),
        'change_timestamp_discrepancies': int(np.count_nonzero(changed_a != changed_b)),
        'first_change_left_s': None if first_change is None else float(times[first_change+1]),
        'timing_status': 'observed' if changed_a.any() or changed_b.any() else 'undefined_constant_stream',
        'left_sha256': array_sha(a), 'right_sha256': array_sha(b),
    }


def fixture_streams(dmx, patch, intent):
    """Use fixture identities, never universe/address positions, for comparisons."""
    controls = decode(dmx, patch)
    result = {}
    for i, fixture in enumerate(patch):
        start = fixture['address']-1
        streams = {
            'commands': dmx[:, fixture['universe'], start:start+fixture['n_channels']],
            'requested_targets_m': intent['targets'][:, i],
            'requested_dimmer': intent['dimmer'][:, i],
            'emitted_dimmer': controls['dimmer'][:, i],
            'emitted_RGB': controls['rgb'][:, i],
        }
        if fixture['kind'] == 'bulb':
            # Bulbs read the anatomical field. Their source_points are zero
            # placeholders and would falsely suggest that they have no source.
            streams['source_field_m'] = intent['field_points']
            streams['source_field_activity'] = intent['field_activity']
            streams['source_field_centers'] = intent['bulb_field_centers']
        else:
            streams['source_point_m'] = intent['source_points'][:, i]
            vector = intent['targets'][:, i]-np.asarray(fixture['position'])
            norm = np.linalg.norm(vector, axis=1)
            streams['requested_angles_deg'] = np.column_stack((
                np.rad2deg(np.arctan2(vector[:, 0], -vector[:, 1])),
                np.rad2deg(np.arcsin(np.clip(vector[:, 2]/norm, -1, 1)))))
            streams['constrained_emitted_angles_deg'] = controls['angles_deg'][:, i]
        identity = {key: value for key, value in fixture.items()
                    if key not in ('universe', 'address')}
        result[fixture['name']] = {'identity': identity, 'streams': streams}
    return result


def compare_common(left, right, times):
    rows = {}
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if set(a['streams']) != set(b['streams']):
            raise ValueError('Fixture capability streams changed: '+name)
        stages = {key: compare_stream(a['streams'][key], b['streams'][key], times)
                  for key in a['streams']}
        rows[name] = {'identity_exact': a['identity'] == b['identity'], 'stages': stages}
    return {'common_fixture_count': len(rows), 'fixtures': rows,
            'exact_common_streams': bool(rows and all(row['identity_exact'] and
                all(stage['exact'] for stage in row['stages'].values()) for row in rows.values())),
            'scope': 'Exact original timestamps and fixture-local command bytes; DMX addresses may change.'}


def onset_lag(origin_s, response_s, *, origin_left_censored=False, response_left_censored=False):
    """Sample-resolution onset lag; a frame-zero onset is known only to be <=t0.

    Null bounds represent an unbounded side, not a zero-latency observation.
    When either response is absent the entire lag is undefined.
    """
    result = {'status': 'undefined_missing_onset', 'exact_s': None,
              'lower_bound_s': None, 'upper_bound_s': None,
              'first_observed_difference_s': None}
    if origin_s is None or response_s is None:
        return result
    difference = float(response_s-origin_s)
    result['first_observed_difference_s'] = difference
    if origin_left_censored and response_left_censored:
        result['status'] = 'unbounded_both_left_censored'
    elif response_left_censored:
        result.update(status='upper_bound_only', upper_bound_s=difference)
    elif origin_left_censored:
        result.update(status='lower_bound_only', lower_bound_s=difference)
    else:
        result.update(status='observed_sample_resolution', exact_s=difference,
                      lower_bound_s=difference, upper_bound_s=difference)
    return result


def _lag_fields(prefix, lag):
    return {prefix+'_s': lag['exact_s'], prefix+'_lower_bound_s': lag['lower_bound_s'],
            prefix+'_upper_bound_s': lag['upper_bound_s'], prefix+'_status': lag['status'],
            'first_observed_'+prefix+'_s': lag['first_observed_difference_s']}


def response_timing(actual, reference, times, source_onset, source_return, threshold):
    """Keep absent/quantization-hidden responses undefined, including their lag."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    magnitude = _frame_magnitude(actual.astype(float)-reference.astype(float))
    active = magnitude > threshold
    first = _first(active)
    last = int(np.flatnonzero(active)[-1]) if first is not None else None
    changes = _frame_magnitude(np.diff(actual.astype(float), axis=0)) > threshold
    last_change = int(np.flatnonzero(changes)[-1]+1) if changes.any() else None
    window = min(SETTLING_WINDOW_FRAMES, len(actual))
    stable_tail = bool(np.all(_frame_magnitude(actual[-window:].astype(float)-actual[-window].astype(float)) <= threshold))
    returned = max(source_return, 0 if last is None else last+1)
    sustained_return = returned < len(actual) and len(actual)-returned >= SETTLING_WINDOW_FRAMES
    settled = returned if first is not None and sustained_return else None
    source_time = None if source_onset is None else float(times[source_onset])
    response_time = None if first is None else float(times[first])
    lag = onset_lag(source_time, response_time, origin_left_censored=source_onset == 0,
                    response_left_censored=first == 0)
    no_anticipation = None
    if lag['upper_bound_s'] is not None and lag['upper_bound_s'] < 0:
        no_anticipation = False
    elif lag['lower_bound_s'] is not None and lag['lower_bound_s'] >= 0:
        no_anticipation = True
    return {
        'status': 'observed' if first is not None else 'undefined_no_resolved_response',
        'threshold': threshold, 'response_frames': int(active.sum()),
        'first_response_frame': first, 'first_response_s': response_time,
        'onset_left_censored': None if first is None else first == 0,
        'source_onset_left_censored': None if source_onset is None else source_onset == 0,
        'last_response_s': None if last is None else float(times[last]),
        'peak_difference': float(magnitude.max()),
        'peak_difference_s': None if first is None else float(times[int(np.argmax(magnitude))]),
        **_lag_fields('lag_from_source_onset', lag),
        'no_response_before_source': no_anticipation,
        'final_frozen_tail_constant': stable_tail,
        'final_frozen_tail_matches_reference': bool(np.all(magnitude[-window:] <= threshold)),
        'last_output_change_s': None if last_change is None else float(times[last_change]),
        'settled_after_source_return_s': None if first is None or settled is None else float(times[settled]-times[source_return]),
        'settling_status': ('undefined_no_resolved_response' if first is None else
                            'sustained_return_to_frozen_reference' if settled is not None else
                            'unconfirmed_within_observation'),
    }


def _compile(world, colors, plan, counts):
    dmx, patch, intent, hard = compile_body(world, *colors, plan, counts)
    return fixture_streams(dmx, patch, intent), hard


def synthetic_timing(pose, colors, plan):
    from scripts.motion.run_body_portfolio import REGIONS, intervention
    frames = 101+SYNTHETIC_TAIL_FRAMES
    times = np.arange(frames)/40.
    stationary = np.repeat(pose[None], frames, axis=0)
    frozen_colors = [np.repeat(value[None], frames, axis=0) for value in colors]
    frozen, hard = _compile(stationary, frozen_colors, plan, COUNT_CASES['117'])
    static_exact = {name: bool(np.all(row['streams']['commands'] == row['streams']['commands'][:1]))
                    for name, row in frozen.items()}
    existing = {f'beam_{i:03d}' for i in range(1, 9)} | {f'laser_{i:03d}' for i in range(1, 6)} | {f'bulb_{i:03d}' for i in range(1, 51)}
    added = sorted(set(frozen)-existing)
    regions = {}
    for region in REGIONS:
        original = intervention(pose, region, frames=101, degrees=12.)
        world = np.concatenate((original, stationary[:SYNTHETIC_TAIL_FRAMES]), axis=0)
        body_mask = _frame_magnitude(world-stationary) > 1e-9
        onset = _first(body_mask)
        source_return = int(np.flatnonzero(body_mask)[-1]+1) if body_mask.any() else 0
        current, current_hard = _compile(world, frozen_colors, plan, COUNT_CASES['117'])
        outputs = {}
        for name in added:
            stages = {}
            for key, values in current[name]['streams'].items():
                if key in ('commands', 'emitted_RGB'):
                    continue
                threshold = 1e-9 if key.endswith(('_m', '_deg')) else 1e-12
                stages[key] = response_timing(values, frozen[name]['streams'][key], times,
                                               onset, source_return, threshold)
            for requested, emitted in (('requested_angles_deg', 'constrained_emitted_angles_deg'),
                                        ('requested_dimmer', 'emitted_dimmer')):
                if requested in stages:
                    a, b = stages[requested]['first_response_s'], stages[emitted]['first_response_s']
                    lag = onset_lag(a, b,
                        origin_left_censored=stages[requested]['onset_left_censored'],
                        response_left_censored=stages[emitted]['onset_left_censored'])
                    stages[emitted].update(_lag_fields('lag_from_requested_onset', lag))
            outputs[name] = {'identity': current[name]['identity'], 'stages': stages}
        regions[region] = {'source_onset_frame': onset,
                           'source_onset_s': None if onset is None else float(times[onset]),
                           'source_onset_left_censored': None if onset is None else onset == 0,
                           'source_return_frame': source_return, 'source_return_s': float(times[source_return]),
                           'body_sha256': array_sha(world), 'hard_checks': current_hard, 'added_outputs': outputs}
    early, unsettled, observed = [], [], set()
    for region, row in regions.items():
        for name, fixture in row['added_outputs'].items():
            for stage, value in fixture['stages'].items():
                if value['no_response_before_source'] is False:
                    early.append({'region': region, 'fixture': name, 'stage': stage,
                                  'lag_s': value['lag_from_source_onset_s'],
                                  'lag_lower_bound_s': value['lag_from_source_onset_lower_bound_s'],
                                  'lag_upper_bound_s': value['lag_from_source_onset_upper_bound_s'],
                                  'first_observed_lag_s': value['first_observed_lag_from_source_onset_s'],
                                  'onset_left_censored': value['onset_left_censored']})
                if stage in ('constrained_emitted_angles_deg', 'emitted_dimmer'):
                    if value['status'] == 'observed':
                        observed.add(name)
                    if value['status'] == 'observed' and value['settling_status'] != 'sustained_return_to_frozen_reference':
                        unsettled.append({'region': region, 'fixture': name, 'stage': stage})
    return {'synthetic_only': True, 'frames': frames, 'fps': 40., 'regions': regions,
            'frozen_only_commands_exact': static_exact, 'frozen_only_hard_checks': hard,
            'frozen_only_no_autonomous_motion': all(static_exact.values()),
            'added_fixture_names': added, 'added_fixtures_with_observed_emitted_timing': sorted(observed),
            'added_fixtures_without_observed_emitted_timing': sorted(set(added)-observed),
            'responses_before_source': early, 'unsettled_final_tails': unsettled,
            'no_anticipation_status': 'fail' if early else 'pass' if observed else 'unqualified',
            'scope': 'Existing 12-degree regional test; 80 appended frozen frames, at least 40-frame sustained return to the compiled frozen reference. Signed offline projection anticipation is not relabeled autonomous motion. No demo trajectory or independent dance information is created.'}


def audit_track(track, plan):
    body = baseline_body(track)
    if not np.array_equal(body.time, np.arange(5400)/30.):
        raise ValueError('Accepted source timestamps must be the complete 180-second 30 Hz prefix')
    times, world = lighting_body(body.world)
    colors = p2_colors(track)
    cases, hard = {}, {}
    for key, counts in COUNT_CASES.items():
        print('AUDIT_COUNTS', track['id'], key, flush=True)
        cases[key], hard[key] = _compile(world, colors, plan, counts)
    comparisons = {a+'_to_'+b: compare_common(cases[a], cases[b], times)
                   for a, b in itertools.combinations(COUNT_CASES, 2)}
    del cases
    synthetic = synthetic_timing(body.world[len(body.world)//2], [c[len(c)//2] for c in colors], plan)
    return {'song': track['id'], 'source_body_sha256': sha(track['body']),
            'actual_40hz_body_sha256': array_sha(world), 'frames': len(world), 'duration_s': 180.,
            'time_sha256': array_sha(times), 'hard_checks': hard, 'common_fixture_comparisons': comparisons,
            'exact_common_streams': all(row['exact_common_streams'] for row in comparisons.values()),
            'synthetic_added_output_timing': synthetic,
            'constrained_signal_scope': 'Compiler does not export prequantization projected angles; constrained motion is inspected after decoding actual emitted bytes.'}


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists() or Path(str(output)+'.sha256.json').exists():
        raise FileExistsError('Preserve existing audit output: '+str(output))
    plan_path = root/'PLAN.json'
    plan = read(plan_path)
    protocol_path = root/'records/SUPPLEMENTAL_DELIVERY_AUDITS_PROTOCOL.json'
    if sha(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Supplemental delivery audit protocol changed')
    protocol = read(protocol_path)
    if protocol['count_timing']['rigs'] != list(COUNT_CASES.values()):
        raise ValueError('Count rigs differ from frozen supplemental protocol')
    if plan['mapper_id'] != '01' or plan['counts'] != COUNT_CASES['63'] or plan['body_fps'] != 30 or plan['fps'] != 40:
        raise ValueError('Audit requires registered mapper01, native30/lighting40 and 8/5/50 baseline')
    verify_files(plan['files'])
    from lightsem.experiments.motion_study import offline_angles
    from lightsem.experiments.beat_conditioned import beat_conditioned_render
    from scripts.motion import run_body_portfolio
    inputs = {str(plan_path): sha(plan_path), str(protocol_path): PROTOCOL_SHA256, **plan['files'], **implementation_pins(),
              **{str(Path(p)): sha(p) for p in (__file__, offline_angles.__file__, beat_conditioned_render.__file__, run_body_portfolio.__file__)}}
    reports = [audit_track(track, plan) for track in plan['tracks']]
    verify_files(inputs)
    invariant = all(row['exact_common_streams'] for row in reports)
    static = all(row['synthetic_added_output_timing']['frozen_only_no_autonomous_motion'] for row in reports)
    anticipation = [row['synthetic_added_output_timing']['no_anticipation_status'] for row in reports]
    unsettled = sum(len(row['synthetic_added_output_timing']['unsettled_final_tails']) for row in reports)
    report = {'schema_version': 1, 'status': 'measured', 'inputs': inputs, 'count_cases': COUNT_CASES,
              'timing_reporting_version': 2,
              'reporting_correction': 'Frame-zero onsets are left-censored; retain first observations and report latency bounds. Criteria and outcomes are not changed.',
              'tracks': reports, 'exact_common_streams': invariant, 'frozen_only_no_autonomous_motion': static,
              'no_anticipation_status': 'fail' if 'fail' in anticipation else 'unqualified' if 'unqualified' in anticipation else 'pass',
              'unsettled_added_output_tails': unsettled, 'GPU_used': False,
              'development_scope': protocol['development_scope'],
              'scope': 'Engineering timing evidence only. Unresolved output timing remains undefined; output duplication adds no independent dance information. No musical or artistic qualification.'}
    if not invariant or not static or unsettled or report['no_anticipation_status'] == 'fail':
        report['status'] = 'measured_with_failed_invariants'
    write_new(output, report)
    write_new(str(output)+'.sha256.json', {'sha256': sha(output)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.root, args.output)
    print('COMPLETE', report['status'], str(args.output), flush=True)
    # Diagnostics remain saved even when fixed timing expectations fail.
    return 0 if report['status'] == 'measured' else 2


if __name__ == '__main__':
    raise SystemExit(main())
