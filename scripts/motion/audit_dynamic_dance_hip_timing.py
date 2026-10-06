"""Supplement unresolved hip-role stimulus coverage with a rigid-yaw CPU test.

The source is a synthetic perturbation of each accepted middle pose, never a
demo dance. Rotating the entire skeleton also excites other fixtures; it does
not establish isolated hip control or add independent choreography information.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.motion import audit_dynamic_dance_counts as timing
from lightsem.experiments.dynamic_dance import mapping
from lightsem.experiments.dynamic_dance.campaign import ROOT, array_sha, read, sha, verify_files, write_new

PROTOCOL_SHA256 = '0188b8b1f0ffc540fe7347b06787462a28106ae44bdf882391a8dca610be9cde'
TIMING_HELPER_SHA256 = 'ec57f1f569ea246c9ea14ba028cac85c707f47e8c9f3ab5dfaa597e14c407300'
DEPENDENCY_MANIFEST_SHA256 = '84a3901188acd15cd8b45a79ec99f859b301ce07e708d237025d8e95a2a74299'
TARGETS = {'beam_010': 1, 'laser_007': 2}
DEGREES = 12.
FPS = 40.
PULSE_FRAMES = 101
TAIL_FRAMES = 80


def yaw_pulse(pose):
    """Exact existing smoothstep pulse; only spatial coordinate rotation uses trig."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (22, 3) or not np.isfinite(pose).all():
        raise ValueError('Expected one finite accepted 22-joint middle pose')
    envelope = np.zeros(PULSE_FRAMES+TAIL_FRAMES)
    u = np.linspace(0., 1., 21)
    smooth = u*u*(3.-2.*u)
    envelope[20:41] = smooth
    envelope[40:61] = 1.
    envelope[60:81] = smooth[::-1]
    angle = np.deg2rad(DEGREES)*envelope
    vector = pose-pose[0]
    result = np.broadcast_to(pose, (len(envelope), 22, 3)).copy()
    result[..., 0] = pose[0, 0]+np.cos(angle)[:, None]*vector[:, 0]+np.sin(angle)[:, None]*vector[:, 2]
    result[..., 2] = pose[0, 2]-np.sin(angle)[:, None]*vector[:, 0]+np.cos(angle)[:, None]*vector[:, 2]
    # Y and root are unchanged exactly, including floating arithmetic endpoints.
    result[..., 1] = pose[:, 1]
    result[:, 0] = pose[0]
    result[envelope == 0] = pose
    return result, envelope


def audit_pose(pose, colors, plan):
    """Measure both hip targets and disclose responses of the complete fixed rig."""
    from lightsem.conductor.body_transfer_v2.skeleton import PARENTS
    if plan['transfer']['fps'] != FPS:
        raise ValueError('Declared source clock must match the unchanged compiler clock')
    world, envelope = yaw_pulse(pose)
    times = np.arange(len(world))/FPS
    reference = np.broadcast_to(np.asarray(pose, float), world.shape).copy()
    frozen_colors = [np.repeat(np.asarray(c, dtype=np.uint8)[None], len(world), axis=0) for c in colors]
    frozen, frozen_hard = timing._compile(reference, frozen_colors, plan, timing.COUNT_CASES['117'])
    actual, actual_hard = timing._compile(world, frozen_colors, plan, timing.COUNT_CASES['117'])
    if set(actual) != set(frozen) or len(actual) != 117:
        raise ValueError('The fixed 117-fixture patch changed')
    for name, joint in TARGETS.items():
        identity = actual[name]['identity']
        if identity['role_index'] != joint or identity['role_blend'] != 0:
            raise ValueError('Target is no longer the registered pure hip role: '+name)
    body_change = timing._frame_magnitude(world-reference) > 1e-9
    onset = timing._first(body_change)
    source_return = int(np.flatnonzero(body_change)[-1]+1) if body_change.any() else 0
    stages_by_fixture, frozen_exact, identity_exact = {}, {}, {}
    for name, row in actual.items():
        identity_exact[name] = row['identity'] == frozen[name]['identity']
        frozen_command = frozen[name]['streams']['commands']
        frozen_exact[name] = bool(np.all(frozen_command == frozen_command[:1]))
        stages = {}
        for key, values in row['streams'].items():
            if key in ('commands', 'emitted_RGB'):
                continue
            threshold = 1e-9 if key.endswith(('_m', '_deg')) else 1e-12
            stages[key] = timing.response_timing(values, frozen[name]['streams'][key], times,
                                                  onset, source_return, threshold)
        for requested, emitted in (('requested_angles_deg', 'constrained_emitted_angles_deg'),
                                    ('requested_dimmer', 'emitted_dimmer')):
            if requested in stages:
                a, b = stages[requested], stages[emitted]
                lag = timing.onset_lag(a['first_response_s'], b['first_response_s'],
                    origin_left_censored=a['onset_left_censored'], response_left_censored=b['onset_left_censored'])
                b.update(timing._lag_fields('lag_from_requested_onset', lag))
        stages_by_fixture[name] = {'identity': row['identity'], 'stages': stages,
            'frozen_commands_sha256': array_sha(frozen_command),
            'emitted_commands_sha256': array_sha(row['streams']['commands'])}
    targets = {}
    for name, joint in TARGETS.items():
        row = stages_by_fixture[name]
        direct = timing.response_timing(world[:, joint], reference[:, joint], times, onset, source_return, 1e-9)
        row['direct_hip_position_m'] = direct
        motion = row['stages']['constrained_emitted_angles_deg']
        required = [direct, row['stages']['source_point_m'], row['stages']['requested_angles_deg'], motion]
        missing = [label for label, value in zip(('hip_position', 'source_readout', 'requested_motion', 'emitted_motion'), required)
                   if value['status'] != 'observed']
        early = [stage for stage, value in row['stages'].items() if value['no_response_before_source'] is False]
        unsettled = [stage for stage in ('constrained_emitted_angles_deg', 'emitted_dimmer')
                     if row['stages'][stage]['status'] == 'observed' and
                     row['stages'][stage]['settling_status'] != 'sustained_return_to_frozen_reference']
        row['coverage_status'] = ('measured_with_timing_failure' if early or unsettled else
            'unqualified_unresolved_response' if missing else 'observed_and_settled')
        row['responses_before_source'] = early
        row['unsettled_emitted_stages'] = unsettled
        row['unresolved_stages'] = missing
        targets[name] = row
    responded = {stage: sorted(name for name, row in stages_by_fixture.items()
                 if stage in row['stages'] and row['stages'][stage]['status'] == 'observed')
                 for stage in ('source_point_m', 'requested_angles_deg', 'constrained_emitted_angles_deg', 'emitted_dimmer')}
    lengths = np.linalg.norm(world[:, 1:]-world[:, np.asarray(PARENTS)[1:]], axis=-1)
    reference_lengths = np.linalg.norm(reference[:, 1:]-reference[:, np.asarray(PARENTS)[1:]], axis=-1)
    report = {'synthetic_only': True, 'frames': len(world), 'fps': FPS, 'time_sha256': array_sha(times),
        'source_onset_frame': onset, 'source_onset_s': None if onset is None else float(times[onset]),
        'source_onset_left_censored': None if onset is None else onset == 0,
        'source_return_frame': source_return, 'source_return_s': float(times[source_return]),
        'source_pose_sha256': array_sha(np.asarray(pose)), 'synthetic_body_sha256': array_sha(world),
        'envelope_sha256': array_sha(envelope), 'maximum_yaw_degrees': DEGREES,
        'root_exact': bool(np.array_equal(world[:, 0], reference[:, 0])),
        'vertical_coordinates_exact': bool(np.array_equal(world[..., 1], reference[..., 1])),
        'unperturbed_frames_exact': bool(np.array_equal(world[envelope == 0], reference[envelope == 0])),
        'maximum_bone_length_roundoff_m': float(np.abs(lengths-reference_lengths).max()),
        'changed_body_joints': np.flatnonzero(np.any(np.linalg.norm(world-reference, axis=-1) > 1e-9, axis=0)).tolist(),
        'target_fixtures': targets, 'all_fixture_timings': stages_by_fixture,
        'broader_body_response': responded, 'fixture_identities_exact': all(identity_exact.values()),
        'frozen_only_commands_exact': frozen_exact, 'frozen_only_no_autonomous_motion': all(frozen_exact.values()),
        'hard_checks': {'frozen': frozen_hard, 'perturbed': actual_hard},
        'source_clock': '181 samples at original 40 Hz; exact existing101-frame pulse plus80-frame frozen tail',
        'scope': 'Synthetic whole-skeleton yaw, including the hips and other joints. Broader output responses are disclosed; no isolated hip-control, increased information or demo-motion claim.'}
    return report


def _verified_dependencies(root):
    manifest_path = root/'records/verification_v1_manifest.json'
    if sha(manifest_path) != DEPENDENCY_MANIFEST_SHA256:
        raise ValueError('Frozen verification dependency manifest changed')
    if sha(timing.__file__) != TIMING_HELPER_SHA256:
        raise ValueError('Import the corrected count_audit_v2 timing helper')
    manifest = read(manifest_path)
    frozen_root = Path(manifest['source_root'])
    project_root = Path(mapping.__file__).resolve().parents[3]
    from lightsem.experiments.motion_study import geometry, offline_angles
    from lightsem.experiments.beat_conditioned import beat_conditioned_render
    from lightsem.experiments.dynamic_dance import contracts, campaign
    from scripts.motion import run_body_portfolio
    paths = set(mapping.implementation_pins())
    paths.update(m.__file__ for m in (geometry, offline_angles, beat_conditioned_render, contracts, campaign, run_body_portfolio))
    inputs = {str(manifest_path): DEPENDENCY_MANIFEST_SHA256, str(Path(timing.__file__)): TIMING_HELPER_SHA256,
              str(Path(__file__)): sha(__file__)}
    for path in paths:
        relative = Path(path).resolve().relative_to(project_root)
        frozen = frozen_root/relative
        expected = manifest['files'].get(str(frozen))
        if expected is None or sha(path) != expected or sha(frozen) != expected:
            raise ValueError('Runtime source differs from verification_v1: '+str(relative))
        inputs[str(Path(path))] = expected
        inputs[str(frozen)] = expected
    return inputs


def _bind_prior_inputs(prior, expected):
    """The addendum changes stimulus coverage, not PLAN, limits, P2 or source files."""
    for path, expected_sha in expected.items():
        if prior.get('inputs', {}).get(path) != expected_sha:
            raise ValueError('Supplemental baseline differs from prior count audit: '+path)


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists() or Path(str(output)+'.sha256.json').exists():
        raise FileExistsError('Preserve existing supplemental output: '+str(output))
    protocol_path = root/'records/HIP_TIMING_ADDENDUM_PROTOCOL.json'
    if sha(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('Frozen hip-coverage protocol changed')
    protocol = read(protocol_path)
    expected = {'amplitude_degrees': DEGREES, 'fps': FPS, 'pulse_frames': PULSE_FRAMES,
                'frozen_tail_frames': TAIL_FRAMES, 'settling_window_frames': timing.SETTLING_WINDOW_FRAMES,
                'counts': timing.COUNT_CASES['117']}
    if any(protocol.get(k) != value for k, value in expected.items()):
        raise ValueError('Diagnostic parameters differ from frozen protocol')
    if {k: v['primary_joint'] for k, v in protocol['target_fixtures'].items()} != TARGETS:
        raise ValueError('Target fixtures differ from frozen protocol')
    plan_path = root/'PLAN.json'; plan = read(plan_path)
    if plan['mapper_id'] != '01' or plan['fps'] != FPS or plan['body_fps'] != 30:
        raise ValueError('Original mapper or clock changed')
    verify_files(plan['files'])
    prior_path = Path(protocol['prior_report']); prior_seal = Path(str(prior_path)+'.sha256.json')
    if read(prior_seal).get('sha256') != sha(prior_path):
        raise ValueError('Prior count audit changed')
    prior_report = read(prior_path)
    _bind_prior_inputs(prior_report, {str(plan_path): sha(plan_path), **plan['files']})
    prior = {row['song']: row for row in prior_report['tracks']}
    inputs = {**_verified_dependencies(root), **plan['files'], str(plan_path): sha(plan_path),
              str(protocol_path): PROTOCOL_SHA256, str(prior_path): sha(prior_path), str(prior_seal): sha(prior_seal)}
    tracks = {row['id']: row for row in plan['tracks']}
    if set(tracks) != set(protocol['songs']):
        raise ValueError('Both registered songs are required')
    rows = []
    for song in protocol['songs']:
        track = tracks[song]; body = mapping.baseline_body(track)
        if not np.array_equal(body.time, np.arange(5400)/30.):
            raise ValueError('Accepted source clock changed')
        if prior[song]['source_body_sha256'] != sha(track['body']):
            raise ValueError('Supplement must use the prior audit accepted body')
        colors = mapping.p2_colors(track)
        print('AUDIT_HIP_TIMING', song, flush=True)
        row = audit_pose(body.world[len(body.world)//2], [c[len(c)//2] for c in colors], plan)
        row.update(song=song, accepted_source_body_sha256=sha(track['body']), source_pose_time_s=90.)
        rows.append(row)
    verify_files(inputs)
    statuses = [v['coverage_status'] for row in rows for v in row['target_fixtures'].values()]
    failure = any(status == 'measured_with_timing_failure' for status in statuses) or any(
        not row['frozen_only_no_autonomous_motion'] or not row['fixture_identities_exact'] for row in rows)
    report = {'schema_version': 1, 'status': 'measured_with_failed_checks' if failure else
              'unqualified_missing_response' if any(s != 'observed_and_settled' for s in statuses) else 'measured',
              'inputs': inputs, 'protocol': protocol, 'tracks': rows, 'GPU_used': False,
              'prior_count_evidence_unchanged': True,
              'scope': 'Additional stimulus coverage after two pure hip roles received no changed source in the six original interventions. No original gates, reports, mapper, models or demo motion changed.'}
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
    return 0 if report['status'] == 'measured' else 2


if __name__ == '__main__':
    raise SystemExit(main())
