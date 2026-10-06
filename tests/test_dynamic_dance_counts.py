"""Count timing audit counterexamples, including real fixed-map emitted bytes."""
from dataclasses import asdict
import json

import numpy as np

from lightsem.conductor.body_transfer import TransferConfig
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.experiments.dynamic_dance.mapping import compile_body
from scripts.motion.audit_dynamic_dance_counts import (
    COUNT_CASES, compare_common, compare_stream, fixture_streams, response_timing,
    synthetic_timing, onset_lag,
)
from scripts.motion.run_body_portfolio import intervention


def test_temporal_shift_is_not_hidden_by_equal_histogram_or_hash_sort():
    time = np.arange(80)/40.
    a = np.zeros((80, 1)); a[20:40] = 1
    b = np.roll(a, 2, axis=0)
    result = compare_stream(a, b, time)
    assert not result['exact']
    assert result['first_mismatch_s'] == .5
    assert result['change_timestamp_discrepancies'] == 4
    assert not result['change_timestamp_sets_exact']


def test_constant_duplicate_streams_do_not_supply_observed_timing():
    result = compare_stream(np.zeros((80, 2)), np.zeros((80, 2)), np.arange(80)/40.)
    assert result['exact']
    assert result['timing_status'] == 'undefined_constant_stream'
    assert result['first_change_left_s'] is None


def test_response_reports_signed_anticipation_and_unsettled_tail():
    time = np.arange(181)/40.
    reference = np.zeros((181, 1)); actual = reference.copy()
    actual[18:83] = 1
    result = response_timing(actual, reference, time, 21, 80, 1e-9)
    assert result['lag_from_source_onset_s'] == time[18]-time[21]
    assert result['no_response_before_source'] is False
    assert result['settled_after_source_return_s'] == time[83]-time[80]
    assert result['final_frozen_tail_constant']
    actual[-20:] = np.arange(20)[:, None]
    failed = response_timing(actual, reference, time, 21, 80, 1e-9)
    assert not failed['final_frozen_tail_constant']
    assert failed['settled_after_source_return_s'] is None


def test_absent_response_has_no_invented_zero_lag_or_settling_evidence():
    reference = np.zeros((181, 1))
    result = response_timing(reference, reference, np.arange(181)/40., 21, 80, 1e-9)
    assert result['status'] == 'undefined_no_resolved_response'
    assert result['lag_from_source_onset_s'] is None
    assert result['no_response_before_source'] is None
    assert result['settled_after_source_return_s'] is None


def test_left_censored_response_retains_anticipation_but_only_bounds_latency():
    time = np.arange(181)/40.
    reference = np.zeros((181, 1)); actual = reference.copy()
    actual[:82] = 1
    result = response_timing(actual, reference, time, 21, 80, 1e-9)
    assert result['first_response_frame'] == 0
    assert result['first_response_s'] == 0.
    assert result['onset_left_censored'] is True
    assert result['lag_from_source_onset_s'] is None
    assert result['lag_from_source_onset_lower_bound_s'] is None
    assert result['lag_from_source_onset_upper_bound_s'] == -.525
    assert result['first_observed_lag_from_source_onset_s'] == -.525
    assert result['no_response_before_source'] is False


def test_left_censored_source_only_bounds_response_latency_from_below():
    time = np.arange(181)/40.
    reference = np.zeros((181, 1)); actual = reference.copy()
    actual[21:82] = 1
    result = response_timing(actual, reference, time, 0, 80, 1e-9)
    assert result['source_onset_left_censored'] is True
    assert result['lag_from_source_onset_s'] is None
    assert result['lag_from_source_onset_lower_bound_s'] == .525
    assert result['lag_from_source_onset_upper_bound_s'] is None
    assert result['no_response_before_source'] is True


def test_requested_to_emitted_censored_pairs_have_correct_bound_direction():
    earlier = onset_lag(.525, 0., response_left_censored=True)
    assert earlier['exact_s'] is None
    assert earlier['lower_bound_s'] is None and earlier['upper_bound_s'] == -.525
    later = onset_lag(0., .525, origin_left_censored=True)
    assert later['exact_s'] is None
    assert later['lower_bound_s'] == .525 and later['upper_bound_s'] is None
    both = onset_lag(0., 0., origin_left_censored=True, response_left_censored=True)
    assert both['exact_s'] is both['lower_bound_s'] is both['upper_bound_s'] is None
    assert both['first_observed_difference_s'] == 0.
    assert both['status'] == 'unbounded_both_left_censored'
    missing = onset_lag(None, 0., response_left_censored=True)
    assert missing['first_observed_difference_s'] is None
    assert missing['status'] == 'undefined_missing_onset'


def test_both_onsets_censored_do_not_establish_zero_lag_or_causal_order():
    time = np.arange(181)/40.
    reference = np.zeros((181, 1)); actual = reference.copy(); actual[:82] = 1
    result = response_timing(actual, reference, time, 0, 80, 1e-9)
    assert result['lag_from_source_onset_s'] is None
    assert result['lag_from_source_onset_lower_bound_s'] is None
    assert result['lag_from_source_onset_upper_bound_s'] is None
    assert result['no_response_before_source'] is None


def test_constant_residual_and_short_return_do_not_prove_settling():
    time = np.arange(181)/40.
    reference = np.zeros((181, 1)); actual = reference.copy()
    actual[21:] = 1
    residual = response_timing(actual, reference, time, 21, 80, 1e-9)
    assert residual['final_frozen_tail_constant']
    assert not residual['final_frozen_tail_matches_reference']
    assert residual['settled_after_source_return_s'] is None
    assert residual['settling_status'] == 'unconfirmed_within_observation'
    actual[160:] = 0
    short = response_timing(actual, reference, time, 21, 80, 1e-9)
    assert short['settled_after_source_return_s'] is None


def test_shared_fixtures_keep_exact_streams_despite_changed_DMX_addresses():
    plan = {'counts': COUNT_CASES['63'], 'transfer': asdict(TransferConfig())}
    world = intervention(REFERENCE, 'left_arm', frames=101, degrees=12.)
    color = np.tile(np.array([10, 20, 30], np.uint8), (len(world), 1))
    cases, patches = {}, {}
    for count in ('1', '63', '117'):
        dmx, patch, intent, _ = compile_body(world, color, color, plan, COUNT_CASES[count])
        cases[count] = fixture_streams(dmx, patch, intent)
        patches[count] = {p['name']: p for p in patch}
    assert patches['63']['laser_001']['address'] != patches['117']['laser_001']['address']
    assert patches['63']['bulb_050']['address'] != patches['117']['bulb_050']['address']
    assert patches['117']['bulb_083']['universe'] == 1
    for a, b, common in [('1', '63', 1), ('1', '117', 1), ('63', '117', 63)]:
        report = compare_common(cases[a], cases[b], np.arange(len(world))/40.)
        assert report['exact_common_streams']
        assert report['common_fixture_count'] == common
        json.dumps(report, allow_nan=False)
    assert 'source_field_activity' in cases['117']['bulb_051']['streams']
    assert 'constrained_emitted_angles_deg' not in cases['117']['bulb_051']['streams']


def test_fixture_identity_change_cannot_pass_on_unchanged_commands():
    a = {'lamp': {'identity': {'role_index': 1}, 'streams': {'commands': np.zeros((10, 1), np.uint8)}}}
    b = {'lamp': {'identity': {'role_index': 2}, 'streams': {'commands': np.zeros((10, 1), np.uint8)}}}
    result = compare_common(a, b, np.arange(10)/40.)
    assert not result['exact_common_streams']


def test_synthetic_added_fixture_audit_uses_actual_body_onset_and_serializes():
    plan = {'counts': COUNT_CASES['63'], 'transfer': asdict(TransferConfig())}
    color = np.array([10, 20, 30], np.uint8)
    report = synthetic_timing(REFERENCE, [color, color], plan)
    assert report['synthetic_only']
    assert len(report['added_fixture_names']) == 54
    assert report['frozen_only_no_autonomous_motion']
    for row in report['regions'].values():
        assert row['source_onset_frame'] == 21
        assert row['source_return_frame'] == 80
        assert len(row['added_outputs']) == 54
        for name, fixture in row['added_outputs'].items():
            if name.startswith('bulb'):
                assert 'constrained_emitted_angles_deg' not in fixture['stages']
    json.dumps(report, allow_nan=False)
