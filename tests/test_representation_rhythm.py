"""Synthetic contract controls; no actual-song scoring or GPU use."""
import json

import numpy as np
import pytest

from lightsem.audio.representation_rhythm import (
    TASKS, apply_timing_bridge, load_ensemble, pick_events, policy_metadata, prepare_native_input,
    score_ensemble, sha256, threshold_lookup, timing_indices,
)


def normalizers():
    return {"mean": np.linspace(.123456789, 3.123456789, 92), "std": np.linspace(.5, 2., 92)}


def inputs(times):
    return np.arange(len(times) * 92, dtype=np.float32).reshape(len(times), 92) / 97


def write_heads(tmp_path, thresholds=None):
    rows = []
    for fold in range(5):
        values = np.asarray([1 + fold, 2 + fold, 3 + fold], dtype=float) if thresholds is None else np.asarray(thresholds)
        path = tmp_path / f"example_fold{fold}_ridge.npz"
        np.savez(path, mean=np.zeros(3), scale=np.ones(3), weights=np.diag(values),
                 intercept=np.zeros(3), thresholds=values)
        rows.append({"method": "example", "outer_fold": fold, "inner_validation_fold": (fold + 1) % 5,
                     "alpha": .001, "checkpoint": str(path), "sha256": sha256(path),
                     "thresholds": dict(zip(TASKS, values.tolist()))})
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"fit_records": rows}))
    return report


@pytest.mark.parametrize("beats", [[], [1.]])
def test_no_or_one_beat_is_neutral_without_clock(beats):
    times = np.arange(121) / 40
    raw, norms = inputs(times), normalizers()
    result = apply_timing_bridge(raw, times, beats, norms)
    assert not result["coverage"].any()
    assert np.array_equal(result["base92"][:, :87], raw[:, :87])
    assert np.all(result["normalized92"][:, timing_indices()] == 0)
    assert np.all(result["interval_index"] == -1)
    np.testing.assert_array_equal(result["base92"][:, 87:], np.broadcast_to(norms["mean"][87:].astype(np.float32), (len(times), 5)))


def test_known_phase_coverage_and_original_bar_contract():
    times = np.array([.5, 1., 1.25, 1.5, 2., 2.5, 3., 4.])
    result = apply_timing_bridge(inputs(times), times, [1., 2., 3.], normalizers())
    np.testing.assert_array_equal(result["coverage"], [False, True, True, True, True, True, False, False])
    fields = result["base92"]
    np.testing.assert_allclose(fields[2, 87:], [1., 0., .75 ** 2.5, .25 / 4, 1.], atol=1e-7)
    np.testing.assert_allclose(fields[5, 87:], [0., -1., .5 ** 2.5, 1.5 / 4, 1.], atol=1e-7)
    assert np.all(result["normalized92"][~result["coverage"], 87:] == 0)
    assert "diagnostic events only" in result["metadata"]["downbeat_use"]


def test_missing_interior_detection_is_one_long_real_interval():
    times = np.arange(0., 7., .25)
    result = apply_timing_bridge(inputs(times), times, [1., 2., 5., 6.], normalizers())
    np.testing.assert_array_equal(result["detected_beats"], [1., 2., 5., 6.])
    np.testing.assert_array_equal(result["real_intervals_s"], [1., 3., 1.])
    at_middle = np.flatnonzero(times == 3.5)[0]
    assert result["interval_index"][at_middle] == 1
    np.testing.assert_allclose(result["base92"][at_middle, 87:], [0., -1., .5 ** 2.5, 1.5 / 4, 3.], atol=1e-7)
    assert result["base92"][np.flatnonzero(times == 3.)[0], 91] == pytest.approx(2.5)


def test_acoustics_and_normalization_match_unchanged_baseline():
    times = np.arange(0., 3., 1 / 30)
    raw, norms = inputs(times), normalizers()
    before = raw.copy()
    result = apply_timing_bridge(raw, times, [.5, 1., 2.], norms)
    expected = ((raw - norms["mean"]) / norms["std"]).astype(np.float32)
    np.testing.assert_array_equal(raw, before)
    assert result["base92"].dtype == raw.dtype
    assert result["base92"][:, :87].tobytes() == before[:, :87].tobytes()
    np.testing.assert_array_equal(result["normalized92"][:, :87], expected[:, :87])


@pytest.mark.parametrize("beats", [[1., 1.], [2., 1.], [1., np.nan], [1., np.inf]])
def test_invalid_beats_rejected(beats):
    times = np.arange(80) / 40
    with pytest.raises(ValueError):
        apply_timing_bridge(inputs(times), times, beats, normalizers())


@pytest.mark.parametrize("times", [[0., 0.], [.1, 0.], [0., np.nan]])
def test_invalid_timestamp_order_rejected(times):
    with pytest.raises(ValueError):
        apply_timing_bridge(inputs(times), times, [], normalizers())


def test_model_input_nonfinite_and_bad_normalizer_rejected():
    times = np.arange(10) / 40
    raw = inputs(times)
    raw[0, 4] = np.nan
    with pytest.raises(ValueError):
        apply_timing_bridge(raw, times, [], normalizers())
    norms = normalizers()
    norms["std"][3] = 0
    with pytest.raises(ValueError):
        apply_timing_bridge(inputs(times), times, [], norms)


def test_authenticated_five_head_policy_and_per_head_outputs(tmp_path):
    report = write_heads(tmp_path)
    heads = load_ensemble(report, "example", expected_sha256=sha256(report))
    times = np.arange(41) / 40
    features = np.zeros((len(times), 3))
    features[8] = 2
    features[20] = 3
    result = score_ensemble(features, times, heads)
    assert result["per_head_scores"].shape == (5, len(times), 3)
    assert result["thresholds"].shape == (5, 3)
    np.testing.assert_allclose(result["ensemble_scores"], features)
    for events in result["events"].values():
        np.testing.assert_array_equal(events, times[[8, 20]])
    assert result["metadata"]["head_budget"] == 5
    assert not heads[0].weights.flags.writeable
    with pytest.raises(ValueError):
        score_ensemble(features, times, heads[:4])
    with pytest.raises(ValueError):
        load_ensemble(report, "example", expected_sha256="0" * 64)


def test_zero_and_negative_thresholds_fail_before_scoring(tmp_path):
    for value in (0., -1., np.nan):
        with pytest.raises(ValueError):
            threshold_lookup({"thresholds": dict(zip(TASKS, [value, 1., 1.]))})
    report = write_heads(tmp_path, thresholds=[0., 1., 1.])
    with pytest.raises(ValueError, match="positive"):
        load_ensemble(report, "example", expected_sha256=sha256(report))


def test_checkpoint_threshold_drift_is_not_accepted(tmp_path):
    report = write_heads(tmp_path)
    content = json.loads(report.read_text())
    content["fit_records"][0]["thresholds"]["beats"] += 1
    report.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="thresholds differ"):
        load_ensemble(report, "example", expected_sha256=sha256(report))


def test_peak_spacing_and_no_constant_or_boundary_events():
    times = np.arange(50) / 40
    scores = np.zeros(50)
    scores[[0, 8, 10, 20, 49]] = [9., 2., 3., 4., 9.]
    np.testing.assert_array_equal(pick_events(scores, times, task="onsets"), times[[8, 10, 20]])
    np.testing.assert_array_equal(pick_events(scores, times, task="beats"), times[[10, 20]])
    assert len(pick_events(np.ones(50), times, task="beats")) == 0
    with pytest.raises(ValueError, match="uniform"):
        pick_events(np.ones(49), np.delete(times, 4), task="beats")
    with pytest.raises(ValueError):
        pick_events(np.full(50, np.nan), times, task="beats")


def test_policy_binds_actual_code_and_neutral_support_rule():
    metadata = policy_metadata()
    assert metadata["implementation"]["sha256"] == sha256(metadata["implementation"]["path"])
    assert metadata["augmentation"] is False
    assert metadata["new_fitting_or_threshold_tuning"] is False
    assert metadata["timing_indices"] == [87, 88, 89, 90, 91]


def test_native_resampling_masks_both_coverage_edges_and_preserves_acoustics():
    times40 = np.arange(41) / 40
    times30 = np.arange(30) / 30
    norms, raw = normalizers(), inputs(times40)
    baseline = prepare_native_input(raw, times40, times30, norms)
    changed = prepare_native_input(raw, times40, times30, norms, beats=[.05, .975])
    np.testing.assert_array_equal(changed["base92"][:, :87], baseline["base92"][:, :87])
    np.testing.assert_array_equal(changed["normalized92"][:, :87], baseline["normalized92"][:, :87])
    assert not changed["coverage"][1]  # 0.033s brackets neutral .025s and supported .05s.
    assert np.all(changed["normalized92"][~changed["coverage"], 87:] == 0)
    ended = prepare_native_input(raw, times40, times30, norms, beats=[.05, .95])
    assert not ended["coverage"][-1]  # 0.966s brackets the final detection and following neutral sample.
    assert np.all(ended["normalized92"][-1, 87:] == 0)
    assert baseline["coverage"] is None
    assert baseline["metadata"]["intervention"] is False


def test_native_empty_detection_is_not_baseline_fallback():
    times40, times30 = np.arange(41) / 40, np.arange(30) / 30
    result = prepare_native_input(inputs(times40), times40, times30, normalizers(), beats=[])
    assert result["metadata"]["intervention"] is True
    assert np.all(result["normalized92"][:, 87:] == 0)


def test_native_clock_is_interpolated_40hz_not_recomputed_30hz():
    times40, times30 = np.arange(41) / 40, np.arange(30) / 30
    norms, raw = normalizers(), inputs(times40)
    result = prepare_native_input(raw, times40, times30, norms, beats=[.1, .9])
    source = apply_timing_bridge(raw, times40, [.1, .9], norms)["base92"]
    expected = np.interp(times30, times40, source[:, 87]).astype(np.float32)
    np.testing.assert_array_equal(result["base92"][result["coverage"], 87], expected[result["coverage"]])
    at = 4  # .1333 seconds is inside the event coverage and between source frames.
    analytic = np.sin(2 * np.pi * ((times30[at] - .1) / .8))
    assert abs(float(result["base92"][at, 87]) - analytic) > 1e-4


def test_native_source_extrapolation_is_rejected():
    times40 = np.arange(40) / 40
    with pytest.raises(ValueError, match="extrapolate"):
        prepare_native_input(inputs(times40), times40, np.arange(31) / 30, normalizers(), beats=[])
