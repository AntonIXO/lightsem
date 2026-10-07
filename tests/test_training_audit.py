"""Contract tests for the autonomous training-audit primitives.

These fixtures deliberately use tiny synthetic bodies and output streams.  The
tests exercise the failure taxonomy that a training run must expose before a
human review: non-finite values, inactivity, exact duplicate outputs, true
input sensitivity, representation collapse, and missing provenance.  None of
the tests treats a non-zero motion hash as evidence that a model improved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.evaluation.training_audit import (
    collapse_metrics,
    compare_training_runs,
    input_sensitivity,
    motion_metrics,
    provenance_report,
    stream_metrics,
)
from scripts.evaluate_training import build_report


def _body(n: int = 180) -> np.ndarray:
    """A finite, smooth 22-joint body with a few independent gestures."""
    t = np.arange(n, dtype=np.float64) / 30.0
    body = np.zeros((n, 22, 3), dtype=np.float64)
    body[:, :, 1] = 1.0
    body[:, 7, 0] = 0.08 * np.sin(2.0 * t)
    body[:, 11, 2] = 0.06 * np.sin(0.7 * t + 0.4)
    body[:, 20, 0] = 0.12 * np.sin(0.35 * t)
    return body


def _get(mapping: dict, *names, default=...):
    """Read a canonical field while permitting the documented aliases."""
    for name in names:
        if name in mapping:
            return mapping[name]
    if default is not ...:
        return default
    raise AssertionError(f"missing one of {names!r} in {mapping!r}")


def test_motion_metrics_reject_nonfinite_and_keep_raw_activity_separate():
    body = _body()
    report = motion_metrics(body, fps=30.0)
    assert _get(report, "finite", "all_finite") is True
    assert _get(report, "constant", "constant_body") is False
    assert _get(report, "speed_p95", "p95_speed_m_s", "p95_speed") > 0.0
    # A non-finite native body must be an explicit diagnostic, never silently
    # converted to zero or omitted from the aggregate.
    bad = body.copy()
    bad[20, 7, 1] = np.nan
    bad_report = motion_metrics(bad, fps=30.0)
    assert _get(bad_report, "finite", "all_finite") is False
    assert _get(bad_report, "finite_fraction", "finite_ratio") < 1.0


def test_stream_metrics_distinguishes_inactivity_from_exact_duplicates():
    t = np.arange(120, dtype=np.float64)
    first = np.sin(t / 11.0)
    duplicate = first.copy()
    independent = np.cos(t / 7.0)
    streams = np.stack((first, duplicate, independent), axis=1)
    report = stream_metrics(streams)
    assert _get(report, "finite", "all_finite") is True
    groups = _get(report, "duplicate_groups", "exact_duplicate_groups")
    # A duplicate group is represented as a collection of stream-index lists;
    # the test intentionally accepts either [0, 1] or [(0, 1)] serialization.
    normalised = {tuple(sorted(map(int, group))) for group in groups if len(group) > 1}
    assert (0, 1) in normalised
    assert _get(report, "inactive_fraction", "inactive_stream_fraction") == 0.0

    nonfinite = streams.copy()
    nonfinite[0, 0] = np.nan
    nonfinite_report = stream_metrics(nonfinite)
    assert _get(nonfinite_report, "finite", "all_finite") is False
    assert _get(nonfinite_report, "nonfinite_count") > 0

    inactive = np.zeros((120, 2), dtype=np.float64)
    inactive_report = stream_metrics(inactive)
    assert _get(inactive_report, "inactive_fraction", "inactive_stream_fraction") == 1.0
    # Zero activity is a stream failure; it is not representation-collapse
    # evidence by itself.
    assert _get(inactive_report, "collapse", "representation_collapse", default=False) is not True


def test_collapse_metrics_detects_low_rank_samples_but_not_single_quiet_event():
    rng = np.random.default_rng(12)
    varied = rng.normal(size=(32, 8))
    collapsed = np.repeat(varied[:1], len(varied), axis=0)
    varied_report = collapse_metrics(varied)
    collapsed_report = collapse_metrics(collapsed)
    assert _get(varied_report, "collapsed", "collapse") is False
    assert _get(collapsed_report, "collapsed", "collapse") is True
    assert _get(collapsed_report, "diversity", "pairwise_distance", "mean_pairwise_distance") == 0.0
    assert _get(varied_report, "effective_rank", "rank", "covariance_rank") > 1.0


def test_input_sensitivity_requires_fixed_controls_and_measures_assigned_change():
    reference = _body()
    same = reference.copy()
    changed = reference.copy()
    changed[:, 7, 0] += 0.08 * np.sin(np.arange(len(changed)) / 9.0)
    report = input_sensitivity(reference, {"same": same, "left_arm": changed})
    rows = report.get("interventions", report)
    same_row, changed_row = rows["same"], rows["left_arm"]
    assert _get(same_row, "changed", "sensitive") is False
    assert _get(same_row, "rms", "rms_difference", "difference_rms") == 0.0
    assert _get(changed_row, "changed", "sensitive") is True
    assert _get(changed_row, "rms", "rms_difference", "difference_rms") > 0.0
    assert _get(report, "finite", "all_finite") is True


def test_missing_provenance_is_unqualified_and_never_a_zero_score():
    required = ("checkpoint_sha256", "audio_sha256", "seed", "mapping_sha256")
    complete = {key: (7 if key == "seed" else "a" * 64) for key in required}
    good = provenance_report(complete, required_fields=required)
    assert _get(good, "complete", "provenance_complete") is True
    assert _get(good, "status", "verdict") in {"qualified", "pass", "complete"}

    missing = dict(complete)
    del missing["mapping_sha256"]
    bad = provenance_report(missing, required_fields=required)
    assert _get(bad, "complete", "provenance_complete") is False
    assert "mapping_sha256" in _get(bad, "missing", "missing_fields")
    assert _get(bad, "status", "verdict") in {
        "unqualified", "diagnostic", "fail", "incomplete"
    }
    # An incomplete receipt cannot be promoted by a numerically quiet output.
    assert _get(bad, "score", "value", default=None) != 0


def test_run_comparison_reports_regression_and_improvement_by_gate():
    baseline = {"gates": {
        "finite": True, "provenance": True, "collapse": True,
        "input_sensitivity": False, "raw_dynamics": True, "streams": True,
    }}
    candidate = {"gates": {
        "finite": True, "provenance": True, "collapse": False,
        "input_sensitivity": True, "raw_dynamics": True, "streams": True,
    }}
    result = compare_training_runs(baseline, candidate)
    assert _get(result, "status", "verdict") == "regression"
    assert "collapse" in _get(result, "candidate_failed_gates", "failed_gates")
    assert "input_sensitivity" in _get(result, "candidate_improved_gates", "improved_gates")

    # A report with no gate evidence cannot be called a regression or an
    # improvement merely because the candidate has a different file hash.
    unknown = compare_training_runs({"gates": {}}, {"gates": {}})
    assert _get(unknown, "status", "verdict") == "unchanged_or_unmeasurable"


def test_cli_report_promotes_only_when_all_declared_health_gates_pass():
    """The script wires primitive diagnostics into a conservative summary."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        body = _body()
        body_path = root / "body.npz"
        stream_path = root / "streams.npz"
        np.savez(body_path, body=body)
        t = np.arange(len(body), dtype=np.float64)
        streams = np.stack((np.sin(t / 9.0), np.cos(t / 11.0), np.sin(t / 13.0 + 0.2)), axis=1)
        np.savez(stream_path, streams=streams)
        sample_path = root / "sample.npz"
        np.savez(sample_path, body=body + 0.01 * np.sin(t[:, None, None] / 7.0))
        intervention_path = root / "intervention.npz"
        intervention = body.copy()
        intervention[:, 7, 0] += 0.05 * np.sin(t / 5.0)
        np.savez(intervention_path, body=intervention)
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps({
            "checkpoint_sha256": "a" * 64,
            "audio_sha256": "b" * 64,
            "source_body_sha256": "c" * 64,
        }))
        args = argparse.Namespace(
            manifest=None, body=body_path, stream=stream_path,
            sample=[f"candidate={sample_path}"],
            intervention=[f"left_arm={intervention_path}"],
            provenance=receipt_path,
            require_provenance="checkpoint_sha256,audio_sha256,source_body_sha256",
            fps=30.0, rgb_channels=None, brightness_channels=None, plot=None,
        )
        report = build_report(args)
        assert all(report["gates"].values())
        assert report["status"] == "diagnostic_only"
        assert report["health_status"] == "healthy_checks_passed"
        assert "musical or artistic" in report["interpretation"]
