"""CPU geometry/binding tests only. No aggregate metric, scoring CLI, or GPU run."""
import argparse
import contextlib
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from lightsem.experiments.light_layers import evaluation as ev

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "run_observer_score_correction_v1.py"
PROTOCOL = DOCS / "OBSERVER_SCORING_NUMERICAL_CORRECTION_V1.json"
spec = importlib.util.spec_from_file_location("observer_score_correction_v1", SOURCE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def equal_tree(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        assert left.dtype == right.dtype
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ev.ContractError:
        return
    raise AssertionError("Expected ContractError")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError("Test receipt exists; no overwrite")
    globals_before = {key: id(value) for key, value in ev.__dict__.items()}
    source_before = ev.implementation_hashes()
    original_functions = {name: getattr(ev, name) for name in ("shifted_candidates", "score_record", "main")}
    extension = c.build_extension({})
    checks = []
    fixture_counts = {}
    fixtures = {}
    for role in ("selected", "final"):
        path = DOCS / f"observer_shift_boundary_fixture_{role}_v1.json"
        fixture = json.loads(path.read_text())
        fixtures[role] = fixture
        times = np.asarray(fixture["candidate_times_s"])
        confidences = np.asarray(fixture["confidences"])
        plan = fixture["plan"]
        before = copy.deepcopy(plan)
        rejected(ev.shifted_candidates, times, confidences, plan)
        changed = extension.shifted_candidates(times, confidences, plan)
        np.testing.assert_array_equal(changed["real_times_s"], times)
        np.testing.assert_array_equal(changed["confidences"], confidences)
        assert len(changed["draws"]) == 20 and plan == before
        repairs = 0
        for draw, output in enumerate(changed["draws"]):
            np.testing.assert_array_equal(output["confidences"], confidences)
            assert len(output["times_s"]) == len(times)
            for row in plan["segments"]:
                a, b, duration = row["start_s"], row["stop_s"], row["duration_s"]
                take = (times >= a) & (times < b)
                local = times[take]-a+row["offsets_s"][draw]
                old = a+np.mod(local, duration)
                new = output["times_s"][take]
                np.testing.assert_array_equal(old != new, old == b)
                np.testing.assert_array_equal(new[old != b], old[old != b])
                assert np.all(new[old == b] == np.nextafter(b, a))
                assert np.all((new >= a) & (new < b))
                assert output["wrapped_candidate_count"] == int(np.sum(local >= duration))
                repairs += int(np.sum(old == b))
        assert repairs == 1
        fixture_counts[role] = repairs
    checks.append("real_selected_and_final_geometry")

    plan = ev.shift_plan(96+np.arange(2560)/40, [[96., 104.]], 40)
    plan["segments"][0]["offsets_s"][0] = 7.575
    assert 96.+np.mod(np.asarray([96.425])-96.+7.575, 8.) == 104.
    rejected(ev.shifted_candidates, [96.425], [.7], plan)
    new = extension.shifted_candidates([96.425], [.7], plan)
    assert new["draws"][0]["times_s"][0] == np.nextafter(104., 96.)
    assert new["draws"][0]["times_s"][0] != 96.
    np.testing.assert_array_equal(new["draws"][0]["confidences"], [.7])
    checks.append("synthetic_upper_endpoint")

    ordinary_cases = 0
    rng = np.random.default_rng(20260922)
    for rate in (10, 40):
        for iteration in range(100):
            start = float(rng.uniform(0, 10000))
            intervals = [[start+28.013, start+31.026], [start+32.037, start+35.999]]
            plan = ev.shift_plan(start+np.arange(64*rate)/rate, intervals, rate)
            times = np.r_[rng.uniform(*intervals[0], 20), rng.uniform(*intervals[1], 20)]
            confidence = rng.random(len(times))
            old = ev.shifted_candidates(times, confidence, plan)
            new = extension.shifted_candidates(times, confidence, plan)
            equal_tree(old, new)
            for row in new["draws"]:
                np.testing.assert_array_equal(ev._regions(row["times_s"], np.asarray(intervals)), [0]*20+[1]*20)
            ordinary_cases += 1
    for intervals, times, confidence in (([[0., 1.], [2., 3.]], [.5, 2.5], [1., 2.]),
                                         ([[0., 8.]], [], [])):
        plan = ev.shift_plan(np.arange(80)/10, intervals, 10)
        equal_tree(ev.shifted_candidates(times, confidence, plan), extension.shifted_candidates(times, confidence, plan))
    checks.append("ordinary_exact_equality")

    plan = ev.shift_plan(np.arange(80)/10, [[0., 8.]], 10)
    for duration in (0., -1., float("nan"), float("inf"), 16.):
        bad = copy.deepcopy(plan)
        bad["segments"][0]["duration_s"] = duration
        with np.errstate(all="ignore"):
            rejected(extension.shifted_candidates, [7.9], [1.], bad)
    for offset in (float("nan"), float("inf"), -float("inf")):
        bad = copy.deepcopy(plan)
        bad["segments"][0]["offsets_s"][0] = offset
        rejected(extension.shifted_candidates, [7.9], [1.], bad)
    # Consistent local geometry, but a genuine gap in the declared union: the
    # ORIGINAL escape guard (not a clipping/helper substitute) must still fire.
    gap = copy.deepcopy(plan)
    gap["eligible_intervals_s"] = [[0., 4.]]
    rejected(ev.shifted_candidates, [3.9], [1.], gap)
    rejected(extension.shifted_candidates, [3.9], [1.], gap)
    assert 'raise ContractError("Circular shift escaped its valid segment")' in extension.source_specification["patched_source"]
    checks.append("gaps_nonfinite_bad_duration_fail")

    for name in ("score_record", "main"):
        original, cloned = getattr(ev, name), getattr(extension, name)
        assert cloned.__code__ is original.__code__
        assert cloned.__defaults__ == original.__defaults__
        assert cloned.__kwdefaults__ == original.__kwdefaults__
        assert inspect.signature(cloned) == inspect.signature(original)
        assert cloned.__globals__ is not original.__globals__
    assert extension.score_record.__kwdefaults__["include_shifts"] is True
    assert extension.audited_score_record.__kwdefaults__["include_shifts"] is True
    assert extension.main.__globals__["score_record"] is extension.audited_score_record
    assert extension.score_record.__globals__["shifted_candidates"] is extension.shifted_candidates
    checks.append("defaults_and_original_globals_preserved")

    source_info = c.source_specification()
    test_attestation = {"status": "passed", "scope": "scoring_correction_cpu_geometry_only",
        "cuda_used": False, "aggregate_scoring_performed": False, "checks_failed": 0,
        "checks_passed": len(c.REQUIRED_CHECKS), "checks": list(c.REQUIRED_CHECKS),
        "source": c.reference(SOURCE), "protocol": c.reference(PROTOCOL),
        "test_source": c.reference(__file__),
        "original_shifted_function_sha256": source_info["original_shifted_function_sha256"],
        "patched_shifted_function_sha256": source_info["patched_shifted_function_sha256"]}
    # Temporary synthetic attestations exercise the registration validator only.
    # None authorizes or executes real scoring; the actual receipt is emitted
    # only after ALL tests pass, at the --receipt path below.
    with tempfile.TemporaryDirectory(prefix="observer-score-correction-tests-") as folder:
        folder = Path(folder)
        receipt = folder / "synthetic-test-receipt.json"
        receipt.write_text(json.dumps(test_attestation))
        registration = folder / "synthetic-registration.json"
        binding = {"source": c.reference(SOURCE), "protocol": c.reference(PROTOCOL), "test_receipt": c.reference(receipt)}
        registration.write_text(json.dumps({"scoring_correction": binding}))
        _, metadata = c.validate_correction_registration(registration, PROTOCOL, receipt)
        for field in (None, "source", "protocol", "test_receipt"):
            broken = copy.deepcopy(binding)
            if field is None:
                content = {}
            else:
                broken[field]["sha256"] = "0"*64
                content = {"scoring_correction": broken}
            registration.write_text(json.dumps(content))
            rejected(c.validate_correction_registration, registration, PROTOCOL, receipt)
        registration.write_text(json.dumps({"scoring_correction": binding}))
        for field in ("source", "protocol", "patched_shifted_function_sha256", "checks_failed"):
            bad = copy.deepcopy(test_attestation)
            if field in ("source", "protocol"):
                bad[field]["sha256"] = "0"*64
            elif field == "checks_failed":
                bad[field] = 1
            else:
                bad[field] = "0"*64
            receipt.write_text(json.dumps(bad))
            b = {**binding, "test_receipt": c.reference(receipt)}
            registration.write_text(json.dumps({"scoring_correction": b}))
            rejected(c.validate_correction_registration, registration, PROTOCOL, receipt)
        receipt.write_text(json.dumps(test_attestation))
        binding["test_receipt"] = c.reference(receipt)
        registration.write_text(json.dumps({"scoring_correction": binding}))
        _, metadata = c.validate_correction_registration(registration, PROTOCOL, receipt)
        checks.append("missing_tampered_bindings_fail_before_scoring")

        for role in ("selected", "final"):
            ext = c.build_extension(metadata)
            fixture = fixtures[role]
            labels = SimpleNamespace(recording_id=fixture["recording_id"], canonical_id=fixture["canonical_id"])
            with ext.audit.record(labels, "color"):
                ext.shifted_candidates(fixture["candidate_times_s"], fixture["confidences"], fixture["plan"])
            report = {"status": "synthetic_geometry_audit_not_scientific_scores", "checkpoint_role": role,
                "bindings": {"unchanged": "frozen_pipeline"}, "records": [{"recording_id": labels.recording_id,
                "canonical_id": labels.canonical_id, "head": "color", "recipe": "S_none_P0", "seed": 0}]}
            original = copy.deepcopy(report)
            output = folder / (role+"_synthetic-report.json")
            ext.report_writer(output, report, folder)
            result = json.loads(output.read_text())
            assert report == original and result["bindings"] == original["bindings"]
            correction = result["scoring_correction"]
            assert "NOT original unamended" in correction["actual_scorer"]
            audit = correction["audit"]
            assert audit["repaired_candidate_draw_values"] == 1 and audit["max_displacement_ulps"] == 1.
            assert audit["max_displacement_seconds"] == 134.6-np.nextafter(134.6, 126.6)
            record = audit["records"][0]
            assert record["checkpoint_role"] == role and record["head"] == "color"
            assert record["draw_repair_counts"][4] == 1 and sum(record["draw_repair_counts"]) == 1
        checks.append("disclosed_report_audit")

    for phase in ("predict", "prepare-baselines", "schema"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                c.main([phase])
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("Non-score CLI phase accepted")
    checks.append("score_only_cli")
    assert set(checks) == set(c.REQUIRED_CHECKS)
    assert {key: id(value) for key, value in ev.__dict__.items()} == globals_before
    assert ev.implementation_hashes() == source_before
    assert all(getattr(ev, name) is function for name, function in original_functions.items())
    assert not torch.cuda.is_initialized()
    result = {**test_attestation, "checks": checks, "fixture_endpoint_corrections": fixture_counts,
        "ordinary_exact_equality_cases": ordinary_cases+2,
        "fixtures": {role: c.reference(DOCS / f"observer_shift_boundary_fixture_{role}_v1.json") for role in fixtures},
        "original_implementation_hashes_before": source_before,
        "original_implementation_hashes_after": ev.implementation_hashes(),
        "claim_limit": "CPU geometry, isolated-clone, registration, and synthetic report tests; no aggregate scoring or GPU"}
    with args.receipt.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
