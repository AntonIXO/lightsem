"""Synthetic CPU checks for narrowly admitted v3 inference provenance.

These tests read no campaign artifacts and never execute inference or an audit.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.motion import audit_dynamic_dance_repeatability as audit


def test_nominal_native_clock_is_reported_without_mutation():
    time = np.arange(5400, dtype=np.float64)/30.
    original = time.copy()
    report = audit.nominal_clock_metrics(time)
    assert report["exactly_nominal"] is True
    assert report["maximum_absolute_nominal_error_s"] == 0.
    assert report["rms_nominal_error_s"] == 0.
    assert report["existing_admission_tolerance_s"] == 1e-8
    assert report["timestamps_rewritten"] is False
    np.testing.assert_array_equal(time, original)
    assert time.tobytes() == original.tobytes()


def test_permitted_clock_roundoff_is_measured_without_snapping():
    time = np.arange(5400, dtype=np.float64)/30.
    time[31] += 9e-9
    time[-1] -= 9e-9
    original = time.copy()
    report = audit.nominal_clock_metrics(time)
    difference = original-np.arange(5400, dtype=np.float64)/30.
    assert report["exactly_nominal"] is False
    assert report["maximum_absolute_nominal_error_s"] == np.abs(difference).max()
    assert report["rms_nominal_error_s"] == np.sqrt(np.mean(difference**2))
    assert report["saved_last_s"] == original[-1]
    assert report["timestamps_rewritten"] is False
    assert time.tobytes() == original.tobytes()
    assert time[31] != np.arange(5400, dtype=np.float64)[31]/30.


@pytest.mark.parametrize("corruption", ["global_shift", "late_frame", "above_tolerance", "wrong_fps"])
def test_native_clock_rejects_meaningful_shift_including_late_frames(corruption):
    time = np.arange(5400, dtype=np.float64)/30.
    if corruption == "global_shift":
        time += 1e-6
    elif corruption == "late_frame":
        # Relative tolerance at 179 s would wrongly admit this local error.
        time[-1] += 1e-6
    elif corruption == "above_tolerance":
        time[0] = 1.1e-8
    else:
        time = np.arange(5400, dtype=np.float64)/40.
    original = time.copy()
    with pytest.raises(ValueError):
        audit.nominal_clock_metrics(time)
    assert time.tobytes() == original.tobytes()


@pytest.mark.parametrize("corruption", ["nan", "infinity", "missing_frame", "extra_frame", "column_vector"])
def test_native_clock_rejects_nonfinite_or_incomplete_clock(corruption):
    time = np.arange(5400, dtype=np.float64)/30.
    if corruption == "nan":
        time[100] = np.nan
    elif corruption == "infinity":
        time[100] = np.inf
    elif corruption == "missing_frame":
        time = time[:-1]
    elif corruption == "extra_frame":
        time = np.arange(5401, dtype=np.float64)/30.
    else:
        time = time[:, None]
    with pytest.raises(ValueError):
        audit.nominal_clock_metrics(time)


_V2_SYNTHETIC = '''from __future__ import annotations
import numpy as np

def _project_rates(rates, mean):
    return np.clip(rates * mean, 0., 1.5)

def regional_time_maps(desired, common_time, fps=30.):
    proposal = desired.copy()
    start, end = 0, len(common_time)-1
    mean = (common_time[end]-common_time[start])*fps/(end-start)
    projected = _project_rates(proposal[start:end], mean)
    return np.cumsum(projected)/fps
'''

_HELPERS = '''
def _regional_mean_roundoff_bound(common_time: np.ndarray, start: int, end: int,
                                  fps: float) -> float:
    frames = len(common_time)-1
    span = end-start
    prefix_steps = span if end < frames else start+frames
    accumulator_ulp = np.spacing(1.5*max(end,1))
    prefix_error = .5*prefix_steps*accumulator_ulp
    endpoint_error = .5*(abs(np.spacing(common_time[start])) +
                           abs(np.spacing(common_time[end])))*abs(fps)
    arithmetic_error = 4.*np.spacing(1.5)
    return float((prefix_error+endpoint_error)/span+arithmetic_error)

def _project_regional_rates(rates: np.ndarray, mean: float, common_time: np.ndarray,
                             start: int, end: int, fps: float) -> np.ndarray:
    try:
        return _project_rates(rates, mean)
    except ValueError as error:
        if str(error) != "Required duration cannot be represented within rate limits":
            raise
        corrected_mean = float(np.clip(mean,0.,1.5))
        error_bound = _regional_mean_roundoff_bound(common_time,start,end,fps)
        if not np.isfinite(mean) or abs(mean-corrected_mean) > error_bound:
            raise
        record = {"event": "regional_clock_roundoff_retry", "start": int(start),
                  "end": int(end), "rejected_mean": float(mean),
                  "corrected_mean": corrected_mean, "analytic_ulp_bound": error_bound,
                  "mean_correction": float(corrected_mean-mean)}
        warnings.warn(json.dumps(record,sort_keys=True), RuntimeWarning,stacklevel=2)
        return _project_rates(rates,corrected_mean)
'''


def corrected_module():
    return _V2_SYNTHETIC.replace(
        "import numpy as np", "import json\nimport warnings\nimport numpy as np"
    ).replace(
        "def regional_time_maps", _HELPERS+"\ndef regional_time_maps"
    ).replace(
        "projected = _project_rates(proposal[start:end], mean)",
        "projected = _project_regional_rates(proposal[start:end], mean, common_time, start, end, fps)",
    )


def test_regional_ast_admits_only_registered_module_delta():
    report = audit.verify_regional_roundoff_ast(_V2_SYNTHETIC, corrected_module())
    assert isinstance(report, dict) and report


@pytest.mark.parametrize("corruption", [
    "projector_math", "rate_limit", "caller_math", "extra_helper", "extra_import",
    "changed_import", "duplicated_registered_import", "changed_forwarded_argument",
    "duplicate_helper", "no_new_call",
])
def test_regional_ast_rejects_other_computational_or_source_changes(corruption):
    corrected = corrected_module()
    if corruption == "projector_math":
        corrected = corrected.replace("rates * mean", "rates + mean")
    elif corruption == "rate_limit":
        corrected = corrected.replace("np.clip(rates * mean, 0., 1.5)", "np.clip(rates * mean, 0., 1.6)")
    elif corruption == "caller_math":
        corrected = corrected.replace("return np.cumsum(projected)/fps", "return np.cumsum(projected)/(fps + 1.)")
    elif corruption == "extra_helper":
        corrected += "\ndef unregistered_helper():\n    return 1\n"
    elif corruption == "extra_import":
        corrected = corrected.replace("import json", "import json\nimport random")
    elif corruption == "changed_import":
        corrected = corrected.replace("import numpy as np", "import numpy as different_np")
    elif corruption == "duplicated_registered_import":
        corrected = corrected.replace("import warnings", "import json")
    elif corruption == "changed_forwarded_argument":
        corrected = corrected.replace(
            "proposal[start:end], mean, common_time, start, end, fps)",
            "proposal[start:end], mean + .01, common_time, start, end, fps)",
        )
    elif corruption == "duplicate_helper":
        corrected += "\ndef _project_regional_rates(*args):\n    return args[0]\n"
    else:
        corrected = corrected.replace(
            "projected = _project_regional_rates(proposal[start:end], mean, common_time, start, end, fps)",
            "projected = _project_rates(proposal[start:end], mean)",
        )
    with pytest.raises(ValueError):
        audit.verify_regional_roundoff_ast(_V2_SYNTHETIC, corrected)


_RECOVERY_FILES = {
    "body.npz", "body.json", "body.json.sha256.json",
    "inference_receipt.json", "inference_receipt.json.sha256.json",
    "learned_plan.npz", "plan.npz", "plan.json", "plan.json.sha256.json",
}


def recovery_fixture(root):
    standard = root/"bodies/D6/cure_for_me"
    copied = root/"recovery_artifacts/D6_cure_infer_v3/completed_body"
    standard.mkdir(parents=True)
    for index, name in enumerate(sorted(_RECOVERY_FILES)):
        path = standard/name
        if name.endswith(".npz"):
            np.savez(path, distinct_payload=np.asarray([index], dtype=np.int64))
        else:
            path.write_text(json.dumps({"artifact": name, "distinct_payload": index}))
    # Metadata sidecars are valid hashes as well as distinct payloads.
    for name in ("body.json", "inference_receipt.json", "plan.json"):
        Path(str(standard/name)+".sha256.json").write_text(json.dumps({"sha256": audit.sha(standard/name)}))
    shutil.copytree(standard, copied)
    receipt = {
        "authorization": str(root/"records/D6_CURE_RECOVERY_AUTH_v3.json"),
        "authorization_sha256": "synthetic-authorization",
        "standard_output_directory": str(standard),
        "copied_output_directory": str(copied),
        "original_artifacts": {str(standard/name): audit.sha(standard/name) for name in sorted(_RECOVERY_FILES)},
        "copied_artifacts": {str(copied/name): audit.sha(copied/name) for name in sorted(_RECOVERY_FILES)},
        "scope": "Synthetic recovery chain; no model output.",
    }
    receipt_path = copied.parent/"RECOVERY.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    seal = Path(str(receipt_path)+".sha256.json")
    seal.write_text(json.dumps({"sha256": audit.sha(receipt_path)}))
    manifest = {
        str(path): {"sha256": audit.sha(path), "bytes": path.stat().st_size}
        for path in [receipt_path, seal, *sorted(copied.iterdir())]
    }
    return receipt, standard, copied, manifest


def test_recovery_copies_bind_all_standard_files_to_guard_owned_names(tmp_path):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    before = deepcopy(receipt)
    pins = {}
    report = audit.verify_recovery_copies(receipt, standard, copied, manifest, pins)
    assert report["files_verified"] == 9
    assert report["guard_body_path"] == str(copied/"body.npz")
    assert receipt == before
    for name in _RECOVERY_FILES:
        assert pins[str(standard/name)] == audit.sha(standard/name)
        assert pins[str(copied/name)] == audit.sha(copied/name)


def test_recovery_rejects_swapped_copy_names_even_when_hash_multiset_matches(tmp_path):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    left, right = copied/"body.npz", copied/"plan.npz"
    left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
    left.write_bytes(right_bytes)
    right.write_bytes(left_bytes)
    for path in (left, right):
        receipt["copied_artifacts"][str(path)] = audit.sha(path)
        manifest[str(path)] = {"sha256": audit.sha(path), "bytes": path.stat().st_size}
    assert sorted(receipt["original_artifacts"].values()) == sorted(receipt["copied_artifacts"].values())
    with pytest.raises(ValueError):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})


@pytest.mark.parametrize("side", ["standard", "copied"])
def test_recovery_rejects_physical_tampering_under_unchanged_receipt(tmp_path, side):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    target = standard if side == "standard" else copied
    (target/"body.npz").write_bytes(b"unregistered replacement")
    with pytest.raises(ValueError):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})


@pytest.mark.parametrize("location", ["original_receipt", "copy_receipt", "guard_manifest", "physical_file"])
def test_recovery_rejects_missing_required_artifact(tmp_path, location):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    name = "learned_plan.npz"
    if location == "original_receipt":
        del receipt["original_artifacts"][str(standard/name)]
    elif location == "copy_receipt":
        del receipt["copied_artifacts"][str(copied/name)]
    elif location == "guard_manifest":
        del manifest[str(copied/name)]
    else:
        (copied/name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})


@pytest.mark.parametrize("field", ["standard_output_directory", "copied_output_directory"])
def test_recovery_rejects_receipt_directory_mismatch(tmp_path, field):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    receipt[field] = str(tmp_path/"unrelated_output")
    with pytest.raises(ValueError):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})


def test_recovery_rejects_extra_artifact_even_when_both_copies_agree(tmp_path):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    for directory, key in ((standard, "original_artifacts"), (copied, "copied_artifacts")):
        path = directory/"unregistered.npy"
        path.write_bytes(b"same extra file")
        receipt[key][str(path)] = audit.sha(path)
    manifest[str(copied/"unregistered.npy")] = {"sha256": audit.sha(copied/"unregistered.npy"), "bytes": 15}
    with pytest.raises(ValueError):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})


def test_recovery_rejects_guard_digest_not_binding_copied_file(tmp_path):
    receipt, standard, copied, manifest = recovery_fixture(tmp_path)
    manifest[str(copied/"body.npz")]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        audit.verify_recovery_copies(receipt, standard, copied, manifest, {})
