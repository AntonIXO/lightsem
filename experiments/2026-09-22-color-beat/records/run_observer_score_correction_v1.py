"""Opt-in scoring-only half-open float64 endpoint correction.

Original evaluator/trainer/predictions and module globals stay unchanged.
Only score is exposed. The three extra correction references must be bound in
registration before any scoring. --verify-only does not score or load tensors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
from lightsem.experiments.light_layers import evaluation as frozen

PROTOCOL_SHA256 = "fecd9aa66a24f9647643f7010e1927fc90190d7f21b41dd46b03c6be69bd9037"
FROZEN_EVALUATION_SHA256 = "401825add5d89f9920e2c946e36b2cdf85e07f625ab3230c5a3844655fcca004"
ORIGINAL_SHIFT_SHA256 = "45f02973621c6c3934d377a2bccded8a42394292acfbb5efea2b98154731fa48"
EXPRESSION = "shifted[take] = a+np.mod(local, duration)"
REPLACEMENT = "shifted[take] = _guarded_endpoint_shift(a, b, duration, local, draw)"
REQUIRED_CHECKS = (
    "real_selected_and_final_geometry", "synthetic_upper_endpoint", "ordinary_exact_equality",
    "gaps_nonfinite_bad_duration_fail", "defaults_and_original_globals_preserved",
    "missing_tampered_bindings_fail_before_scoring", "disclosed_report_audit",
    "score_only_cli")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reference(path):
    return frozen.reference(Path(path).resolve())


def clone_function(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = (None if function.__kwdefaults__ is None
                             else dict(function.__kwdefaults__))
    result.__annotations__ = dict(function.__annotations__)
    result.__dict__.update(function.__dict__)
    result.__doc__ = function.__doc__
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    return result


def source_specification():
    if reference(frozen.__file__)["sha256"] != FROZEN_EVALUATION_SHA256:
        raise frozen.ContractError("Frozen evaluator file changed")
    original = inspect.getsource(frozen.shifted_candidates)
    if digest(original) != ORIGINAL_SHIFT_SHA256 or original.count(EXPRESSION) != 1:
        raise frozen.ContractError("Expected one exact frozen shifted expression")
    patched = original.replace(EXPRESSION, REPLACEMENT, 1)
    guard = 'raise ContractError("Circular shift escaped its valid segment")'
    if guard not in original or guard not in patched:
        raise frozen.ContractError("Original segment-escape assertion must remain")
    return {"original_source": original, "patched_source": patched,
            "original_shifted_function_sha256": digest(original),
            "patched_shifted_function_sha256": digest(patched),
            "frozen_score_record_source_sha256": digest(inspect.getsource(frozen.score_record)),
            "frozen_main_source_sha256": digest(inspect.getsource(frozen.main)),
            "expression_replacements": 1}


def validate_correction_registration(registration_path, protocol_path, test_receipt_path):
    """Additional opt-in bindings; never reinterpret the frozen pipeline hash."""
    registration = json.loads(Path(registration_path).read_text())
    binding = registration.get("scoring_correction")
    expected = {"source": reference(__file__), "protocol": reference(protocol_path),
                "test_receipt": reference(test_receipt_path)}
    if not isinstance(binding, dict) or set(binding) != set(expected):
        raise frozen.ContractError("Missing exact source/protocol/test_receipt scoring_correction binding")
    for name, item in expected.items():
        if binding[name] != item:
            raise frozen.ContractError("Scoring correction binding changed: " + name)
        frozen.checked_ref(binding[name])
    if expected["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise frozen.ContractError("Scoring correction protocol differs from parent-frozen protocol")
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("protocol_id") != "observer-scoring-half-open-roundoff-20260922-v1":
        raise frozen.ContractError("Unexpected correction protocol")
    receipt = json.loads(Path(test_receipt_path).read_text())
    spec = source_specification()
    if (receipt.get("status") != "passed" or receipt.get("scope") != "scoring_correction_cpu_geometry_only"
            or receipt.get("cuda_used") is not False or receipt.get("aggregate_scoring_performed") is not False
            or receipt.get("checks_failed") != 0
            or set(receipt.get("checks", [])) != set(REQUIRED_CHECKS)
            or receipt.get("checks_passed") != len(REQUIRED_CHECKS)
            or receipt.get("source") != expected["source"]
            or receipt.get("protocol") != expected["protocol"]):
        raise frozen.ContractError("Correction test receipt does not attest this exact source/protocol")
    for key in ("original_shifted_function_sha256", "patched_shifted_function_sha256"):
        if receipt.get(key) != spec[key]:
            raise frozen.ContractError("Correction test receipt function hash mismatch: " + key)
    test_source = receipt.get("test_source")
    if not isinstance(test_source, dict) or frozen.checked_ref(test_source).resolve() != Path(__file__).with_name("test_observer_score_correction_v1.py").resolve():
        raise frozen.ContractError("Correction test source is missing or changed")
    metadata = {**copy.deepcopy(expected), "registration": reference(registration_path),
                "original_frozen_pipeline_implementation": frozen.implementation_hashes(),
                **{k: v for k, v in spec.items() if not k.endswith("source")}}
    return registration, metadata


class RepairAudit:
    def __init__(self):
        self.calls = []
        self.active = None
        self.unassigned_repairs = []

    @contextmanager
    def record(self, labels, head, include_shifts=True):
        if self.active is not None:
            raise frozen.ContractError("Nested scoring audit context is unsupported")
        call = {"recording_id": labels.recording_id, "canonical_id": labels.canonical_id,
                "head": head, "include_shifts": bool(include_shifts), "repair_count": 0,
                "draw_repair_counts": [0] * 20, "max_displacement_seconds": 0.,
                "max_displacement_ulps": 0., "repairs": []}
        self.calls.append(call)
        self.active = call
        try:
            yield
        finally:
            self.active = None

    def repaired(self, start, stop, draw, count, seconds):
        row = {"draw_zero_based": int(draw), "segment_start_s": float(start),
               "segment_stop_s": float(stop), "repaired_count": int(count),
               "displacement_seconds": float(seconds), "displacement_ulps": 1.}
        if self.active is None:
            self.unassigned_repairs.append(row)
            return
        call = self.active
        call["repair_count"] += int(count)
        call["draw_repair_counts"][draw] += int(count)
        call["max_displacement_seconds"] = max(call["max_displacement_seconds"], float(seconds))
        call["max_displacement_ulps"] = 1.
        call["repairs"].append(row)

    def for_report(self, report):
        records = report.get("records", [])
        if self.unassigned_repairs or len(records) != len(self.calls):
            raise frozen.ContractError("Correction audit does not match scored record inventory")
        role = report.get("checkpoint_role")
        if role not in ("selected", "final"):
            raise frozen.ContractError("Correction audit needs explicit selected/final role")
        calls = []
        for record, call in zip(records, self.calls):
            if any(record.get(key) != call[key] for key in ("recording_id", "canonical_id", "head")):
                raise frozen.ContractError("Correction audit record/head order changed")
            calls.append({**copy.deepcopy(call), "checkpoint_role": role,
                          "recipe": record["recipe"], "seed": record["seed"]})
        return {"repaired_candidate_draw_values": sum(call["repair_count"] for call in calls),
                "audited_score_records": len(calls),
                "max_displacement_seconds": max((c["max_displacement_seconds"] for c in calls), default=0.),
                "max_displacement_ulps": max((c["max_displacement_ulps"] for c in calls), default=0.),
                "ulp_definition": "one inward float64 nextafter(stop,start) representable step",
                "records": calls}


def build_extension(metadata):
    """Isolated clones only. Original frozen module dictionary is never changed."""
    spec = source_specification()
    audit = RepairAudit()

    def guarded_endpoint_shift(a, b, duration, local, draw):
        if (not np.isfinite([a, b, duration]).all() or duration <= 0 or b <= a
                or duration != b-a):
            raise frozen.ContractError("Invalid finite positive segment duration")
        if not np.isfinite(local).all():
            raise frozen.ContractError("Nonfinite circular-shift local coordinates")
        residue = np.mod(local, duration)
        if (not np.isfinite(residue).all() or np.any(residue < 0) or np.any(residue >= duration)):
            raise frozen.ContractError("Circular-shift residue outside [0,duration)")
        shifted = a+residue
        if not np.isfinite(shifted).all():
            raise frozen.ContractError("Nonfinite circular-shift result")
        endpoint = shifted == b
        if np.any(endpoint):
            inward = np.nextafter(b, a)
            if not a <= inward < b:
                raise frozen.ContractError("No representable inward endpoint")
            shifted[endpoint] = inward
            audit.repaired(a, b, draw, int(np.sum(endpoint)), b-inward)
        # All other arithmetic is unchanged. The original frozen membership
        # assertion below the substituted expression still rejects real escapes.
        return shifted

    shifted_namespace = dict(frozen.shifted_candidates.__globals__)
    shifted_namespace["_guarded_endpoint_shift"] = guarded_endpoint_shift
    exec(compile(spec["patched_source"], "<hash-bound-observer-shift-correction-v1>", "exec"), shifted_namespace)
    shifted = shifted_namespace["shifted_candidates"]
    score_namespace = dict(frozen.score_record.__globals__)
    score_namespace["shifted_candidates"] = shifted
    score_record = clone_function(frozen.score_record, score_namespace)

    def audited_score_record(labels, curves, thresholds, head, *, include_shifts=True):
        with audit.record(labels, head, include_shifts):
            return score_record(labels, curves, thresholds, head, include_shifts=include_shifts)
    audited_score_record.__defaults__ = frozen.score_record.__defaults__
    audited_score_record.__kwdefaults__ = dict(frozen.score_record.__kwdefaults__)

    def report_writer(path, report, campaign_root):
        # Recheck the opt-in inputs before publishing an amended report.
        for name in ("source", "protocol", "test_receipt", "registration"):
            frozen.checked_ref(metadata[name])
        if frozen.implementation_hashes() != metadata["original_frozen_pipeline_implementation"]:
            raise frozen.ContractError("Frozen pipeline source changed during amended scoring")
        actual = {**copy.deepcopy(metadata),
            "actual_scorer": "disclosed opt-in endpoint-corrected scorer; NOT original unamended executable",
            "original_bindings_meaning": "unchanged frozen pipeline/targets/predictions; actual scorer identified here",
            "correction": "Only computed shifted==own_segment_stop becomes nextafter(stop,start)",
            "execution": "isolated shifted source substitution; score_record/main bytecode clones with separate globals",
            "audit": audit.for_report(report)}
        frozen.write_json_exclusive(path, {**report, "scoring_correction": actual}, campaign_root)

    main_namespace = dict(frozen.main.__globals__)
    main_namespace["score_record"] = audited_score_record
    main_namespace["write_json_exclusive"] = report_writer
    main = clone_function(frozen.main, main_namespace)
    return SimpleNamespace(main=main, shifted_candidates=shifted, score_record=score_record,
        audited_score_record=audited_score_record, report_writer=report_writer,
        audit=audit, source_specification=spec, guarded_endpoint_shift=guarded_endpoint_shift)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score",))
    for name in ("manifest", "transforms", "registration", "output", "prediction-index", "baselines",
                 "correction-protocol", "correction-test-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    registration, metadata = validate_correction_registration(
        args.registration, args.correction_protocol, args.correction_test_receipt)
    # These are explicitly the original frozen pipeline bindings. No hashes are
    # relaxed/relabelled to pass the amended scorer off as that old executable.
    frozen._registration(args.registration, args.manifest, args.transforms, phase="score")
    for name in ("prediction_index", "baselines"):
        if frozen.checked_ref(registration.get(name)).resolve() != getattr(args, name).resolve():
            raise frozen.ContractError("Frozen score input reference mismatch: " + name)
    if args.output.exists():
        raise frozen.ContractError("Output exists; no overwrite/retry of failed artifacts")
    extension = build_extension(metadata)
    if args.verify_only:
        print(json.dumps({"status": "bindings_verified_not_scored", "scoring_correction": metadata}, sort_keys=True))
        return 0
    original_args = ["score"]
    for name in ("manifest", "transforms", "registration", "output", "prediction_index", "baselines"):
        original_args.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    extension.main(original_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
