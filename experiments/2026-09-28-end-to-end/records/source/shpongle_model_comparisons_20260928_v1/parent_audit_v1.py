#!/usr/bin/env python3
"""Parent-side V2 completion-chain + probability sanity audit. No model imports.

Fixed prospective tolerance checks gross score/logit mismatch. Saved FP32 scores
remain cue authority. This does NOT replay CUDA sigmoid bit-exactly, alter cues,
apply a palette-argmax tolerance, render media or certify video delivery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

RECIPES = ("S_none_P0", "S_V6_P0", "I_VR_P0", "S_VR_P0_Ashuf")
PROBABILITY_ATOL = 8 * 2**-23
EXPECTED_PLAN_SHA256 = "0e9286f7428346a2ff01d2f5cd45b75fd39ac1b564d663bb209a2e48f17cffd4"
FRAMES40 = 11092
GIB = 2**30


class Blocked(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Blocked(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Noncanonical path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference(path):
    path = canonical(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def checked(ref, expected_path):
    require(isinstance(ref, dict) and ref.get("path") == str(expected_path), "Wrong artifact path")
    path = canonical(expected_path)
    require(path.is_file() and digest(path) == ref.get("sha256"), "Missing/changed artifact")
    if "bytes" in ref:
        require(type(ref["bytes"]) is int and ref["bytes"] == path.stat().st_size, "Artifact size differs")
    return path


def checked_declared(ref):
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str), "Malformed reference")
    return checked(ref, ref["path"])


def read_json(path):
    return json.loads(Path(path).read_text())


def check_sources(sources):
    require(isinstance(sources, dict) and bool(sources), "Empty source binding")
    for path, sha in sources.items():
        checked({"path": path, "sha256": sha}, path)


def native_receipt(receipt, *, count, plan_sha256, sources, tests, command, log_ref):
    """A process exit or status string alone cannot certify a unittest run."""
    for key, value in {"tests_run": count, "tests_passed": count, "failures": 0, "errors": 0, "skipped": 0,
                       "exit_code": 0, "real_model_forward_calls": 0, "real_checkpoint_loads": 0, "sealed_TEST_reads": 0}.items():
        require(type(receipt.get(key)) is int and receipt[key] == value, "Native receipt count/exit: " + key)
    require(receipt.get("status") == "passed" and receipt.get("plan_sha256") == plan_sha256,
            "Native receipt status/plan")
    require(receipt.get("source_files") == sources and receipt.get("test_sources") == tests,
            "Native receipt source/test binding")
    require(receipt.get("command") == command and receipt.get("log") == log_ref,
            "Native receipt command/log binding")
    text = checked_declared(log_ref).read_text()
    summaries = re.findall(r"^Ran (\d+) tests? in .+$", text, flags=re.MULTILINE)
    require(summaries == [str(count)] and text.rstrip().splitlines()[-1:] == ["OK"],
            "Native log must end with the one expected unittest run and OK")
    require(len(re.findall(r"^.+ \.\.\. ok$", text, flags=re.MULTILINE)) == count, "Native log explicit passed-test count")
    require(not re.search(r"^FAILED|^OK \(|\.\.\. skipped", text, flags=re.MULTILINE),
            "Native log reports failures/skips")


def probability_sanity(logits, scores, expected_length):
    """Stable float64 reference, fixed absolute tolerance; never replace scores."""
    import numpy as np
    require(isinstance(logits, np.ndarray) and isinstance(scores, np.ndarray)
            and logits.shape == scores.shape == (expected_length,)
            and logits.dtype == scores.dtype == np.float32, "Logit/score shape or dtype")
    require(np.isfinite(logits).all() and np.isfinite(scores).all(), "Nonfinite logit/score")
    require(((scores >= 0) & (scores <= 1)).all(), "Score outside probability range")
    x = logits.astype(np.float64)
    expected = np.empty_like(x)
    positive = x >= 0
    expected[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    expected[~positive] = e / (1 + e)
    maximum = float(np.max(np.abs(scores.astype(np.float64) - expected), initial=0))
    require(maximum <= PROBABILITY_ATOL, "Gross saved score/logit mismatch")
    return {"status": "passed", "values_checked": expected_length, "maximum_absolute_error": maximum,
            "absolute_tolerance": PROBABILITY_ATOL, "reference": "stable_float64_sigmoid",
            "exact_CUDA_sigmoid_claim": False, "saved_scores_remain_cue_authority": True}


def arm_chain(plan, arm, plan_ref, delivery_ref, complete_ref):
    directory = canonical(Path(plan["derived_root"]) / arm["subdir"])
    complete = read_json(checked(complete_ref, directory / "complete.json"))
    require(complete.get("status") == "complete" and complete.get("recipe") == arm["recipe"], "Arm completion status/recipe")
    for key, expected in (("plan", plan_ref), ("authorization", delivery_ref)):
        require(complete.get(key) == expected, "Arm completion binding: " + key)
    inference_ref, proof_ref = complete.get("inference"), complete.get("proof")
    inference = read_json(checked(inference_ref, directory / "inference_saved.json"))
    proof = read_json(checked(proof_ref, directory / "protected_byte_equality.json"))
    for key, expected in (("recipe", arm["recipe"]), ("checkpoint", arm["checkpoint"]),
                          ("plan", plan_ref), ("authorization", delivery_ref),
                          ("inputs", plan["inputs"]), ("source_files", plan["source_files"]),
                          ("audio_route", arm["audio_route"]),
                          ("ashuf_not_primary_donor_evaluation", arm["non_primary_donor_route"])):
        require(inference.get(key) == expected, "Inference binding: " + key)
    runtime = inference.get("runtime", {})
    for key, expected in {"encoder_forward_calls": 1, "all_state_head_calls": 1,
                          "actual_action_logits_authority": True, "full_T_palette_head_recomputation": False,
                          "no_argmax_tolerance_expansion": True, "float32": True, "autocast": False,
                          "TF32": False, "model_eval": True, "chunking": False,
                          "training": False, "visual_or_pose_input": False}.items():
        require(type(runtime.get(key)) is type(expected) and runtime[key] == expected, "Inference runtime scope: " + key)
    model_path = checked(inference.get("model_outputs"), directory / "model_outputs.npz")
    checked(inference.get("overlay"), directory / "overlay_dmx.npz")
    require(proof.get("status") == "passed" and proof.get("model_forward_calls") == 0
            and proof.get("protected_slots_count") == 469 and proof.get("changed_protected_bytes") == 0
            and proof.get("protected_bytes_compared") == FRAMES40 * 469, "Protected-byte proof summary")
    for key in ("exact_RGB_and_dimmer_reconstruction", "no_invented_color_cues",
                "accent_events_recomputed_from_scores", "color_cues_recomputed_from_scores",
                "palette_visualization_rebuilt_from_bound_prototypes", "actual_action_logits_and_compact_choices_replayed",
                "final_state_exact", "no_rebatched_palette_logit_authority", "no_argmax_tolerance_expansion"):
        require(proof.get(key) is True, "Missing mapped-output proof: " + key)
    return model_path, {"recipe": arm["recipe"], "completion": complete_ref, "inference": inference_ref,
                        "mapping_proof": proof_ref, "model_outputs": inference["model_outputs"],
                        "overlay": inference["overlay"]}


def audit_four(plan, plan_ref, delivery_ref):
    """Read all four arms. Caller writes one receipt only after this returns."""
    import numpy as np
    aggregate_path = canonical(Path(plan["derived_root"]) / "all_complete.json")
    aggregate_ref = reference(aggregate_path)
    aggregate = read_json(aggregate_path)
    require(aggregate.get("status") == "complete" and aggregate.get("plan") == plan_ref
            and aggregate.get("completion_kind") == "INFERENCE_MAPPING_ONLY"
            and aggregate.get("videos_completed") is False and aggregate.get("media_audits_completed") is False
            and aggregate.get("anchor_not_rerun") is True, "Aggregate completion scope")
    refs = aggregate.get("arms", [])
    require(len(refs) == len(plan["arms"]) == 4, "Exactly four completion links required")
    with np.load(plan["inputs"]["baseline"]["path"], allow_pickle=False) as arrays:
        baseline = arrays["companion"]
    require(baseline.shape == (FRAMES40, 512) and baseline.dtype == np.uint8, "Bound baseline shape/type")
    array_sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    rows = []
    for arm, complete_ref in zip(plan["arms"], refs):
        model_path, row = arm_chain(plan, arm, plan_ref, delivery_ref, complete_ref)
        with np.load(model_path, allow_pickle=False) as arrays:
            row["probability_sanity"] = {
                "accent": probability_sanity(arrays["accent_logit"], arrays["accent_score"], FRAMES40),
                "color": probability_sanity(arrays["chromatic_change_logit"], arrays["color_score"], FRAMES40 // 4)}
        proof = read_json(checked_declared(row["mapping_proof"]))
        with np.load(row["overlay"]["path"], allow_pickle=False) as arrays:
            overlay, allowed = arrays["overlay"], arrays["allowed_slot_mask"]
        require(overlay.shape == baseline.shape and overlay.dtype == np.uint8
                and allowed.shape == (512,) and allowed.dtype == np.bool_ and int(allowed.sum()) == 43,
                "Proof/output fingerprint schema")
        hashes = {"baseline_full_sha256": array_sha(baseline), "overlay_full_sha256": array_sha(overlay),
                  "baseline_protected_sha256": array_sha(baseline[:, ~allowed]),
                  "overlay_protected_sha256": array_sha(overlay[:, ~allowed])}
        require(all(proof.get(k) == v for k, v in hashes.items()), "Proof does not fingerprint these bound outputs")
        require(hashes["baseline_protected_sha256"] == hashes["overlay_protected_sha256"], "Protected byte fingerprint differs")
        row["mapping_proof_array_hashes"] = hashes
        rows.append(row)
    # Authenticate files again before publication; no outputs are modified.
    checked(aggregate_ref, aggregate_path)
    for row in rows:
        for key in ("completion", "inference", "mapping_proof", "model_outputs", "overlay"):
            checked_declared(row[key])
    return {"status": "passed", "completion_kind": "INFERENCE_MAPPING_READBACK_ONLY",
            "videos_completed": False, "media_audits_completed": False, "model_calls": 0,
            "plan": plan_ref, "delivery_authorization": delivery_ref, "aggregate": aggregate_ref,
            "arms": rows, "absolute_probability_tolerance": PROBABILITY_ATOL,
            "exact_CUDA_sigmoid_claim": False, "saved_scores_remain_cue_authority": True,
            "palette_argmax_tolerance_added": False,
            "scope": "Authenticates V2 canonical completion/proof/output chain and checks gross event-score/logit consistency; mapping semantics are certified by the authenticated V2 proof, not independently recomputed here. No video delivery or human audiovisual acceptance claim."}


def authorize(policy_path, authorization_path):
    policy_ref, authorization_ref = reference(policy_path), reference(authorization_path)
    policy, authorization = read_json(policy_path), read_json(authorization_path)
    require(policy.get("schema_version") == 1 and policy.get("probability_atol") == PROBABILITY_ATOL,
            "Prospective policy changed")
    postflight = read_json(checked_declared(policy["postflight_policy"]))
    require(postflight.get("id") == "four-demo-postflight-guard-20260928-v1"
            and postflight["event_probability_sanity"]["maximum_absolute_error"] == PROBABILITY_ATOL,
            "Parent prospective postflight policy mismatch")
    require(authorization.get("status") == "authorized_readback_audit" and authorization.get("policy") == policy_ref,
            "Auditor not authorized for this policy")
    require(str(Path(__file__).resolve()) in policy["source_files"], "Auditor source omitted")
    check_sources(policy["source_files"]); check_sources(policy["test_sources"])
    plan_ref = policy["plan"]
    require(plan_ref["sha256"] == EXPECTED_PLAN_SHA256 and authorization.get("plan") == plan_ref, "Wrong V2 plan")
    auditor_receipt = read_json(checked_declared(authorization["auditor_native_test_receipt"]))
    native_receipt(auditor_receipt, count=policy["native_test_count"], plan_sha256=plan_ref["sha256"],
                   sources=policy["source_files"], tests=policy["test_sources"], command=policy["native_test_command"],
                   log_ref=authorization["auditor_native_test_log"])
    require(authorization["auditor_native_test_log"]["path"] == policy["native_test_log_path"], "Auditor test log path")
    plan = read_json(checked_declared(plan_ref))
    require(tuple(arm["recipe"] for arm in plan["arms"]) == RECIPES, "Four fixed recipe order")
    require(plan.get("anchor_inference_allowed") is False and plan.get("audio_route") == "recipient_audio_delivery", "Delivery route/anchor scope")
    delivery_ref = authorization["delivery_authorization"]
    delivery = read_json(checked_declared(delivery_ref))
    require(delivery.get("status") == "authorized_delivery" and {"infer", "verify"}.issubset(delivery.get("phases", []))
            and delivery.get("plan_sha256") == plan_ref["sha256"]
            and delivery.get("source_files") == plan["source_files"] and delivery.get("test_sources") == plan["test_sources"],
            "V2 delivery authorization mismatch")
    require(delivery.get("native_test_command") == plan["native_test_command"]
            and delivery.get("native_test_log", {}).get("path") == plan["native_test_log_path"], "V2 native command/log path")
    native_receipt(read_json(checked_declared(delivery["native_test_receipt"])), count=22,
                   plan_sha256=plan_ref["sha256"], sources=plan["source_files"], tests=plan["test_sources"],
                   command=plan["native_test_command"], log_ref=delivery["native_test_log"])
    for arm in plan["arms"]:
        ck = arm["checkpoint"]
        require((ck["recipe"], ck["seed"], ck["role"]) == (arm["recipe"], 0, "selected"), "Checkpoint role/seed/recipe")
        require(arm["audio_route"] == "recipient_audio_delivery"
                and arm["non_primary_donor_route"] is (arm["recipe"] == "S_VR_P0_Ashuf"), "Per-arm delivery route")
        checked_declared(ck); checked_declared(ck["training_run_receipt"])
    check_sources(plan["source_files"]); check_sources(plan["test_sources"])
    for ref in plan["inputs"].values(): checked_declared(ref)
    require(authorization.get("output") == policy["output"], "Unbound audit output")
    return plan, plan_ref, delivery_ref, policy_ref, authorization_ref, policy


def publish(path, value):
    path = canonical(path)
    partial = path.with_name(path.name + ".partial")
    require(path.parent.is_dir() and not path.exists() and not partial.exists(), "Refuse overwrite/resume")
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 256 * 1024 and shutil.disk_usage(path.parent).free - len(data) >= GIB,
            "Auditor receipt size/free floor")
    with partial.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit",))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    plan, plan_ref, delivery_ref, policy_ref, authorization_ref, policy = authorize(args.policy, args.authorization)
    result = audit_four(plan, plan_ref, delivery_ref)
    for ref in (plan_ref, delivery_ref, policy_ref, authorization_ref): checked_declared(ref)
    check_sources(policy["source_files"]); check_sources(policy["test_sources"])
    check_sources(plan["source_files"]); check_sources(plan["test_sources"])
    for ref in plan["inputs"].values(): checked_declared(ref)
    result.update(policy=policy_ref, auditor_authorization=authorization_ref,
                  auditor_source_files=policy["source_files"], auditor_test_sources=policy["test_sources"])
    publish(policy["output"], result)
    print(json.dumps({"status": "passed", "receipt": reference(policy["output"])}), flush=True)


if __name__ == "__main__":
    main()
