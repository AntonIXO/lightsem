"""Synthetic parent-auditor tests. No project/model imports or observed data.

Run only after parent binds this new test and auditor source. All filesystem
fixtures are temporary and synthetic; no checkpoint or campaign path is read.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location("parent_audit_v1", Path(__file__).with_name("parent_audit_v1.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value))
    return audit.reference(path)


class Probability(unittest.TestCase):
    def test_zero_logit_half_score_passes_without_mutation(self):
        logits = np.zeros(3, np.float32); scores = np.full(3, .5, np.float32)
        before = scores.copy()
        result = audit.probability_sanity(logits, scores, 3)
        self.assertEqual(result["maximum_absolute_error"], 0.)
        self.assertFalse(result["exact_CUDA_sigmoid_claim"])
        np.testing.assert_array_equal(scores, before)

    def test_zero_logit_zero_or_three_quarter_score_rejected(self):
        for score in (0., .75):
            with self.assertRaises(audit.Blocked):
                audit.probability_sanity(np.zeros(1, np.float32), np.array([score], np.float32), 1)

    def test_stable_extreme_finite_logits(self):
        logits = np.array([-np.finfo(np.float32).max, -1000, -100, 0, 100, 1000, np.finfo(np.float32).max], np.float32)
        scores = np.array([0, 0, 0, .5, 1, 1, 1], np.float32)
        with np.errstate(over="raise", invalid="raise"):
            result = audit.probability_sanity(logits, scores, 7)
        self.assertLessEqual(result["maximum_absolute_error"], audit.PROBABILITY_ATOL)

    def test_fixed_eight_epsilon_bound_without_tolerance_override(self):
        x = np.zeros(1, np.float32)
        audit.probability_sanity(x, np.array([.5 + audit.PROBABILITY_ATOL], np.float32), 1)
        with self.assertRaises(audit.Blocked):
            audit.probability_sanity(x, np.array([.5 + audit.PROBABILITY_ATOL + 2**-23], np.float32), 1)
        self.assertEqual(audit.PROBABILITY_ATOL, 8 * np.finfo(np.float32).eps)

    def test_nonfinite_range_dtype_and_shape_rejected(self):
        good = np.array([.5], np.float32)
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(audit.Blocked): audit.probability_sanity(np.array([bad], np.float32), good, 1)
            with self.assertRaises(audit.Blocked): audit.probability_sanity(np.zeros(1, np.float32), np.array([bad], np.float32), 1)
        for wrong in (np.array([-1], np.float32), np.array([2], np.float32), good.astype(np.float64), good.reshape(1, 1)):
            with self.assertRaises(audit.Blocked): audit.probability_sanity(np.zeros(1, np.float32), wrong, 1)


class Chain(unittest.TestCase):
    def fixture(self, root):
        plan_ref = {"path": "synthetic-plan", "sha256": "synthetic-p"}
        delivery_ref = {"path": "synthetic-authorization", "sha256": "synthetic-a"}
        plan = {"derived_root": str(root), "source_files": {"synthetic-source": "fixed"},
                "inputs": {"synthetic-audio": {"path": "unread", "sha256": "fixed"}}, "arms": []}
        baseline = np.zeros((audit.FRAMES40, 512), np.uint8)
        allowed = np.zeros(512, bool); allowed[:43] = True
        np.savez_compressed(root / "baseline.npz", companion=baseline)
        plan["inputs"]["baseline"] = audit.reference(root / "baseline.npz")
        array_sha = lambda a: audit.hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        refs = []
        for recipe in audit.RECIPES:
            directory = root / recipe; directory.mkdir()
            arm = {"recipe": recipe, "subdir": recipe, "audio_route": "recipient_audio_delivery",
                   "non_primary_donor_route": recipe == "S_VR_P0_Ashuf",
                   "checkpoint": {"recipe": recipe, "role": "selected", "seed": 0, "path": "never-read.pt", "sha256": "fixed"}}
            plan["arms"].append(arm)
            np.savez_compressed(directory / "model_outputs.npz",
                accent_logit=np.zeros(audit.FRAMES40, np.float32), accent_score=np.full(audit.FRAMES40, .5, np.float32),
                chromatic_change_logit=np.zeros(audit.FRAMES40 // 4, np.float32), color_score=np.full(audit.FRAMES40 // 4, .5, np.float32))
            np.savez_compressed(directory / "overlay_dmx.npz", overlay=baseline, allowed_slot_mask=allowed)
            runtime = {"encoder_forward_calls": 1, "all_state_head_calls": 1, "actual_action_logits_authority": True,
                       "full_T_palette_head_recomputation": False, "no_argmax_tolerance_expansion": True,
                       "float32": True, "autocast": False, "TF32": False, "model_eval": True,
                       "chunking": False, "training": False, "visual_or_pose_input": False}
            inference = {"recipe": recipe, "checkpoint": arm["checkpoint"], "plan": plan_ref, "authorization": delivery_ref,
                         "source_files": plan["source_files"], "inputs": plan["inputs"], "audio_route": arm["audio_route"],
                         "ashuf_not_primary_donor_evaluation": arm["non_primary_donor_route"], "runtime": runtime,
                         "model_outputs": audit.reference(directory / "model_outputs.npz"), "overlay": audit.reference(directory / "overlay_dmx.npz")}
            proof = {"status": "passed", "model_forward_calls": 0, "protected_slots_count": 469,
                     "changed_protected_bytes": 0, "protected_bytes_compared": audit.FRAMES40 * 469,
                     "baseline_full_sha256": array_sha(baseline), "overlay_full_sha256": array_sha(baseline),
                     "baseline_protected_sha256": array_sha(baseline[:, ~allowed]),
                     "overlay_protected_sha256": array_sha(baseline[:, ~allowed])}
            for key in ("exact_RGB_and_dimmer_reconstruction", "no_invented_color_cues", "accent_events_recomputed_from_scores",
                        "color_cues_recomputed_from_scores", "palette_visualization_rebuilt_from_bound_prototypes",
                        "actual_action_logits_and_compact_choices_replayed", "final_state_exact",
                        "no_rebatched_palette_logit_authority", "no_argmax_tolerance_expansion"):
                proof[key] = True
            complete = {"status": "complete", "recipe": recipe, "plan": plan_ref, "authorization": delivery_ref,
                        "inference": save(directory / "inference_saved.json", inference),
                        "proof": save(directory / "protected_byte_equality.json", proof)}
            refs.append(save(directory / "complete.json", complete))
        aggregate = {"status": "complete", "plan": plan_ref, "completion_kind": "INFERENCE_MAPPING_ONLY",
                     "videos_completed": False, "media_audits_completed": False, "anchor_not_rerun": True, "arms": refs}
        save(root / "all_complete.json", aggregate)
        return plan, plan_ref, delivery_ref, refs

    def test_canonical_all_four_chain_returns_readback_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            result = audit.audit_four(plan, p, a)
            self.assertEqual(len(result["arms"]), 4)
            self.assertEqual(result["completion_kind"], "INFERENCE_MAPPING_READBACK_ONLY")
            self.assertFalse(result["videos_completed"])
            self.assertFalse(result["media_audits_completed"])
            self.assertFalse(result["palette_argmax_tolerance_added"])
            self.assertFalse((root / "audit.json").exists())  # readback does not publish midway

    def test_same_hash_wrong_output_or_proof_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            arm = plan["arms"][0]; d = root / arm["subdir"]
            original = audit.read_json(d / "inference_saved.json")
            complete = audit.read_json(d / "complete.json")
            for field in ("model_outputs", "overlay"):
                wrong = copy.deepcopy(original)
                src = Path(wrong[field]["path"]); alias = d / ("alias_" + src.name); alias.write_bytes(src.read_bytes())
                wrong[field] = audit.reference(alias)
                complete["inference"] = save(d / "inference_saved.json", wrong)
                ref = save(d / "complete.json", complete)
                with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, ref)
            complete["inference"] = save(d / "inference_saved.json", original)
            alias = d / "alias_proof.json"; alias.write_bytes((d / "protected_byte_equality.json").read_bytes())
            complete["proof"] = audit.reference(alias)
            with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, save(d / "complete.json", complete))

    def test_changed_bytes_and_wrong_completion_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            arm = plan["arms"][0]; d = root / arm["subdir"]
            alias = d / "alias_complete.json"; alias.write_bytes((d / "complete.json").read_bytes())
            with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, audit.reference(alias))
            with (d / "model_outputs.npz").open("ab") as stream: stream.write(b"tamper")
            with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, refs[0])

    def test_coupled_plan_auth_input_source_checkpoint_recipe_route_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            arm = plan["arms"][-1]; d = root / arm["subdir"]
            original = audit.read_json(d / "inference_saved.json"); complete = audit.read_json(d / "complete.json")
            for field in ("recipe", "checkpoint", "plan", "authorization", "inputs", "source_files", "audio_route", "ashuf_not_primary_donor_evaluation"):
                wrong = copy.deepcopy(original); wrong[field] = None
                complete["inference"] = save(d / "inference_saved.json", wrong)
                with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, save(d / "complete.json", complete))
            complete["inference"] = save(d / "inference_saved.json", original)
            complete["authorization"] = {"wrong": True}
            with self.assertRaises(audit.Blocked): audit.arm_chain(plan, arm, p, a, save(d / "complete.json", complete))

    def test_rehashed_wrong_proof_array_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            d = root / plan["arms"][0]["subdir"]
            proof = audit.read_json(d / "protected_byte_equality.json"); proof["overlay_full_sha256"] = "0" * 64
            complete = audit.read_json(d / "complete.json"); complete["proof"] = save(d / "protected_byte_equality.json", proof)
            aggregate = audit.read_json(root / "all_complete.json"); aggregate["arms"][0] = save(d / "complete.json", complete)
            save(root / "all_complete.json", aggregate)
            with self.assertRaises(audit.Blocked): audit.audit_four(plan, p, a)

    def test_aggregate_order_scope_or_missing_last_arm_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); plan, p, a, refs = self.fixture(root)
            path = root / "all_complete.json"; original = audit.read_json(path)
            wrong = copy.deepcopy(original); wrong["arms"].reverse(); save(path, wrong)
            with self.assertRaises(audit.Blocked): audit.audit_four(plan, p, a)
            wrong = copy.deepcopy(original); wrong["videos_completed"] = True; save(path, wrong)
            with self.assertRaises(audit.Blocked): audit.audit_four(plan, p, a)
            save(path, original); Path(refs[-1]["path"]).unlink()
            with self.assertRaises(audit.Blocked): audit.audit_four(plan, p, a)


class NativeReceipt(unittest.TestCase):
    def fixture(self, root):
        log = root / "native.log"
        log.write_text("\n".join(f"test_{i} (Synthetic) ... ok" for i in range(22)) + "\n\nRan 22 tests in 0.1s\n\nOK\n")
        args = {"count": 22, "plan_sha256": "synthetic", "sources": {"source": "fixed"},
                "tests": {"test": "fixed"}, "command": "bound synthetic command", "log_ref": audit.reference(log)}
        receipt = {"status": "passed", "tests_run": 22, "tests_passed": 22, "failures": 0, "errors": 0, "skipped": 0,
                   "exit_code": 0, "real_model_forward_calls": 0, "real_checkpoint_loads": 0, "sealed_TEST_reads": 0,
                   "plan_sha256": args["plan_sha256"], "source_files": args["sources"], "test_sources": args["tests"],
                   "command": args["command"], "log": args["log_ref"]}
        return receipt, args, log

    def test_native_twenty_two_pass_log_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, args, _ = self.fixture(Path(directory).resolve())
            audit.native_receipt(receipt, **args)

    def test_wrong_counts_skip_failure_error_exit_or_model_reads_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, args, _ = self.fixture(Path(directory).resolve())
            for key, value in (("tests_run", 21), ("tests_passed", 21), ("skipped", 1), ("failures", 1), ("errors", 1),
                               ("exit_code", 1), ("real_model_forward_calls", 1), ("real_checkpoint_loads", 1), ("sealed_TEST_reads", 1)):
                wrong = dict(receipt); wrong[key] = value
                with self.assertRaises(audit.Blocked): audit.native_receipt(wrong, **args)

    def test_bad_log_or_source_command_plan_status_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, args, log = self.fixture(Path(directory).resolve())
            for key in ("source_files", "test_sources", "command", "plan_sha256", "status"):
                wrong = dict(receipt); wrong[key] = "wrong"
                with self.assertRaises(audit.Blocked): audit.native_receipt(wrong, **args)
            for bad in ("Ran 0 tests in 0s\nOK\n", "Ran 22 tests in 0s\nOK (skipped=1)\n", "Ran 22 tests in 0s\nOK\n"):
                log.write_text(bad); args["log_ref"] = audit.reference(log); receipt["log"] = args["log_ref"]
                with self.assertRaises(audit.Blocked): audit.native_receipt(receipt, **args)


class Publication(unittest.TestCase):
    def test_exclusive_receipt_and_retained_partial_block_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); path = root / "receipt.json"
            with patch.object(audit.shutil, "disk_usage", return_value=SimpleNamespace(free=2 * audit.GIB)):
                audit.publish(path, {"status": "synthetic"})
                with self.assertRaises(audit.Blocked): audit.publish(path, {})
                partial = root / "other.json.partial"; partial.write_text("interrupted")
                with self.assertRaises(audit.Blocked): audit.publish(root / "other.json", {})
                self.assertTrue(partial.exists())


if __name__ == "__main__":
    unittest.main()
