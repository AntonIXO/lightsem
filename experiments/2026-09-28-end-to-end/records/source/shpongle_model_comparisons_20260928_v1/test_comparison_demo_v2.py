"""Synthetic checks only. No real checkpoint/data reads, CUDA, render or vault.

Run in the dedicated project environment after parent binds this file's hash.
"""
import copy
from dataclasses import make_dataclass
import importlib.util
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comparison_demo_v2", HERE / "run_comparison_demo_v2.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def synthetic_arm(recipe):
    return {"recipe": recipe, "subdir": recipe, "audio_route": "recipient_audio_delivery",
            "non_primary_donor_route": recipe == "S_VR_P0_Ashuf", "checkpoint_key": recipe,
            "checkpoint": {"path": "synthetic-not-read.pt", "sha256": "unused", "recipe": recipe,
                           "seed": 0, "role": "selected", "bindings": {"example_hash": "abc"}}}


def synthetic_metadata(recipe):
    def cfg(width):
        return {"d_model": width, "in_dim": 94, "palette_size": 32, "visual_aux_dim": demo.AUX[recipe],
                "pose_aux_dim": 0, "max_frames40": 12000}
    return {"model_config": cfg(demo.WIDTHS[recipe]), "reference_model_config": cfg(128),
            "config": {"recipe": recipe, "seed": 0}, "example_hash": "abc",
            "architecture": "independent" if recipe == "I_VR_P0" else "shared"}


def synthetic_plan():
    return {"schema_version": 2, "arms": [synthetic_arm(r) for r in demo.RECIPES],
            "scientific_cap_bytes": demo.CAP, "derived_reservation_bytes": demo.RESERVATION,
            "minimum_free_bytes": demo.GIB, "audio_route": "recipient_audio_delivery",
            "anchor_inference_allowed": False, "mapping_id": "sep22_fixed_overlay_v1"}


class FactoryAndBindings(unittest.TestCase):
    def test_four_recipes_fixed_order_and_anchor_absent(self):
        demo.validate_plan(synthetic_plan())
        changed = synthetic_plan(); changed["arms"].reverse()
        with self.assertRaises(demo.Blocked): demo.validate_plan(changed)
        changed = synthetic_plan(); changed["arms"][0] = synthetic_arm("S_VR_P0")
        with self.assertRaises(demo.Blocked): demo.validate_plan(changed)

    def test_seed_role_and_audio_route_fail_closed(self):
        for field, bad in (("seed", 1), ("role", "final")):
            plan = synthetic_plan(); plan["arms"][0]["checkpoint"][field] = bad
            with self.assertRaises(demo.Blocked): demo.validate_plan(plan)
        plan = synthetic_plan(); plan["arms"][-1]["audio_route"] = "tempo_donor"
        with self.assertRaises(demo.Blocked): demo.validate_plan(plan)
        plan = synthetic_plan(); plan["arms"][-1]["non_primary_donor_route"] = False
        with self.assertRaises(demo.Blocked): demo.validate_plan(plan)
        plan = synthetic_plan(); plan["anchor_inference_allowed"] = True
        with self.assertRaises(demo.Blocked): demo.validate_plan(plan)

    def test_source_hash_changed_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"; source.write_text("fixed")
            bound = {str(source): demo.digest(source)}
            demo.source_checks(bound)
            source.write_text("changed")
            with self.assertRaises(demo.Blocked): demo.source_checks(bound)

    def test_all_four_canonical_loader_and_fake_forward_dimensions(self):
        for recipe in demo.RECIPES:
            arm = synthetic_arm(recipe); metadata = synthetic_metadata(recipe); calls = []
            Config = make_dataclass("SyntheticConfig", [(k, int) for k in metadata["model_config"]])
            class FakeModel:
                training = False
                cfg = Config(**metadata["model_config"])
                def __call__(self, audio):
                    calls.append(tuple(audio.shape))
                    return {"context10": np.zeros((1, audio.shape[1]//4, self.cfg.d_model), np.float32)}
            def fake_loader(path, *, device, expected_bindings):
                self.assertEqual(path, "synthetic-not-read.pt")
                self.assertEqual(device, "cpu")
                self.assertEqual(expected_bindings, {"example_hash": "abc"})
                return FakeModel(), metadata
            model, loaded = demo.load_bound_model(arm, fake_loader, "cpu")
            output = model(np.zeros((1, 80, 94), np.float32))  # fake forward only
            self.assertEqual(output["context10"].shape, (1, 20, demo.WIDTHS[recipe]))
            self.assertEqual(calls, [(1, 80, 94)])
            self.assertEqual(loaded["reference_model_config"]["d_model"], 128)

    def test_independent_double_narrow_and_binding_rejected(self):
        arm = synthetic_arm("I_VR_P0"); metadata = synthetic_metadata("I_VR_P0")
        metadata["reference_model_config"]["d_model"] = 76
        with self.assertRaises(demo.Blocked): demo.validate_metadata(metadata, arm)
        metadata = synthetic_metadata("I_VR_P0"); metadata["model_config"]["d_model"] = 44
        with self.assertRaises(demo.Blocked): demo.validate_metadata(metadata, arm)
        metadata = synthetic_metadata("I_VR_P0"); metadata["example_hash"] = "wrong"
        with self.assertRaises(demo.Blocked): demo.validate_metadata(metadata, arm)


class ClockAndStorage(unittest.TestCase):
    def test_clock_shape_type_and_quantization(self):
        raw = np.zeros((11092, 94), np.float32); baseline = np.zeros((11092, 512), np.uint8)
        times = np.arange(11092) / 40; bus = times.astype(np.float32)
        demo.validate_clock(raw, raw.copy(), baseline, times, bus)
        bad = times.copy(); bad[100] += .025
        with self.assertRaises(demo.Blocked): demo.validate_clock(raw, raw, baseline, bad, bus)
        with self.assertRaises(demo.Blocked): demo.validate_clock(raw, raw, baseline, times, times)
        with self.assertRaises(demo.Blocked): demo.validate_clock(raw[:-1], raw, baseline, times, bus)
        badraw = raw.copy(); badraw[0, 0] = np.nan
        with self.assertRaises(demo.Blocked): demo.validate_clock(badraw, raw, baseline, times, bus)

    def test_reserve_all_four_against_original_cap_and_floor(self):
        receipt = demo.space_values(952976108, 0, 0, 1254000000)
        self.assertEqual(receipt["reserved_final_combined_bytes"], 986530540)
        self.assertEqual(receipt["remaining_reserved_bytes"], 33554432)
        # Consumed reservation remains part of the same total; no per-arm reset.
        later = demo.space_values(952976108, 10000000, 10000000, 1254000000 - 10000000)
        self.assertEqual(later["reserved_final_combined_bytes"], 986530540)
        self.assertLessEqual(demo.TOTAL_BOUND, demo.RESERVATION)
        with self.assertRaises(demo.Blocked): demo.space_values(demo.CAP, 0, 0, 2*demo.GIB)
        with self.assertRaises(demo.Blocked): demo.space_values(100, 0, 0, demo.GIB + demo.RESERVATION - 1)
        with self.assertRaises(demo.Blocked): demo.space_values(50000000, demo.RESERVATION, demo.RESERVATION, 2*demo.GIB, pending=1)
        with self.assertRaises(demo.Blocked): demo.space_values(1, 1, 2, 2*demo.GIB)

    def test_exclusive_atomic_writes_and_no_cap_override(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory); derived = campaign / "derived"; derived.mkdir()
            plan = {"campaign_root": str(campaign), "derived_root": str(derived)}
            # Inject only synthetic accounting; do not depend on real free disk.
            guard = demo.guard
            demo.guard = lambda _plan, pending=0: demo.space_values(0, 0, 0, 2*demo.GIB, pending)
            try:
                path = derived / "receipt.json"
                demo.exclusive_bytes(path, b"{}", plan)
                self.assertEqual(path.read_bytes(), b"{}")
                self.assertFalse(path.with_name(path.name + ".partial").exists())
                with self.assertRaises(demo.Blocked): demo.exclusive_bytes(path, b"x", plan)
                with self.assertRaises(demo.Blocked): demo.exclusive_bytes(campaign / "outside", b"x", plan)
            finally:
                demo.guard = guard
        plan = synthetic_plan(); plan["derived_reservation_bytes"] += 1
        with self.assertRaises(demo.Blocked): demo.validate_plan(plan)


class Mapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imports canonical software helpers, never weights or observed datasets.
        cls.helpers = demo.load_helpers()
        cls.rig = cls.helpers["canonical_rig"]()

    def synthetic(self, cue=False):
        n = 80; t = np.arange(n)/40; slow = n//4
        baseline = np.full((n, 512), 80, np.uint8)
        allowed, rgb, dim, angles = demo.independent_slots(self.rig)
        baseline[:, dim[0]] = 0
        ids = np.zeros(slow, np.int64); cues = np.zeros(slow, bool)
        if cue: ids[5:] = 1; cues[5] = True
        previous = np.r_[np.int64(32), ids[:-1]]
        color_score = np.zeros(slow, np.float32)
        if cue: color_score[5] = .75
        actions = np.array([0, 5] if cue else [0], np.int64)
        choices = np.zeros((len(actions), 33), np.uint8)
        if cue: choices[1] = 1
        action_logits = np.zeros((len(actions), 32), np.float32)
        action_logits[0, 0] = 1.
        if cue: action_logits[1, 1] = 1.
        scores = {"accent_logit": np.zeros(n, np.float32), "chromatic_change_logit": np.zeros(slow, np.float32),
                  "accent_score": np.zeros(n, np.float32), "color_score": color_score,
                  "context10": np.zeros((slow, 128), np.float32), "generated_palette_ids": ids,
                  "previous_ids": previous, "color_cues": cues, "final_ids": np.asarray(ids[-1], np.int64),
                  "action_indices": actions, "action_previous_ids": previous[actions].copy(),
                  "action_logits": action_logits, "action_argmax_choices": choices,
                  "time40_s": t, "time10_s": np.arange(slow)/10}
        prototypes = np.zeros((32, 72)); prototypes[:, 5] = 1; prototypes[1] = 0; prototypes[1, 41] = 1
        return baseline, t, scores, prototypes

    def test_palette_primary_and_zero_event_exact_identity(self):
        p = np.zeros((1, 72)); p[0, 5] = 1
        rgb, _, _ = demo.palette_rgb(p)
        np.testing.assert_allclose(rgb, [[1, 0, 0]], atol=1e-15)
        baseline, t, scores, prototypes = self.synthetic()
        output = demo.map_outputs(scores, baseline, t, prototypes, self.helpers)
        np.testing.assert_array_equal(output["overlay"], baseline)
        self.assertEqual(len(output["accent_times_s"]), 0)
        self.assertEqual(len(output["color_cue_times_s"]), 0)
        self.assertFalse(output["color_active40"].any())

    def test_first_actual_cue_and_accent_byte_parity(self):
        baseline, t, scores, prototypes = self.synthetic(cue=True)
        scores["accent_score"][40] = .75
        output = demo.map_outputs(scores, baseline, t, prototypes, self.helpers)
        proof = demo.verify_mapping(scores, output, baseline, self.rig, prototypes, self.helpers)
        self.assertEqual(proof["protected_slots_count"], 469)
        self.assertEqual(proof["changed_protected_bytes"], 0)
        self.assertEqual(proof["model_forward_calls"], 0)
        self.assertEqual(output["color_cue_times_s"].tolist(), [.5])
        allowed, _, dim, _ = demo.independent_slots(self.rig)
        np.testing.assert_array_equal(output["overlay"][:, ~allowed], baseline[:, ~allowed])
        self.assertTrue((output["overlay"][:, dim[0]] == 0).all())
        self.assertLessEqual(float(output["accent_pulse40"].max()), .35000001)

    def test_disallowed_mutation_and_invented_cue_rejected(self):
        baseline, t, scores, prototypes = self.synthetic()
        output = demo.map_outputs(scores, baseline, t, prototypes, self.helpers)
        bad = {k: v.copy() for k, v in output.items()}
        allowed, _, _, _ = demo.independent_slots(self.rig)
        bad["overlay"][0, np.flatnonzero(~allowed)[0]] ^= 1
        with self.assertRaises(demo.Blocked): demo.verify_mapping(scores, bad, baseline, self.rig, prototypes, self.helpers)
        bad = {k: v.copy() for k, v in output.items()}; bad["color_active40"][0] = True
        with self.assertRaises(demo.Blocked): demo.verify_mapping(scores, bad, baseline, self.rig, prototypes, self.helpers)
        wrong = copy.deepcopy(scores); wrong["previous_ids"][0] = 0
        with self.assertRaises(demo.Blocked): demo.verify_mapping(wrong, output, baseline, self.rig, prototypes, self.helpers)
        wrong = copy.deepcopy(scores); wrong["time40_s"] = wrong["time40_s"] + .025
        bad = {k: v.copy() for k, v in output.items()}; bad["time"] += .025
        with self.assertRaises(demo.Blocked): demo.verify_mapping(wrong, bad, baseline, self.rig, prototypes, self.helpers)

    def test_falsified_events_cues_rgb_ids_final_state_and_shapes_rejected(self):
        baseline, t, scores, prototypes = self.synthetic(cue=True)
        scores["accent_score"][40] = .75
        output = demo.map_outputs(scores, baseline, t, prototypes, self.helpers)
        for field in ("accent_times_s", "accent_confidences", "palette_rgb", "palette_hsv", "palette_complex_z"):
            bad = {k: v.copy() for k, v in output.items()}
            if field == "accent_times_s": bad[field][0] += .025
            elif field == "accent_confidences": bad[field][0] = .6
            elif field == "palette_complex_z": bad[field][0] += .1j
            else: bad[field][0, 0] = .25
            with self.assertRaises(demo.Blocked):
                demo.verify_mapping(scores, bad, baseline, self.rig, prototypes, self.helpers)
        variants = []
        wrong = copy.deepcopy(scores); wrong["color_cues"][6] = True; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["generated_palette_ids"][5] = 2; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["final_ids"] = np.asarray(0, np.int64); variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["action_logits"][1] = 0; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["action_argmax_choices"][1, 0] = 2; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["action_previous_ids"][1] = 3; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["action_indices"][1] = 6; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["accent_score"] = wrong["accent_score"].astype(np.float64); variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["context10"] = wrong["context10"][:, :76]; variants.append(wrong)
        wrong = copy.deepcopy(scores); wrong["color_score"][0] = 1.5; variants.append(wrong)
        for wrong in variants:
            with self.assertRaises(demo.Blocked):
                demo.verify_mapping(wrong, output, baseline, self.rig, prototypes, self.helpers, expected_width=128)
        bad = {k: v.copy() for k, v in output.items()}; bad["palette_ids40"] = bad["palette_ids40"].astype(np.int32)
        with self.assertRaises(demo.Blocked):
            demo.verify_mapping(scores, bad, baseline, self.rig, prototypes, self.helpers)

    def test_saved_plan_authorization_source_input_and_route_bindings(self):
        arm = synthetic_arm("S_VR_P0_Ashuf")
        plan = {"source_files": {"source": "hash"}, "inputs": {"audio": {"path": "x", "sha256": "y"}}}
        plan_ref, binding_ref = {"path": "plan", "sha256": "p"}, {"path": "binding", "sha256": "b"}
        inference = {"recipe": arm["recipe"], "checkpoint": arm["checkpoint"], "plan": plan_ref,
                     "authorization": binding_ref, "source_files": plan["source_files"], "inputs": plan["inputs"],
                     "audio_route": "recipient_audio_delivery", "ashuf_not_primary_donor_evaluation": True}
        demo.validate_saved_bindings(inference, plan, arm, plan_ref, binding_ref)
        for field in ("recipe", "checkpoint", "plan", "authorization", "source_files", "inputs", "audio_route", "ashuf_not_primary_donor_evaluation"):
            bad = copy.deepcopy(inference); bad[field] = None
            with self.assertRaises(demo.Blocked): demo.validate_saved_bindings(bad, plan, arm, plan_ref, binding_ref)


class WorkspaceAccountingV2(unittest.TestCase):
    def test_other_workspace_artifacts_consume_shared_cap(self):
        receipt = demo.space_values(952976108, 1000000, 0, 1254000000)
        self.assertEqual(receipt["reserved_final_combined_bytes"], 987530540)
        with self.assertRaises(demo.Blocked):
            demo.space_values(952976108, 70000000, 1000000, 1254000000)

    def test_disjoint_same_volume_roots_and_nested_alias_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); original = root / "original"; original.mkdir()
            workspace = root / "workspace"; derived = workspace / "comparison"
            got = demo.validate_accounting_roots(original, workspace, derived)
            self.assertEqual(got, (original, workspace, derived))
            self.assertFalse(workspace.exists())  # validation must not create directories
            with self.assertRaises(demo.Blocked):
                demo.validate_accounting_roots(original, original / "nested", original / "nested/comparison")
            with self.assertRaises(demo.Blocked):
                demo.validate_accounting_roots(original, original, original / "comparison")
            alias = root / "alias"; alias.symlink_to(original, target_is_directory=True)
            with self.assertRaises(demo.Blocked):
                demo.validate_accounting_roots(original, alias, alias / "comparison")
            with patch.object(demo, "device_for", side_effect=[1, 2, 2]):
                with self.assertRaises(demo.Blocked):
                    demo.validate_accounting_roots(original, workspace, derived)

    def test_accounted_tree_rejects_child_symlink_and_guard_wrong_fixed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); (root / "data").write_bytes(b"abc")
            self.assertEqual(demo.size_tree(root), 3)
            (root / "alias").symlink_to(root / "data")
            with self.assertRaises(demo.Blocked): demo.size_tree(root)
            wrong = {"campaign_root": str(root), "workspace_root": str(root / "new"),
                     "derived_root": str(root / "new/comparison")}
            with self.assertRaises(demo.Blocked): demo.guard(wrong)


class FinitePaletteV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.history = staticmethod(demo.load_helpers()["generated_history"])

    def head(self, injected=None):
        torch = self.torch
        class FakeHead:
            cfg = SimpleNamespace(palette_size=32)
            calls = 0
            def palette_logits(self, context, ids):
                self.calls += 1
                logits = torch.zeros((*ids.shape, 32), dtype=context.dtype)
                logits[..., 0] = 1.
                if injected is not None:
                    logits[..., 5] = torch.where(ids == 7, torch.tensor(injected), logits[..., 5])
                return logits
        return FakeHead()

    def inputs(self):
        torch = self.torch
        context = torch.zeros((1, 20, 8))
        cues = torch.full((1, 20), -20.)
        cues[:, 5] = 20.
        return context, cues

    def test_unchanged_finite_case_no_extra_head_pass_or_selection_change(self):
        context, cues = self.inputs()
        a, b = self.head(), self.head()
        raw = self.history(a, context, cues, initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
        checked = self.history(demo.FinitePaletteProxy(b), context, cues,
                               initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
        self.assertEqual(a.calls, b.calls)
        for key in raw: self.assertTrue(self.torch.equal(raw[key], checked[key]), key)
        before = b.calls
        direct = b.palette_logits(context, checked["previous_ids"])
        through = demo.FinitePaletteProxy(b).palette_logits(context, checked["previous_ids"])
        self.assertEqual(b.calls, before + 2)
        self.assertTrue(self.torch.equal(direct, through))

    def test_unvisited_state_nan_and_inf_fail_before_argmax(self):
        context, cues = self.inputs()
        for bad in (float("nan"), float("inf"), -float("inf")):
            unchecked_head = self.head(bad)
            # Demonstrate the precise gap: generated states never visit ID7;
            # visited logits are finite although an all-state row was nonfinite.
            raw = self.history(unchecked_head, context, cues, initial_ids=None,
                               state_ablation=False, cue_threshold=.5, initialize=True)
            self.assertFalse(bool((raw["previous_ids"] == 7).any()))
            self.assertTrue(bool(self.torch.isfinite(unchecked_head.palette_logits(context, raw["previous_ids"])).all()))
            checked_head = self.head(bad)
            with self.assertRaises(demo.Blocked):
                self.history(demo.FinitePaletteProxy(checked_head), context, cues,
                             initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
            self.assertEqual(checked_head.calls, 1)

    def test_capture_actual_all_state_table_once_without_rebatched_authority(self):
        context, cues = self.inputs()
        head = self.head()
        proxy = demo.FinitePaletteProxy(head, capture_actions=True)
        rollout = self.history(proxy, context, cues, initial_ids=None,
                               state_ablation=False, cue_threshold=.5, initialize=True)
        self.assertEqual(head.calls, 1)
        self.assertEqual(proxy.head_calls, 1)
        self.assertEqual(proxy.action_tables.shape, (2, 33, 32))
        self.assertEqual(proxy.action_tables.dtype, np.float32)
        actual_states = rollout["previous_ids"][0, [0, 5]].numpy()
        selected = proxy.action_tables[np.arange(2), actual_states]
        choices = proxy.action_tables.argmax(-1).astype(np.uint8)
        np.testing.assert_array_equal(selected.argmax(-1), choices[np.arange(2), actual_states])
        # No separately rebatched full-T palette head was evaluated to validate argmax.
        self.assertEqual(head.calls, 1)

    def test_visited_logits_and_prerollout_context_cues_fail_closed(self):
        context, _ = self.inputs()
        for bad in (float("nan"), float("inf")):
            head = self.head(bad)
            with self.assertRaises(demo.Blocked):
                demo.FinitePaletteProxy(head).palette_logits(context, self.torch.full((1, 20), 7))
            for key in ("accent_logit", "chromatic_change_logit", "context10"):
                output = {"accent_logit": self.torch.zeros((1, 80)),
                          "chromatic_change_logit": self.torch.zeros((1, 20)),
                          "context10": self.torch.zeros((1, 20, 8))}
                demo.validate_forward_outputs(output, 8, frames40=80)
                output[key].reshape(-1)[0] = bad
                with self.assertRaises(demo.Blocked): demo.validate_forward_outputs(output, 8, frames40=80)


class AuthorizationV2(unittest.TestCase):
    def make_files(self, root):
        source = root / "helper.py"; source.write_text("fixed synthetic helper")
        test = root / "test.py"; test.write_text("synthetic test source")
        sources = {str(Path(demo.__file__).resolve()): demo.digest(demo.__file__), str(source): demo.digest(source)}
        tests = {str(test): demo.digest(test)}
        log = root / "tests.log"; log.write_text("synthetic native test log")
        command = "synthetic explicitly bound native command"
        plan = synthetic_plan(); plan.update(source_files=sources, test_sources=tests,
                                            native_test_command=command, native_test_log_path=str(log))
        plan_path = root / "plan.json"; plan_path.write_text(json.dumps(plan))
        receipt = {"status": "passed", "source_files": sources, "test_sources": tests,
                   "real_model_forward_calls": 0, "sealed_TEST_reads": 0, "exit_code": 0,
                   "plan_sha256": demo.digest(plan_path), "command": command, "log": demo.reference(log)}
        receipt_path = root / "receipt.json"; receipt_path.write_text(json.dumps(receipt))
        binding = {"status": "authorized_delivery", "phases": ["preflight", "infer", "verify"],
                   "plan_sha256": demo.digest(plan_path), "source_files": sources, "test_sources": tests,
                   "native_test_receipt": demo.reference(receipt_path),
                   "native_test_command": command, "native_test_log": demo.reference(log)}
        binding_path = root / "binding.json"; binding_path.write_text(json.dumps(binding))
        return plan_path, binding_path, receipt_path, source, binding, receipt

    def attempt(self, plan_path, binding_path, phase):
        # Authorization tests do not inspect any actual campaign path or data.
        with patch.object(demo, "guard", return_value={"synthetic": True}):
            return demo.authorize(plan_path, binding_path, phase)

    def test_valid_synthetic_authorization_then_wrong_phase_plan_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, binding_path, receipt_path, source, binding, receipt = self.make_files(root)
            self.attempt(plan_path, binding_path, "preflight")
            self.attempt(plan_path, binding_path, "infer")
            bad = copy.deepcopy(binding); bad["phases"] = ["preflight"]
            binding_path.write_text(json.dumps(bad))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")
            bad = copy.deepcopy(binding); bad["plan_sha256"] = "0" * 64
            binding_path.write_text(json.dumps(bad))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "preflight")
            bad = copy.deepcopy(binding); bad["source_files"] = {}
            binding_path.write_text(json.dumps(bad))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "preflight")
            binding_path.write_text(json.dumps(binding)); source.write_text("changed")
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "preflight")

    def test_native_receipt_hash_status_maps_and_execution_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, binding_path, receipt_path, source, binding, receipt = self.make_files(root)
            bad_binding = copy.deepcopy(binding); bad_binding["native_test_receipt"]["sha256"] = "0" * 64
            binding_path.write_text(json.dumps(bad_binding))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")
            bad_binding = copy.deepcopy(binding); bad_binding.pop("native_test_receipt")
            binding_path.write_text(json.dumps(bad_binding))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")
            for key, bad in (("status", "failed"), ("source_files", {}), ("test_sources", {}),
                             ("real_model_forward_calls", 1), ("sealed_TEST_reads", 1), ("exit_code", 1),
                             ("plan_sha256", "wrong"), ("command", "different command"), ("log", {})):
                changed = copy.deepcopy(receipt); changed[key] = bad
                receipt_path.write_text(json.dumps(changed))
                current = copy.deepcopy(binding); current["native_test_receipt"] = demo.reference(receipt_path)
                binding_path.write_text(json.dumps(current))
                with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")
            receipt_path.write_text(json.dumps(receipt))
            current = copy.deepcopy(binding); current["native_test_receipt"] = demo.reference(receipt_path)
            current["native_test_command"] = "unbound command"
            binding_path.write_text(json.dumps(current))
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")
            current = copy.deepcopy(binding); current["native_test_receipt"] = demo.reference(receipt_path)
            binding_path.write_text(json.dumps(current))
            Path(receipt["log"]["path"]).write_text("tampered log")
            with self.assertRaises(demo.Blocked): self.attempt(plan_path, binding_path, "infer")


if __name__ == "__main__":
    unittest.main()
