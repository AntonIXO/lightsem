"""Synthetic checks only. No real checkpoint/data reads, CUDA, render or vault.

Run in the dedicated project environment after parent binds this file's hash.
"""
import copy
from dataclasses import make_dataclass
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comparison_demo_v1", HERE / "run_comparison_demo_v1.py")
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
    return {"schema_version": 1, "arms": [synthetic_arm(r) for r in demo.RECIPES],
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
        receipt = demo.space_values(952976108, 0, 1254000000)
        self.assertEqual(receipt["reserved_final_campaign_bytes"], 986530540)
        self.assertEqual(receipt["remaining_reserved_bytes"], 33554432)
        # Consumed reservation remains part of the same total; no per-arm reset.
        later = demo.space_values(952976108 + 10000000, 10000000, 1254000000 - 10000000)
        self.assertEqual(later["reserved_final_campaign_bytes"], 986530540)
        self.assertLessEqual(demo.TOTAL_BOUND, demo.RESERVATION)
        with self.assertRaises(demo.Blocked): demo.space_values(demo.CAP, 0, 2*demo.GIB)
        with self.assertRaises(demo.Blocked): demo.space_values(100, 0, demo.GIB + demo.RESERVATION - 1)
        with self.assertRaises(demo.Blocked): demo.space_values(50000000, demo.RESERVATION, 2*demo.GIB, pending=1)
        with self.assertRaises(demo.Blocked): demo.space_values(1, 2, 2*demo.GIB)

    def test_exclusive_atomic_writes_and_no_cap_override(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory); derived = campaign / "derived"; derived.mkdir()
            plan = {"campaign_root": str(campaign), "derived_root": str(derived)}
            # Inject only synthetic accounting; do not depend on real free disk.
            guard = demo.guard
            demo.guard = lambda _plan, pending=0: demo.space_values(0, 0, 2*demo.GIB, pending)
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
        scores = {"accent_score": np.zeros(n, np.float32), "context10": np.zeros((slow, 128), np.float32),
                  "palette_logits": np.zeros((slow, 32), np.float32), "generated_palette_ids": ids,
                  "previous_ids": previous, "color_cues": cues, "time40_s": t, "time10_s": np.arange(slow)/10}
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
        proof = demo.verify_mapping(scores, output, baseline, self.rig)
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
        with self.assertRaises(demo.Blocked): demo.verify_mapping(scores, bad, baseline, self.rig)
        bad = {k: v.copy() for k, v in output.items()}; bad["color_active40"][0] = True
        with self.assertRaises(demo.Blocked): demo.verify_mapping(scores, bad, baseline, self.rig)
        wrong = copy.deepcopy(scores); wrong["previous_ids"][0] = 0
        with self.assertRaises(demo.Blocked): demo.verify_mapping(wrong, output, baseline, self.rig)
        wrong = copy.deepcopy(scores); wrong["time40_s"] = wrong["time40_s"] + .025
        bad = {k: v.copy() for k, v in output.items()}; bad["time"] += .025
        with self.assertRaises(demo.Blocked): demo.verify_mapping(wrong, bad, baseline, self.rig)


if __name__ == "__main__":
    unittest.main()
