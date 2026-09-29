import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lightsem.experiments.stage11.stage11_common import (
    artifact_manifest, authenticated_json, digest, json_digest, read_authenticated,
)
from lightsem.experiments.stage11.stage11_evaluation import (
    CORE_SEEDS, GENERATION_SEEDS, SCENE_ARMS, _compute_aggregate, _context,
    _run_directory, _song_values, _test_unlock, _training_report, _validate_rows, _verify_captures,
    aggregate, evaluate_run,
)


MODULE = "lightsem.experiments.stage11.stage11_evaluation"


def fixture():
    arms = {
        "D_old": {"family": "motion", "population": "old", "steps": 12000},
        "D_full": {"family": "motion", "population": "full", "steps": 12000},
        **{arm: {"family": "scene", "steps": 32000} for arm in SCENE_ARMS},
    }
    protocol = {"arms": arms, "training_seeds": list(CORE_SEEDS),
                "generation_seeds": list(GENERATION_SEEDS), "extension_seeds": [5]}
    data = {
        family: [{"key": f"{family}_{song}", "canonical_id": f"song{song:02d}",
                  "recording_id": f"recording{song}", "partition": "val", "n_frames": 600}
                 for song in range(40)] for family in ("motion", "scene")
    }

    def metrics(ap=0.6, temporal=0.7, marginal=0.09):
        return {"temporal_error": temporal, "marginal_error": marginal,
                "variation_ratio": 0.9, "hf_energy_ratio_2to5": 1.1,
                "novelty": {"ap": ap, "eligible": True, "event_count": 5,
                            "positive_frames": 25, "negative_frames": 555, "valid_frames": 580,
                            "beat_only_ap": 0.1, "circular_shift_mean_ap": 0.1,
                            "circular_shift_aps": [0.1] * 200}}

    reports = {}
    for arm, cfg in arms.items():
        for seed in CORE_SEEDS:
            rows = []
            for index, entry in enumerate(data[cfg["family"]]):
                base = {"key": entry["key"], "canonical_id": entry["canonical_id"],
                        "recording_id": entry["recording_id"], "n_frames": entry["n_frames"]}
                if cfg["family"] == "motion":
                    rows.append({**base, "motion": {"primary_r2": 0.1 if arm == "D_old" else 0.13}, "events": {}})
                else:
                    for gs in GENERATION_SEEDS:
                        result = metrics(ap=0.6 if arm == "S_temporal" else 0.2,
                                         temporal=1.0 if arm == "S_pointwise" else 0.7,
                                         marginal=0.1 if arm == "S_pointwise" else 0.09)
                        row = {**base, "generation_seed": gs, "metrics": result}
                        if arm == "S_temporal":
                            row["donors"] = [{"key": f"scene_{(index + shift) % 40}", "n_frames": 600,
                                               "matched_metrics": metrics(), "donor_metrics": metrics(ap=0.2)}
                                              for shift in range(1, 7)]
                        rows.append(row)
            reports[(arm, seed)] = {"rows": rows}
    arm_seeds = {arm: list(CORE_SEEDS) for arm in arms}
    return protocol, data, reports, arm_seeds


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.data, cls.reports, cls.arm_seeds = fixture()

    def test_complete_positive_matrix_reports_real_seed_song_denominators(self):
        result = _compute_aggregate(self.protocol, self.data, self.reports, self.arm_seeds, "val")
        self.assertEqual(result["eligible_families"], ["motion", "scene"])
        self.assertTrue(all(gate["passed"] for gate in result["gates"].values()))
        comparison = result["comparisons"]["H11_T"]
        self.assertEqual(comparison["n_training_seeds"], 5)
        self.assertEqual(comparison["n_songs"], 40)
        self.assertEqual(comparison["n_paired_cells"], 200)
        self.assertAlmostEqual(comparison["delta"], 0.3)
        counts = result["metric_denominators"]["S_temporal"]["0"]["novelty_ap"]
        self.assertEqual(counts["attempted_rows"], 160)
        self.assertEqual(counts["valid_songs"], 40)
        self.assertEqual(result["comparisons"]["conditioning_control_policy"]["circular_shift_statistic"],
                         "mean_of_200_predeclared_shifts")

    def test_generation_draws_and_multicoach_recordings_do_not_become_independent_songs(self):
        entries = [{"key": "r1a", "canonical_id": "song", "recording_id": "r1"},
                   {"key": "r1b", "canonical_id": "song", "recording_id": "r1"},
                   {"key": "r2a", "canonical_id": "song", "recording_id": "r2"}]
        rows = [{"key": key, "value": value} for key, value in (("r1a", 0), ("r1b", 0), ("r2a", 1))]
        values, counts = _song_values(rows, entries, lambda row: row["value"])
        self.assertEqual(values["song"], 0.5)
        self.assertEqual(counts["valid_recordings"], 2)
        self.assertEqual(counts["valid_songs"], 1)
        expanded = [{**row, "generation_seed": seed} for row in rows for seed in GENERATION_SEEDS]
        repeated, _ = _song_values(expanded, entries, lambda row: row["value"])
        self.assertEqual(values, repeated)

    def test_one_failed_conditioning_control_keeps_scene_family_closed(self):
        reports = copy.deepcopy(self.reports)
        for seed in CORE_SEEDS:
            for row in reports[("S_swapped", seed)]["rows"]:
                row["metrics"]["novelty"]["ap"] = 0.59
        result = _compute_aggregate(self.protocol, self.data, reports, self.arm_seeds, "val")
        self.assertTrue(result["gates"]["H11_T"]["passed"])
        self.assertFalse(result["gates"]["H11_A"]["passed"])
        self.assertFalse(result["gates"]["H11_A"]["checks"]["shuffled_training"])
        self.assertEqual(result["eligible_families"], ["motion"])

    def test_all_controls_use_same_song_intersection_and_fail_low_coverage(self):
        reports = copy.deepcopy(self.reports)
        for seed in CORE_SEEDS:
            for row in reports[("S_temporal", seed)]["rows"]:
                if row["canonical_id"] in {f"song{i:02d}" for i in range(9)}:
                    for donor in row["donors"]:
                        for side in ("matched_metrics", "donor_metrics"):
                            donor[side]["novelty"]["eligible"] = False
                            donor[side]["novelty"]["ap"] = None
        result = _compute_aggregate(self.protocol, self.data, reports, self.arm_seeds, "val")
        self.assertFalse(result["gates"]["H11_A"]["passed"])
        self.assertEqual(result["eligibility"]["H11_A"]["eligible"], 31)
        comparisons = result["comparisons"]["H11_A"]
        self.assertEqual(len({tuple(value["canonical_ids"]) for value in comparisons.values()}), 1)
        self.assertTrue(all(value["n_songs"] == 31 for value in comparisons.values()))

    def test_zero_eligible_rows_cannot_produce_a_successful_gate(self):
        reports = copy.deepcopy(self.reports)
        for seed in CORE_SEEDS:
            for arm in ("D_old", "D_full"):
                for row in reports[(arm, seed)]["rows"]:
                    row["motion"]["primary_r2"] = None
        result = _compute_aggregate(self.protocol, self.data, reports, self.arm_seeds, "val")
        self.assertFalse(result["gates"]["H11_D"]["passed"])
        self.assertIsNone(result["comparisons"]["H11_D"])
        self.assertEqual(result["eligibility"]["H11_D"]["eligible"], 0)
        self.assertEqual(len(result["eligibility"]["H11_D"]["excluded_canonical_ids"]), 40)

    def test_missing_or_duplicate_generation_rows_are_rejected(self):
        rows = self.reports[("S_pointwise", 0)]["rows"]
        with self.assertRaisesRegex(RuntimeError, "missing 1 rows"):
            _validate_rows(rows[:-1], self.data["scene"], "scene")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            _validate_rows(rows + [rows[0]], self.data["scene"], "scene")

    def test_donor_support_mismatch_is_rejected(self):
        rows = copy.deepcopy(self.reports[("S_temporal", 0)]["rows"])
        rows[0]["donors"][0]["donor_metrics"]["novelty"]["positive_frames"] += 1
        with self.assertRaisesRegex(RuntimeError, "different support"):
            _validate_rows(rows, self.data["scene"], "scene", donors_required=True)

    def test_missing_matrix_is_named_and_all_gates_closed(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            root = Path(temp)
            binding = {"protocol_sha256": "protocol", "data_sha256": "data",
                       "normalizers_sha256": "norms", "source_sha256": "source"}
            with patch(f"{MODULE}._context", return_value=(self.protocol, self.data, {}, binding)):
                result = aggregate(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["missing_runs"]), 25)
            self.assertIn({"arm": "D_old", "seed": 0, "reason": "evaluation_report_missing"}, result["missing_runs"])
            self.assertTrue(all(not gate["passed"] for gate in result["gates"].values()))
            self.assertEqual(read_authenticated(root / "results" / "aggregate_val.json"), result)
            self.assertFalse((root / "test_unlock.json").exists())
            first_hash = digest(root / "results" / "aggregate_val.json")
            with patch(f"{MODULE}._context", return_value=(self.protocol, self.data, {}, binding)):
                repeated = aggregate(root)
            self.assertEqual(result, repeated)
            self.assertEqual(first_hash, digest(root / "results" / "aggregate_val.json"))

    def test_started_partial_extension_is_visible_and_blocks_completeness(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            root = Path(temp)
            (root / "runs" / "S_temporal" / "seed_5").mkdir(parents=True)
            with patch(f"{MODULE}._context", return_value=(self.protocol, self.data, {}, {})):
                result = aggregate(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["expected_runs"]), 28)
            self.assertEqual(result["included_extension_seeds"], [5])
            self.assertTrue(all(5 in result["arm_seeds"][arm] for arm in SCENE_ARMS))
            self.assertNotIn(5, result["arm_seeds"]["D_old"])

    def test_test_access_is_rejected_before_input_verification_or_gpu_import(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            root = Path(temp)
            with patch(f"{MODULE}._context", return_value=(self.protocol, self.data, {}, {})), \
                    patch(f"{MODULE}._verify_inputs") as verify:
                with self.assertRaisesRegex(RuntimeError, "no authenticated"):
                    evaluate_run(root, "S_temporal", 0, split="test")
                verify.assert_not_called()

    def test_unlock_pins_validation_config_and_all_selected_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            root = Path(temp)
            binding = {"protocol_sha256": "p", "data_sha256": "d", "normalizers_sha256": "n", "source_sha256": "s"}
            input_path = root / "validation-input.json"
            authenticated_json(input_path, {"synthetic": True})
            validation = {"status": "complete", "identity": binding,
                          "eligible_families": ["motion"], "arm_seeds": self.arm_seeds,
                          "gates": {"H11_D": {"passed": True}},
                          "input_reports": {str(input_path): digest(input_path)}}
            vp = root / "results" / "aggregate_val.json"
            authenticated_json(vp, validation)
            arms = {}
            for arm in ("D_old", "D_full"):
                hashes = {}
                for seed in CORE_SEEDS:
                    path = root / "runs" / arm / f"seed_{seed}" / "best.pt"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(f"{arm}-{seed}".encode())
                    hashes[str(seed)] = digest(path)
                arms[arm] = {"config_sha256": json_digest(self.protocol["arms"][arm]),
                             "training_seeds": list(CORE_SEEDS), "checkpoints": hashes}
            unlock = {"schema_version": 1, "protocol_sha256": "p", "validation_aggregate_sha256": digest(vp),
                      "eligible_families": ["motion"], "arms": arms}
            authenticated_json(root / "test_unlock.json", unlock)
            self.assertEqual(_test_unlock(root, self.protocol, binding), unlock)
            (root / "runs" / "D_old" / "seed_0" / "best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                _test_unlock(root, self.protocol, binding)

    def test_capture_shape_hash_and_unique_matrix_are_verified(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            path = Path(temp) / "capture.npz"
            np.savez_compressed(path, truth=np.zeros((50, 6)), prediction=np.zeros((50, 6)))
            row = {"key": "capture", "n_frames": 50, "capture_path": str(path), "capture_sha256": digest(path)}
            report = {"rows": [row], "artifacts": artifact_manifest([path])}
            _verify_captures(report, [], "scene")
            with self.assertRaisesRegex(RuntimeError, "duplicated"):
                _verify_captures({**report, "rows": [row, row]}, [], "scene")
            np.savez_compressed(path, truth=np.zeros((50, 6)), prediction=np.zeros((49, 6)))
            row["capture_sha256"] = digest(path)
            report["artifacts"] = artifact_manifest([path])
            with self.assertRaisesRegex(RuntimeError, "shape"):
                _verify_captures(report, [], "scene")

    def test_training_report_requires_bound_identity_and_intact_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="stage11-evaluation-test-") as temp:
            root = Path(temp)
            directory = root / "runs" / "D_old" / "seed_0"
            directory.mkdir(parents=True)
            checkpoint = directory / "best.pt"
            checkpoint.write_bytes(b"frozen checkpoint")
            binding = {"protocol_sha256": "p", "data_sha256": "d", "normalizers_sha256": "n", "source_sha256": "s"}
            report = {"status": "completed", "steps": 12000, "nonfinite": False, "oom": False,
                      "identity": {**binding, "arm": "D_old", "seed": 0, "config": self.protocol["arms"]["D_old"]},
                      "artifacts": artifact_manifest([checkpoint])}
            authenticated_json(directory / "report.json", report)
            self.assertEqual(_training_report(root, "D_old", 0, self.protocol, binding), report)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                _training_report(root, "D_old", 0, self.protocol, {**binding, "data_sha256": "changed"})
            checkpoint.write_bytes(b"modified checkpoint")
            with self.assertRaisesRegex(RuntimeError, "Artifact drift"):
                _training_report(root, "D_old", 0, self.protocol, binding)

    def test_sensitivity_report_does_not_replace_primary(self):
        root = Path("/campaign")
        primary = _run_directory(root, "val", "S_temporal", 0)
        sensitivity = _run_directory(root, "val", "S_temporal", 0, 4)
        self.assertEqual(sensitivity, primary / "sensitivity_steps_4")
        self.assertNotEqual(primary / "report.json", sensitivity / "report.json")


if __name__ == "__main__":
    unittest.main()
