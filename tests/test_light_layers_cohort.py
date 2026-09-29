"""Synthetic CPU tests. No real cohort, model, or TEST raw inputs are opened."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import scripts.light_layers.run_light_layers_cohort as c


def metadata(rid, cid, partition="train", eligible=True):
    return {"recording_id": rid, "canonical_id": cid, "partition": partition,
            "eligible_core": eligible, "coaches": []}


def split_for(rows):
    return {"recordings": {row["recording_id"]: {
        "canonical_id": row["canonical_id"], "partition": row["partition"]} for row in rows}}


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source_row(self, rid="train1", cid="family1", partition="train"):
        row = metadata(rid, cid, partition)
        directory = self.root / rid
        directory.mkdir()
        times = np.arange(0, 80.01, .1)
        boxes = np.tile(np.array([10., 20., 30., 50.]), (len(times), 2, 1))
        pose = directory / "pose.npz"
        np.savez_compressed(pose, t=times, box=boxes, meta=json.dumps({"width": 100, "height": 100}),
                            confidence=np.array([object()], dtype=object))
        row.update(active_start_s=.03, active_end_s=80.03, duration_s=80.1,
                   source_path=str(directory / "video.webm"), pose_path=str(pose),
                   wav_path=str(directory / "audio.wav"), audio_path=str(directory / "bus.npz"),
                   visual_path=str(directory / "must_not_read_visual.npz"),
                   pose_provenance={"width": 100, "height": 100},
                   coaches=[{"coach_index": 0, "partition": partition},
                            {"coach_index": 1, "partition": partition}])
        for key in ("source_path", "wav_path", "audio_path"):
            Path(row[key]).write_bytes((rid + key).encode())
        row["pose_sha256"] = c.sha256_file(row["pose_path"])
        row["source_sha256"] = c.sha256_file(row["source_path"])
        return row

    def test_stable_hash_is_exact_concatenation(self):
        expected = hashlib.sha256(b"campaigncohort-v1canonical").hexdigest()
        self.assertEqual(c.canonical_order_key("campaign", "canonical"), expected)
        self.assertNotEqual(c.canonical_order_key("campaign2", "canonical"), expected)

    def test_group_selection_stable_lexicographic_and_split_preserved(self):
        families = sorted(("g1", "g2", "g3"), key=lambda cid: c.canonical_order_key("campaign", cid))
        rows = [metadata("b", families[0]), metadata("a", families[0]),
                metadata("c", families[1]), metadata("d", families[2]),
                metadata("v1", "valfamily", "val"), metadata("v2", "valfamily", "val"),
                metadata("TEST_MUST_NOT_READ", "testfamily", "test"),
                metadata("INELIGIBLE_MUST_NOT_READ", "excludedfamily", eligible=False)]
        calls = []

        def check(row):
            calls.append(row["recording_id"])
            self.assertNotEqual(row["partition"], "test")
            return {"qualified": row["recording_id"] != "a", "reasons": ["source_missing"] if row["recording_id"] == "a" else []}

        one, decisions = c.choose_groups(rows, split_for(rows), "campaign", check, {"train": 2, "val": 1})
        two, _ = c.choose_groups(list(reversed(rows)), split_for(rows), "campaign", check, {"train": 2, "val": 1})
        self.assertEqual([r["recording_id"] for r, _ in one], [r["recording_id"] for r, _ in two])
        self.assertEqual({r["recording_id"] for r, _ in one}, {"b", "c", "v1"})
        self.assertEqual(len({r["canonical_id"] for r, _ in one}), 3)
        self.assertNotIn("TEST_MUST_NOT_READ", calls)
        self.assertNotIn("INELIGIBLE_MUST_NOT_READ", calls)
        self.assertIn("d", calls)  # All source candidates screened even after quota.
        statuses = {d["recording_id"]: d["status"] for d in decisions}
        self.assertEqual(statuses["a"], "source_excluded")
        self.assertEqual(statuses["v2"], "qualified_not_selected_same_canonical")
        self.assertEqual(statuses["d"], "qualified_not_selected_quota")
        self.assertEqual(len(decisions), len(rows))

    def test_window_is_global_grid_not_active_offset_grid(self):
        start, duration = c.centered_window(.037, 100.037, 101.)
        self.assertEqual((start, duration), (18., 64.))
        self.assertAlmostEqual(start * 10, round(start * 10))
        self.assertNotAlmostEqual((start - .037) * 10, round((start - .037) * 10))
        self.assertEqual(c.centered_window(.05, 100.05, 101.)[0], 18.1)
        for values in ((0, 63, 63), (20, 83, 100), (0, float("nan"), 100), (-1, 100, 100)):
            with self.assertRaises(ValueError):
                c.centered_window(*values)

    def arrays(self):
        times = np.arange(0., 80.01, .1)
        boxes = np.tile([1., 2., 3., 4.], (len(times), 3, 1))
        return times, boxes

    def test_padded_times_cover_window_and_maxgap(self):
        times, boxes = self.arrays()
        reasons, detail = c.qualify_pose_arrays(times, boxes, [0, 2], 8.)
        self.assertEqual(reasons, [])
        self.assertEqual(detail["checked_interval_s"], [7.8, 72.2])
        self.assertLessEqual(detail["max_gap_s"], .2)
        chosen = (times >= 8.) & (times <= 72.)
        reasons, _ = c.qualify_pose_arrays(times[chosen], boxes[chosen], [0, 2], 8.)
        self.assertIn("pose_times_do_not_cover_padded_window", reasons)
        chosen = (times < 30.) | (times > 30.2)
        reasons, _ = c.qualify_pose_arrays(times[chosen], boxes[chosen], [0, 2], 8.)
        self.assertIn("pose_gap_above_0.2s", reasons)
        times[20] = times[19]
        self.assertIn("pose_times_not_finite_strictly_increasing", c.qualify_pose_arrays(times, boxes, [0], 8.)[0])

    def test_every_registered_coach_and_every_frame_including_padding(self):
        times, boxes = self.arrays()
        for index, value in ((79, [np.nan]*4), (800, [1., 2., 1., 4.])):
            boxes[index, 2] = value
        reasons, detail = c.qualify_pose_arrays(times, boxes, [0, 2], 8.)
        self.assertIn("registered_coach_box_missing_nonfinite_or_nonpositive", reasons)
        self.assertEqual(detail["invalid_box_frames_per_registered_coach"]["2"], 1)
        # An unregistered slot does not redefine the registered coach list.
        self.assertEqual(c.qualify_pose_arrays(times, boxes, [0, 1], 8.)[0], [])
        boxes[79, 2] = [1., 2., 1., 4.]
        self.assertIn("registered_coach_box_missing_nonfinite_or_nonpositive", c.qualify_pose_arrays(times, boxes, [2], 8.)[0])
        self.assertEqual(c.qualify_pose_arrays(times, boxes[:, 0], [0], 8.)[0], [])
        self.assertIn("registered_coach_indices_invalid", c.qualify_pose_arrays(times, boxes, [3], 8.)[0])
        self.assertIn("registered_coach_indices_invalid", c.qualify_pose_arrays(times, boxes, [], 8.)[0])

    def test_gap_and_boxes_include_bracketing_frames(self):
        times, boxes = self.arrays()
        times += .03
        boxes[77, 1] = np.nan  # 7.73 is just before 7.8, needed for edge coverage.
        reasons, detail = c.qualify_pose_arrays(times, boxes, [1], 8.)
        self.assertIn("registered_coach_box_missing_nonfinite_or_nonpositive", reasons)
        self.assertEqual(detail["first_checked_pose_index"], 77)

    def test_TEST_open_is_forbidden_even_with_forged_partition_or_alias(self):
        train = self.source_row()
        test = metadata("test", "testfamily", "test")
        test.update({key: str(self.root / ("TEST_NEVER_OPEN_" + key)) for key in c.RAW_KEYS})
        access = c.SourceAccess([train, test], split_for([train, test]))
        forged = dict(test, partition="train", eligible_core=True)
        with patch.object(Path, "open", side_effect=AssertionError("raw read attempted")):
            for row in (test, forged):
                with self.assertRaises(PermissionError):
                    access.open(row, "pose_path")
                with self.assertRaises(PermissionError):
                    c.qualify_record(row, access)
        alias = dict(train, pose_path=test["pose_path"])
        alias_access = c.SourceAccess([alias, test], split_for([alias, test]))
        with patch.object(Path, "open", side_effect=AssertionError("TEST alias read attempted")):
            with self.assertRaises(PermissionError):
                alias_access.open(alias, "pose_path")
        self.assertEqual(dict(access.access_counts), {})

    def test_split_disagreement_fails_before_source_open(self):
        row = self.source_row()
        split = split_for([row])
        split["recordings"][row["recording_id"]]["partition"] = "test"
        with patch.object(Path, "open", side_effect=AssertionError("raw read attempted")):
            with self.assertRaisesRegex(ValueError, "historical split disagreement"):
                c.SourceAccess([row], split)
        crossed = dict(row, recording_id="another", partition="val", coaches=[])
        with self.assertRaisesRegex(ValueError, "crosses historical partitions"):
            c.SourceAccess([row, crossed], split_for([row, crossed]))

    def test_source_only_check_does_not_load_confidence_or_legacy_targets(self):
        row = self.source_row()
        access = c.SourceAccess([row], split_for([row]))
        result = c.qualify_record(row, access)
        self.assertTrue(result["qualified"], result)
        self.assertFalse(Path(row["visual_path"]).exists())
        Path(row["wav_path"]).unlink()
        result = c.qualify_record(row, access)
        self.assertFalse(result["qualified"])
        self.assertIn("unreadable_wav_path", result["reasons"])

    def test_selected_pose_hash_verified_no_silent_replacement(self):
        row = self.source_row()
        access = c.SourceAccess([row], split_for([row]))
        result = c.qualify_record(row, access)
        altered = dict(row, pose_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "stop without replacement"):
            c.hash_record(altered, result, access, {}, self.root)

    def test_coverage90_is_separate_requires_both_intervals_and_joint_presence(self):
        times, boxes = self.arrays()
        # Half the center8s is missing, although full64s coverage exceeds 90%.
        boxes[(times >= 36.) & (times < 40.), 0] = np.nan
        reasons, pose_detail = c.qualify_pose_arrays(times, boxes, [0, 1], 8.)
        strict = {"qualified": False, "reasons": reasons, "detail": {"pose": pose_detail}}
        supplemental = c.coverage90_qualification(strict)
        self.assertFalse(supplemental["qualified"])
        self.assertIn("center8s_registered_coach_presence_below_0.90", supplemental["reasons"])
        self.assertNotIn("window64s_registered_coach_presence_below_0.90", supplemental["reasons"])
        self.assertEqual(strict["reasons"], ["registered_coach_box_missing_nonfinite_or_nonpositive"])
        times, boxes = self.arrays()
        # Disjoint coach failures count as jointly invalid frames, not average coverage.
        boxes[(times >= 12.) & (times < 16.), 0] = np.nan
        boxes[(times >= 20.) & (times < 24.), 1] = np.nan
        reasons, pose_detail = c.qualify_pose_arrays(times, boxes, [0, 1], 8.)
        supplemental = c.coverage90_qualification({"qualified": False, "reasons": reasons, "detail": {"pose": pose_detail}})
        self.assertIn("window64s_registered_coach_presence_below_0.90", supplemental["reasons"])
        self.assertEqual(pose_detail["unregistered_pose_slot_indices"], [2])
        times, boxes = self.arrays()
        boxes[100, 0] = np.nan
        reasons, pose_detail = c.qualify_pose_arrays(times, boxes, [0, 1], 8.)
        strict = {"qualified": False, "reasons": reasons, "detail": {"pose": pose_detail}}
        self.assertTrue(c.coverage90_qualification(strict)["qualified"])
        strict["reasons"].append("pose_gap_above_0.2s")
        self.assertFalse(c.coverage90_qualification(strict)["qualified"])

    def test_end_to_end_candidate_report_insufficiency_and_no_clobber(self):
        rows = [self.source_row(), self.source_row("val1", "valfamily", "val")]
        test = metadata("test", "testfamily", "test")
        test.update({key: str(self.root / ("TEST_FORBIDDEN_" + key)) for key in c.RAW_KEYS})
        rows.append(test)
        hashes = {row[key]: c.sha256_file(row[key]) for row in rows[:2] for key in ("wav_path", "audio_path")}
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"recordings": rows, "artifact_hashes": hashes}))
        split = self.root / "split_manifest.json"
        split.write_text(json.dumps(split_for(rows)))
        result = c.build_cohort(manifest, split, self.root / "campaign")
        self.assertEqual(result["summary"]["train"]["shortfall"], 99)
        self.assertEqual(result["summary"]["val"]["shortfall"], 39)
        cohort = json.loads(Path(result["files"]["cohort"]).read_text())
        report = json.loads(Path(result["files"]["report"]).read_text())
        self.assertFalse(cohort["frozen_for_extraction"])
        self.assertEqual(report["TEST_raw_open_count"], 0)
        self.assertEqual(len(report["decisions"]), 3)
        self.assertEqual(report["cohort_sha256"], c.sha256_file(result["files"]["cohort"]))
        val = next(row for row in cohort["inputs"] if row["partition"] == "val")
        self.assertEqual(val["partition_role"], "historical_VAL_exposed_development")
        self.assertEqual(val["rhythm_path"], str(self.root / "campaign" / "rhythm_dev_features" / "val1.npz"))
        for row in cohort["inputs"]:
            expected = row["input_metadata_sha256"]
            payload = dict(row)
            del payload["input_metadata_sha256"]
            self.assertEqual(expected, c.object_sha256(payload))
        with self.assertRaises(FileExistsError):
            c.build_cohort(manifest, split, self.root / "campaign")


if __name__ == "__main__":
    unittest.main()
