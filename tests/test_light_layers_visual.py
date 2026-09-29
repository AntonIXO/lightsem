"""CPU-only synthetic tests. No real videos, model weights, or TEST inputs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import unittest
from unittest.mock import patch
import tempfile

from lightsem.experiments.light_layers import visual as v
from lightsem.experiments.stage11_data import resize_mask, UI


class VisualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp_path = Path(self.temp.name)

    def test_import_does_not_force_torch(self):
        tmp_path = self.tmp_path
        cp = subprocess.run([sys.executable, "-c", "import sys; from lightsem.experiments.light_layers import visual; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stderr


    def test_geometry_and_original_pixel_boxes(self):
        tmp_path = self.tmp_path
        assert v.decode_geometry((1920, 1080)) == (640, 360)
        boxes = np.array([[300, 150, 600, 450]], float)
        out = v.transform_boxes(boxes, (1200, 600), (600, 300))
        np.testing.assert_allclose(out, [[150, 75, 300, 225]])
        with self.assertRaises(ValueError):
            v.transform_boxes([[np.nan, 0, 1, 1]], (100, 100), (50, 50))
        with self.assertRaises(ValueError):
            v.transform_boxes([[20, 0, 1, 1]], (100, 100), (50, 50))


    def test_ui_top_left_right_and_stage11_baseline(self):
        tmp_path = self.tmp_path
        wh = (100, 100)
        ui = v.ui_mask(wh)
        assert ui[0, 50] and ui[13, 99] and not ui[14, 50]
        assert ui[72, 70] and ui[99, 99]
        assert ui[85, 0] and ui[99, 44] and not ui[99, 45]
        assert ui[30, 0] and ui[79, 8] and not ui[80, 8]
        assert not ui[50, 50]
        boxes = np.array([[25, 30, 50, 70]], float)
        for dilation in (.15, 0.):
            ours = v.retained_mask(v.box_foreground(boxes, wh, dilation), ui)
            old = resize_mask(boxes, wh, wh, dilation=dilation) != 0
            np.testing.assert_array_equal(ours, old)


    def test_fixed_dilation_and_fallback(self):
        tmp_path = self.tmp_path
        a = np.zeros((20, 20), bool)
        a[10, 10] = True
        out = v.retained_mask(a, np.zeros_like(a), 2)
        assert (~out).sum() == 25
        boxes = np.array([[[10, 10, 60, 80]], [[20, 20, 55, 85]]], float)
        dilation, minimum = v.baseline_dilation(boxes, [0, 1], (100, 100), (100, 100))
        assert dilation == .15 and minimum >= 100
        # Padding erases the remaining strip, while unpadded boxes do not.
        boxes = np.array([[[10, 0, 90, 100]]], float)
        dilation, minimum = v.baseline_dilation(boxes, [0], (100, 100), (100, 100))
        assert dilation == 0. and minimum >= 100


    def test_timestamp_mapping_exact_grid_and_previous_pose(self):
        tmp_path = self.tmp_path
        grid = v.sample_grid(11.3, 64.)
        assert grid.dtype == np.float64 and len(grid) == 640
        np.testing.assert_allclose(grid, 11.3 + np.arange(640)/10, atol=1e-14)
        assert len(v.sample_grid(11.3, 121.)) == 1210
        source = np.array([10., 10.04, 10.09, 10.16, 10.21])
        np.testing.assert_array_equal(v.nearest_indices(source, [10., 10.1, 10.2], .1), [0, 2, 4])
        np.testing.assert_array_equal(v.previous_pose_indices(source, [10., 10.1, 10.2]), [0, 2, 3])
        with self.assertRaises(ValueError):
            v.previous_pose_indices(source, [9.99])
        with self.assertRaises(ValueError):
            v.previous_pose_indices(source, [12.])


    def test_missing_boxes_are_not_empty_background(self):
        tmp_path = self.tmp_path
        boxes = np.array([[[1, 1, 3, 4], [np.nan]*4], [[np.nan]*4, [np.nan]*4]])
        out, index, valid = v.boxes_at_time(boxes, [0., .1], .08)
        assert index == 0 and out.shape == (1, 4)
        np.testing.assert_array_equal(valid, [True, False])
        with self.assertRaisesRegex(ValueError, "all coaches"):
            v.boxes_at_time(boxes, [0., .1], .12)


    def test_atomic_bytes_dtypes_no_pickle_no_overwrite(self):
        tmp_path = self.tmp_path
        path = tmp_path/"sample.npz"
        arrays = {"rich96": np.zeros((640, 96), np.float16),
                  "source_times_s": np.arange(640, dtype=np.float64)/10,
                  "valid": np.ones(640, bool)}
        v.write_arrays(path, arrays)
        with np.load(path, allow_pickle=False) as z:
            assert z["rich96"].dtype == np.float16
            assert z["source_times_s"].dtype == np.float64
            assert z["valid"].dtype == bool
        assert path.stat().st_size < 20000
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            v.write_arrays(path, arrays)
        assert path.read_bytes() == before


    def test_fixed_audit_frames_and_rgb(self):
        tmp_path = self.tmp_path
        from PIL import Image
        assert v.audit_indices(640) == [0, 320, 639]
        image = np.zeros((4, 6, 3), np.uint8)
        image[..., 0] = 255
        keep = np.ones((4, 6), bool)
        v.save_audit(tmp_path/"audit.png", image, keep, keep)
        result = np.asarray(Image.open(tmp_path/"audit.png"))
        np.testing.assert_array_equal(result[:, :6], image)
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])


    def test_manifest_accepts_leading_hyphen_recording_id(self):
        source, pose, checkpoint = (self.tmp_path/name for name in ("source", "pose", "sam2.1_s.pt"))
        for path in (source, pose, checkpoint):
            path.write_bytes(b"synthetic")
        row = {"recording_id": "--74o20SmOs", "canonical_id": "synthetic", "partition": "train",
               "source_path": str(source), "pose_path": str(pose), "source_sha256": v.sha256(source),
               "pose_sha256": v.sha256(pose), "start_s": 0., "duration_s": 64.,
               "original_width": 100, "original_height": 100}
        manifest = {"inputs": [row], "checkpoint_sha256": v.sha256(checkpoint)}
        assert v.validate_manifest(manifest, checkpoint) == [row]
        row["recording_id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            v.validate_manifest(manifest, checkpoint)

    def _development_manifest(self):
        source, pose, checkpoint = (self.tmp_path/name for name in ("source", "pose", "sam2.1_s.pt"))
        for path in (source, pose, checkpoint):
            path.write_bytes(b"synthetic development guard fixture")
        row = {"recording_id": "historical-val", "canonical_id": "synthetic", "partition": "val",
               "source_path": str(source), "pose_path": str(pose), "source_sha256": v.sha256(source),
               "pose_sha256": v.sha256(pose), "start_s": 0., "duration_s": 64.,
               "original_width": 100, "original_height": 100}
        return {"inputs": [row], "checkpoint_sha256": v.sha256(checkpoint)}, checkpoint

    def test_development_requires_both_explicit_flag_and_exposure(self):
        manifest, checkpoint = self._development_manifest()
        with patch.object(v, "sha256") as read_hash:
            with self.assertRaises(ValueError):
                v.validate_manifest(manifest, checkpoint)
            with self.assertRaisesRegex(ValueError, "exposure"):
                v.validate_manifest(manifest, checkpoint, allow_development=True)
            manifest["exposure"] = "historical_development"
            with self.assertRaises(ValueError):
                v.validate_manifest(manifest, checkpoint)
            read_hash.assert_not_called()
        assert v.validate_manifest(manifest, checkpoint, allow_development=True) == manifest["inputs"]
        # TRAIN plus historical VAL is explicitly permitted under the same gate.
        train = dict(manifest["inputs"][0], recording_id="train-record", partition="train")
        manifest["inputs"].insert(0, train)
        assert len(v.validate_manifest(manifest, checkpoint, allow_development=True)) == 2

    def test_test_partition_never_opened_even_beyond_prefix_limit(self):
        manifest, checkpoint = self._development_manifest()
        manifest["exposure"] = "historical_development"
        manifest["inputs"].insert(0, dict(manifest["inputs"][0], recording_id="train", partition="train"))
        manifest["inputs"][1]["partition"] = "test"
        with patch.object(v, "sha256") as read_hash:
            with self.assertRaisesRegex(ValueError, "TEST"):
                v.validate_manifest(manifest, checkpoint, limit_records=1, allow_development=True)
            read_hash.assert_not_called()
        manifest["inputs"][1]["partition"] = "unknown"
        with patch.object(v, "sha256") as read_hash:
            with self.assertRaises(ValueError):
                v.validate_manifest(manifest, checkpoint, allow_development=True)
            read_hash.assert_not_called()

    def test_development_flag_is_boolean_and_default_is_train_only(self):
        assert not v.VisualConfig().allow_development
        assert v.VisualConfig(allow_development=True).allow_development
        with self.assertRaises(ValueError):
            v.VisualConfig(allow_development=1)
        with self.assertRaises(ValueError):
            v.validate_manifest({}, "unused", allow_development="yes")

    def test_development_cli_passes_explicit_flag(self):
        argv = ["--manifest", "synthetic.json", "--checkpoint", "synthetic.pt", "--output", "new"]
        with patch.object(v, "run", return_value={"status": "complete"}) as run:
            assert v.main(argv) == 0
            assert not run.call_args.args[3].allow_development
            assert v.main(argv + ["--allow-development"]) == 0
            assert run.call_args.args[3].allow_development

    def test_missing_frame_gate_needs_manifest_policy_and_registered_indices(self):
        manifest, checkpoint = self._development_manifest()
        manifest["inputs"][0]["partition"] = "train"
        with patch.object(v, "sha256") as read_hash:
            with self.assertRaisesRegex(ValueError, "gap_policy"):
                v.validate_manifest(manifest, checkpoint, invalidate_missing_frames=True)
            manifest["gap_policy"] = "invalidate_frame_no_imputation"
            with self.assertRaisesRegex(ValueError, "registered"):
                v.validate_manifest(manifest, checkpoint, invalidate_missing_frames=True)
            read_hash.assert_not_called()
        manifest["inputs"][0]["coaches"] = [{"coach_index": 0}]
        assert v.validate_manifest(manifest, checkpoint, invalidate_missing_frames=True) == manifest["inputs"]
        for bad in ([], [{"coach_index": 0}, {"coach_index": 0}], [{"coach_index": -1}], [{"coach_index": True}]):
            manifest["inputs"][0]["coaches"] = bad
            with self.assertRaisesRegex(ValueError, "registered"):
                v.validate_manifest(manifest, checkpoint, invalidate_missing_frames=True)
        with self.assertRaises(ValueError):
            v.VisualConfig(invalidate_missing_frames=1)
        argv = ["--manifest", "synthetic.json", "--checkpoint", "synthetic.pt", "--output", "new"]
        with patch.object(v, "run", return_value={"status": "complete"}) as run:
            v.main(argv + ["--invalidate-missing-frames"])
            assert run.call_args.args[3].invalidate_missing_frames
            v.main(argv)
            assert not run.call_args.args[3].invalidate_missing_frames

    def test_pose_support_all_registered_not_all_cache_slots(self):
        boxes = np.array([[[1, 1, 3, 4], [np.nan]*4], [[1, 1, 3, 4], [3, 3, 2, 8]]])
        _, _, age, valid = v.pose_frame_support(boxes, [1., 1.1], [.9, 1., 1.4], registered_indices=[0])
        np.testing.assert_array_equal(age, [False, True, False])
        np.testing.assert_array_equal(valid, [False, True, False])
        _, coach, _, valid = v.pose_frame_support(boxes, [1., 1.1], [1., 1.1], registered_indices=[0, 1])
        assert coach[:, 0].all() and not coach[:, 1].any() and not valid.any()
        for bad in ([0, 0], [2], []):
            with self.assertRaises(ValueError):
                v.pose_frame_support(boxes, [1., 1.1], [1.], registered_indices=bad)

    def _extract_synthetic_gap(self, *, enabled=True, all_missing=False):
        times = np.arange(20, dtype=np.float64)/10
        boxes = np.broadcast_to(np.array([[12., 12., 18., 20.], [20., 12., 24., 20.]]), (20, 2, 4)).copy()
        boxes[10] = np.nan if all_missing else boxes[10]
        boxes[10, 1] = np.nan
        rgb = np.zeros((24, 32, 3), np.uint8)
        rgb[..., 0] = 255
        row = {"recording_id": "synthetic", "canonical_id": "synthetic", "partition": "train",
               "start_s": 0., "duration_s": 2., "original_width": 32, "original_height": 24,
               "source_path": "NOT_OPENED", "coaches": [{"coach_index": 0}, {"coach_index": 1}]}
        mapping = {"source_times_s": times, "requested_times_s": times.copy(), "source_pts": np.arange(20),
                   "frame_indices": np.arange(20), "time_base": "1/10", "stream_start_time_s": 0.}
        class StubSegmenter:
            calls = 0
            def reset_peak(self): pass
            def peak_bytes(self): return 0
            def __call__(self, image, prompts):
                self.calls += 1
                return v.box_foreground(prompts, (32, 24)), np.ones(len(prompts))
        segmenter = StubSegmenter()
        with patch.object(v, "probe_frame_mapping", return_value=mapping), \
             patch.object(v, "load_pose", return_value=(boxes, times, {})), \
             patch.object(v, "iter_rgb_frames", return_value=(rgb for _ in times)), \
             patch.object(v, "require_disk_floor", return_value=2**40):
            receipt = v.extract_record(row, self.tmp_path, segmenter,
                                       v.VisualConfig(invalidate_missing_frames=enabled, audit=True))
        return receipt, segmenter

    def test_gap_has_no_ghost_background_or_supervised_labels(self):
        receipt, segmenter = self._extract_synthetic_gap()
        assert segmenter.calls == 19  # No SAM inference at missing coach frame.
        assert receipt["frames"] == 20 and receipt["invalid_frames"] == 1
        assert receipt["frame_valid_fraction"] == .95
        assert receipt["audit"]["invalid_frame_indices"] == [10]
        assert receipt["pose"]["registered_coach_indices"] == [0, 1]
        for mode in v.MODES:
            with np.load(self.tmp_path/f"{mode}.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["source_times_s"], np.arange(20)/10)
                assert not z["frame_valid"][10] and z["frame_valid"].sum() == 19
                assert not z["sam_prediction_valid"][10]
                for key in ("rich96", "hs_histogram", "value_summary", "retained_pixels", "valid_pixels",
                            "retained_fraction", "chromatic_fraction", "mask_activity"):
                    assert np.all(z[key][10] == 0), key
                for key in ("color_valid", "quadrant_color_valid", "value_valid"):
                    assert not np.any(z[key][10]), key
                # A missing label has no valid loss weight, even when zero is a
                # plausible real observation. Consumers must use these flags.
                label_weight = z["frame_valid"] & z["color_valid"] & z["value_valid"]
                assert not label_weight[10] and label_weight.sum() == 19
                assert not z["novelty_eligible"][8:14].any()
                assert z["novelty_eligible"][3:8].all()
                assert not z["mask_activity_known"][[0, 10, 11]].any()
                assert z["mask_activity_known"][12]
                assert not z["novelty_window_mask_activity_known"][8:14].any()
        from PIL import Image
        audit = np.asarray(Image.open(self.tmp_path/"audit_0010.png"))
        assert audit.shape == (50, 32, 3)  # RGB + header, no invented mask panels.
        np.testing.assert_array_equal(audit[26:, :, 0], 255)

    def test_strict_policy_still_rejects_missing_all_coaches(self):
        with self.assertRaisesRegex(ValueError, "all coaches"):
            self._extract_synthetic_gap(enabled=False, all_missing=True)
        assert not (self.tmp_path/"complete.json").exists()

    def test_manifest_rejects_nontrain_before_any_file_open(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            v.validate_manifest({"inputs": [{"partition": "test"}]}, tmp_path/"missing")


    def test_balanced_select_expression_validation(self):
        expression = v.balanced_select_expression(np.arange(640))
        assert expression.count("eq(n") == 640
        assert "eq(n\\,0)" in expression and "eq(n\\,639)" in expression
        for bad in ([], [1, 1], [2, 1], [-1, 1], [0., 1.]):
            with self.assertRaises(ValueError):
                v.balanced_select_expression(bad)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_cpu_640_term_select_regression(self):
        # Flat sum fails in libavutil at this size even for a tiny video.
        path = self.tmp_path/"synthetic640.mkv"
        cp = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                             "color=c=red:s=32x24:r=10:d=64", "-c:v", "ffv1", str(path)],
                            capture_output=True)
        assert cp.returncode == 0, cp.stderr
        mapping = {"frame_indices": np.arange(640, dtype=np.int64)}
        count = 0
        for frame in v.iter_rgb_frames(path, mapping, (16, 12)):
            assert frame.shape == (12, 16, 3)
            assert frame[0, 0, 0] > 240 and frame[0, 0, 2] < 10
            count += 1
        assert count == 640

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg unavailable")
    def test_cpu_synthetic_video_pts_and_rgb(self):
        tmp_path = self.tmp_path
        path = tmp_path/"synthetic.mkv"
        cp = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                             "color=c=red:s=32x24:r=25:d=1", "-c:v", "ffv1", str(path)], capture_output=True)
        assert cp.returncode == 0, cp.stderr
        grid = np.arange(8, dtype=np.float64)/10 + .1
        mapping = v.probe_frame_mapping(path, grid, (32, 24), .04)
        np.testing.assert_allclose(mapping["source_times_s"], [.08, .2, .28, .4, .48, .6, .68, .8], atol=.001)
        assert len(mapping["source_pts"]) == len(grid)
        count = 0
        for frame in v.iter_rgb_frames(path, mapping, (16, 12)):
            assert frame.shape == (12, 16, 3)
            assert frame[0, 0, 0] > 240 and frame[0, 0, 2] < 10
            count += 1
        assert count == len(grid)

if __name__ == "__main__":
    unittest.main()
