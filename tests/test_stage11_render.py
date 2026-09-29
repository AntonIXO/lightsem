"""CPU-only rendering-contract checks; no EGL context is created."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from lightsem.experiments.stage11 import stage11_render as render


@dataclass
class FakeConfig:
    width: int = 854
    height: int = 480
    fps: float = 30.0
    haze: float = 0.115
    cam_pos: tuple = (0.0, 2.6, 11.0)
    cam_target: tuple = (0.0, 3.2, -3.0)
    cam_orbit: float = 0.0
    intensity_gain: float = 2.4


class FakeRenderer:
    def __init__(self):
        self.cfg = FakeConfig()
        self.images = {}
        self.streams = {}

    def render_frame(self, universe, t):
        # Nonzero background and nonlinear brightness ensure matching cannot
        # assume a linear relation between DMX intensity and screen luma.
        light = 5 + round(200 * (float(universe[4]) / 255) ** 0.8)
        return np.full((36, 64, 3), light, dtype=np.uint8)

    def render_video(self, frames, stream_fps, out_path, **kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-video:" + frames.tobytes())
        self.streams[str(path)] = frames.copy()
        duration = kwargs["duration_s"]
        self.images[str(path)] = [self.render_frame(frames[min(round(t * stream_fps), len(frames) - 1)], t)
                                  for t in np.arange(round(duration * self.cfg.fps)) / self.cfg.fps]
        return path


class EncodedMismatchRenderer(FakeRenderer):
    """Raw probes rank the opposite arm as brighter than encoded observation."""

    def render_frame(self, universe, t):
        coefficient = 60 if universe[0] == 17 else 170
        light = 4 + round(coefficient * (float(universe[4]) / 255) ** 0.8)
        return np.full((36, 64, 3), light, dtype=np.uint8)

    def render_video(self, frames, stream_fps, out_path, **kwargs):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-encoded-video:" + frames.tobytes())
        self.streams[str(path)] = frames.copy()
        images = []
        for t in np.arange(round(kwargs["duration_s"] * self.cfg.fps)) / self.cfg.fps:
            frame = frames[min(round(t * stream_fps), len(frames) - 1)]
            coefficient = 180 if frame[0] == 17 else 90
            light = 8 + round(coefficient * (float(frame[4]) / 255) ** 0.55)
            images.append(np.full((36, 64, 3), light, dtype=np.uint8))
        self.images[str(path)] = images
        return path


def dmx(level, n=40):
    out = np.zeros((n, 512), dtype=np.uint8)
    out[:, 4] = level
    return out


class Stage11RenderTests(unittest.TestCase):
    def test_blinder_is_linear_in_v2_and_historical_metric_is_unchanged(self):
        rig = canonical_rig()
        frames = np.zeros((3, 512), dtype=np.uint8)
        for fixture in rig.fixtures:
            if fixture.kind == "blinder":
                frames[:, fixture.address - 1] = [0, 64, 128]
        legacy = decode_stream(rig, frames)
        updated = render.decode_metric_v2(rig, frames)
        index = np.flatnonzero(legacy["kind"] == "blinder")[0]
        dim = frames[:, rig.fixtures[index].address - 1] / 255
        np.testing.assert_allclose(emitted_light(legacy)[:, index], dim ** 2)
        np.testing.assert_allclose(render.emitted_v2(updated)[:, index], dim)
        np.testing.assert_allclose(render.emitted_v2(legacy)[:, index], dim)
        self.assertAlmostEqual(updated["sat"][1, index], 0.15)
        np.testing.assert_allclose(legacy["val"][:, index], dim)

    def test_v2_matches_fixed_white_strobe_renderer_and_true_zero_rgb(self):
        rig = canonical_rig()
        frames = np.zeros((1, 512), dtype=np.uint8)
        for fixture in rig.fixtures:
            if fixture.kind == "strobe":
                frames[0, fixture.address - 1] = 128
                frames[0, fixture.address + 1] = 0
            elif fixture.kind == "wash":
                frames[0, fixture.address - 1] = 128
        decoded = render.decode_metric_v2(rig, frames)
        intensity = render.emitted_v2(decoded)
        np.testing.assert_allclose(intensity[:, decoded["kind"] == "strobe"], 128 / 255)
        self.assertEqual(float(intensity[:, decoded["kind"] == "wash"].sum()), 0)

    def test_dimmer_gain_preserves_every_other_byte(self):
        rig = canonical_rig()
        frames = np.random.default_rng(3).integers(0, 256, (5, 512), dtype=np.uint8)
        original = frames.copy()
        scaled = render.scale_dimmer(frames, 0.5, rig=rig)
        dimmer_columns = [f.address - 1 + (4 if f.kind == "moving_head" else 0) for f in rig.fixtures]
        other = np.setdiff1d(np.arange(512), dimmer_columns)
        np.testing.assert_array_equal(scaled[:, other], frames[:, other])
        np.testing.assert_array_equal(scaled[:, dimmer_columns], np.floor(frames[:, dimmer_columns] / 2 + 0.5))
        np.testing.assert_array_equal(original, frames)
        with self.assertRaises(ValueError):
            render.scale_dimmer(frames, 1.1)

    def test_screen_luma_uses_fixed_weights_and_records_clipping(self):
        red = np.zeros((36, 64, 3), dtype=np.uint8)
        red[..., 0] = 255
        stats = render.image_luma_stats([red])
        self.assertAlmostEqual(stats["mean"], 0.2126)
        self.assertAlmostEqual(stats["rgb_clip_fraction"], 1 / 3)
        self.assertEqual(stats["luma_size"], [64, 36])
        self.assertEqual(stats["roi"], "whole frame")
        self.assertEqual(render.symmetric_luma_gap(0, 0), 0)
        self.assertAlmostEqual(render.symmetric_luma_gap(1, 2), 2 / 3)

    def test_screen_calibration_handles_nonlinearity_and_only_dims_brighter(self):
        renderer = FakeRenderer()
        original_a, original_b = dmx(60), dmx(190)
        a, b, stats = render.match_luma(original_a, original_b, renderer, 1.0)
        self.assertTrue(stats["probe_pass"])
        self.assertLessEqual(stats["symmetric_gap"], 0.03)
        self.assertEqual(stats["gains"][0], 1.0)
        self.assertLess(stats["gains"][1], 1.0)
        np.testing.assert_array_equal(a, original_a)
        self.assertLess(float(b.mean()), float(original_b.mean()))
        self.assertFalse(stats["physical_brightness_matched"])
        self.assertLessEqual(len(stats["probe_history"]), 12)

    def test_probe_timeline_matches_video_lookup(self):
        indices, times = render._probe_indices(1200, 30.0, 24)
        self.assertEqual(len(indices), 24)
        np.testing.assert_array_equal(indices, np.rint(times * 40).astype(int))
        self.assertLess(times[-1], 30)

    def test_scene_mapping_is_deterministic_and_rejects_misaligned_bar_phase(self):
        scene = np.tile([1.0, 0.0, 0.8, 0.0, 0.3, 0.9], (20, 1))
        phase = (np.arange(80) / 80) % 1
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        a = render.frames_scene(scene, beats, phase, 2.0)
        b = render.frames_scene(scene, beats, phase, 2.0)
        self.assertEqual(a.frames.shape, (80, 512))
        np.testing.assert_array_equal(a.frames, b.frames)
        self.assertEqual(a.cues[0]["tau_level"], 0.3)
        self.assertEqual(a.cues[0]["wave_amp"], 0.6)
        with self.assertRaisesRegex(ValueError, "exactly"):
            render.frames_scene(scene, beats, phase[:-1], 2.0)

    def test_excerpt_selection_is_hash_based_and_grid_aligned(self):
        first = render.select_excerpt_start("song-a", 200, salt="frozen")
        self.assertEqual(first, render.select_excerpt_start("song-a", 200, salt="frozen"))
        self.assertNotEqual(first, render.select_excerpt_start("song-b", 200, salt="frozen"))
        self.assertGreaterEqual(first, 5)
        self.assertLessEqual(first, 165)
        self.assertAlmostEqual(first * 40, round(first * 40))
        with self.assertRaises(ValueError):
            render.select_excerpt_start("song-a", 39, salt="frozen")

    def test_full_matched_render_checks_encoded_video_without_overwriting(self):
        renderer = FakeRenderer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(render, "measure_video_luma", side_effect=lambda p: render.image_luma_stats(renderer.images[str(p)])):
                result = render.render_matched_pair(dmx(55), dmx(155), None, 1.0, root / "pair", renderer=renderer)
            self.assertTrue(result["passed"])
            self.assertLessEqual(result["symmetric_gap"], 0.03)
            self.assertTrue((root / "pair" / "match.json").is_file())
            with self.assertRaises(FileExistsError):
                render.render_matched_pair(dmx(55), dmx(155), None, 1.0, root / "pair", renderer=renderer)

    def test_fixed_camera_is_enforced(self):
        renderer = FakeRenderer()
        renderer.cfg.cam_orbit = 0.07
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cam_orbit"):
                render.render_frames(dmx(55), Path(tmp) / "out.mp4", renderer=renderer)

    def test_encoded_feedback_ignores_disagreeing_raw_probes_and_dims_actual_brighter(self):
        renderer = EncodedMismatchRenderer()
        left, right = dmx(180), dmx(180)
        left[:, 0], right[:, 0] = 17, 29  # Non-dimmer identities/spatial appearance.
        originals = [left.copy(), right.copy()]
        raw = [render.image_luma_stats([renderer.render_frame(frames[0], 0)])["mean"]
               for frames in originals]
        self.assertGreater(raw[1], raw[0])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "pair"
            with (patch.object(render, "probe_luma", side_effect=AssertionError("raw probes must not calibrate encoding")),
                  patch.object(render, "measure_video_luma", side_effect=lambda p: render.image_luma_stats(renderer.images[str(p)]))):
                result = render.render_matched_pair(left, right, None, 1.0, directory, renderer=renderer)
            self.assertTrue(result["passed"])
            self.assertLessEqual(result["symmetric_gap"], .03)
            calibration = result["calibration"]
            self.assertGreater(calibration["before"][0]["mean"], calibration["before"][1]["mean"])
            self.assertLess(calibration["gains"][0], 1.)
            self.assertEqual(calibration["gains"][1], 1.)
            self.assertEqual(calibration["target_dark_side"], 1)
            self.assertFalse(calibration["pre_encoding_probe_used_for_acceptance"])
            self.assertEqual(calibration["maximum_corrections"], 12)
            history = calibration["encoded_history"]
            self.assertEqual(history[0]["gain"], 0.)
            self.assertTrue(all(row["side"] == 0 and 0 <= row["gain"] <= 1 for row in history))
            self.assertLessEqual(len(history), 12)
            self.assertEqual(len(list(directory.glob("*.mp4"))), 2 + len(history))
            for row in history:
                self.assertEqual(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest(), row["video_sha256"])
            np.testing.assert_array_equal(left, originals[0])
            np.testing.assert_array_equal(right, originals[1])
            np.testing.assert_array_equal(renderer.streams[result["paths"][1]], right)
            self.assertEqual(result["frames_sha256"][0],
                             render._hash_frames(renderer.streams[result["paths"][0]]))

    def test_matching_encoded_originals_need_no_correction(self):
        renderer = FakeRenderer()
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(render, "probe_luma", side_effect=AssertionError("raw probes must not guide acceptance")),
                  patch.object(render, "measure_video_luma", side_effect=lambda p: render.image_luma_stats(renderer.images[str(p)]))):
                result = render.render_matched_pair(dmx(100), dmx(100), None, 1., Path(tmp) / "pair", renderer=renderer)
            self.assertTrue(result["passed"])
            self.assertEqual(result["calibration"]["gains"], [1., 1.])
            self.assertEqual(result["calibration"]["encoded_history"], [])
            self.assertEqual(len(list((Path(tmp) / "pair").glob("*.mp4"))), 2)

    def test_encoded_zero_bracket_failure_cannot_be_reported_as_pass(self):
        renderer = FakeRenderer()
        audits = [render.image_luma_stats([np.full((36, 64, 3), v, dtype=np.uint8)]) for v in (30, 45, 45)]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "failed"
            with patch.object(render, "measure_video_luma", side_effect=audits):
                with self.assertRaisesRegex(RuntimeError, "not published"):
                    render.render_matched_pair(dmx(55), dmx(155), None, 1.0, directory, renderer=renderer)
            result = json.loads((directory / "match.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["calibration"]["encoded_history"][0]["gain"], 0.)
            self.assertEqual(len(result["calibration"]["encoded_history"]), 1)
            self.assertTrue((directory / "correction__attempt00.mp4").is_file())
            self.assertEqual(len(list(directory.glob("*.mp4"))), 3)

    def test_encoded_search_is_bounded_and_retains_all_unmatched_trials(self):
        renderer = FakeRenderer()
        def quantized_observer(path):
            if Path(path).name.startswith("left"):
                light = 30
            else:
                light = 20 if renderer.streams[str(path)][0, 4] <= 80 else 40
            return render.image_luma_stats([np.full((36, 64, 3), light, np.uint8)])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "pair"
            with patch.object(render, "measure_video_luma", side_effect=quantized_observer):
                with self.assertRaisesRegex(RuntimeError, "not published"):
                    render.render_matched_pair(dmx(55), dmx(155), None, 1., directory, renderer=renderer)
            result = json.loads((directory / "match.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["calibration"]["encoded_history"]), 12)
            self.assertEqual(len(list(directory.glob("*.mp4"))), 14)

    def test_failed_encoded_calibration_does_not_publish_a_rating_package(self):
        renderer = FakeRenderer()
        audits = [render.image_luma_stats([np.full((36, 64, 3), v, np.uint8)]) for v in (30, 45, 45)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "source.wav"
            audio.write_bytes(b"fixture audio")
            candidates = [{"canonical_id": f"song{i}", "audio_path": audio, "offset_s": 0., "duration_s": 1.,
                           "arms": {"a": dmx(55), "b": dmx(155)}, "comparisons": [["a", "b"], ["b", "a"]]}
                          for i in range(2)]
            def audio_stub(source, target, offset, duration):
                target = Path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture excerpt")
                return target
            with (patch.object(render, "extract_audio_excerpt", side_effect=audio_stub),
                  patch.object(render, "measure_video_luma", side_effect=audits)):
                with self.assertRaisesRegex(RuntimeError, "not published"):
                    render.prepare_pairs(root, {}, candidates, n_songs=2, renderer=renderer)
            folder = root / "results" / "renders"
            self.assertFalse((folder / "pairs.json").exists())
            self.assertFalse((folder / "ratings.jsonl").exists())
            self.assertEqual(list(folder.glob("pair*__*.mp4")), [])
            self.assertEqual(len(list((folder / "materials" / "primary000").glob("*.mp4"))), 3)

    def test_encoded_video_audit_reads_all_frames_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "red.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi",
                "-i", "color=c=red:s=64x36:r=10:d=0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ], check=True)
            result = render.measure_video_luma(path)
            self.assertEqual(result["n_frames"], 5)
            self.assertEqual(result["audit"], "all encoded video frames")
            self.assertAlmostEqual(result["mean"], 0.2126, delta=0.005)

    def test_package_has_40_primary_4_repeat_4_sham_and_anonymous_consistent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "source.wav"
            audio.write_bytes(b"source-audio")
            candidates = [{"canonical_id": f"song{i:02d}", "audio_path": audio,
                           "offset_s": float(i), "duration_s": 1.0,
                           "arms": {"old": dmx(40), "new": dmx(80), "temporal": dmx(100), "pointwise": dmx(120)},
                           "arm_provenance": {name: {"arm": name, "training_seed": i % 5,
                                                      "generation_seed": 102 if name == "new" else 101,
                                                      "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest()}
                                              for name in ("old", "new", "temporal", "pointwise")},
                           "comparisons": [["old", "new"], ["pointwise", "temporal"]]}
                          for i in range(20)]

            def audio_stub(source, target, offset, duration):
                target = Path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f"audio {offset} {duration}".encode())
                return target

            def match_stub(a, b, audio, duration, directory, **kwargs):
                directory = Path(directory)
                directory.mkdir(parents=True)
                paths = []
                for label, frames in (("left", a), ("right", b)):
                    path = directory / f"{label}.mp4"
                    path.write_bytes(label.encode() + frames.tobytes())
                    paths.append(str(path))
                result = {"paths": paths, "encoded_video": [{"mean": 0.1}, {"mean": 0.1}],
                          "frames_sha256": [render._hash_frames(a), render._hash_frames(b)],
                          "symmetric_gap": 0.0, "passed": True}
                (directory / "match.json").write_text(json.dumps(result))
                return result

            with patch.object(render, "extract_audio_excerpt", side_effect=audio_stub), patch.object(render, "render_matched_pair", side_effect=match_stub):
                result = render.prepare_pairs(root, {"render": {"presentation_seed": 20}}, candidates, renderer=FakeRenderer())
            pairs = result["pairs"]
            self.assertEqual(len(pairs), 48)
            self.assertEqual(sum(p["kind"] == "primary" for p in pairs), 40)
            self.assertEqual(sum(p["kind"] == "repeat" for p in pairs), 4)
            self.assertEqual(sum(p["kind"] == "sham" for p in pairs), 4)
            directory = root / "results" / "renders"
            self.assertEqual((directory / "ratings.jsonl").read_bytes(), b"")
            by_id = {p["id"]: p for p in pairs}
            for pair in pairs:
                for label in ("A", "B"):
                    name = pair["videos"][label]
                    self.assertEqual(name, f"{pair['id']}__{label}.mp4")
                    self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), pair["video_sha256"][label])
                    provenance = pair["arm_provenance"][label]
                    self.assertEqual(provenance["arm"], pair["arm_identity"][label])
                    self.assertEqual(provenance["generation_seed"], 102 if provenance["arm"] == "new" else 101)
                    self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(provenance["arm"].encode()).hexdigest())
                if pair["kind"] == "sham":
                    self.assertEqual(pair["video_sha256"]["A"], pair["video_sha256"]["B"])
                    self.assertEqual(pair["arm_identity"]["A"], pair["arm_identity"]["B"])
                    self.assertEqual(pair["arm_provenance"]["A"], pair["arm_provenance"]["B"])
                if pair["kind"] == "repeat":
                    source = by_id[pair["source_primary_pair"]]
                    self.assertEqual(set(pair["video_sha256"].values()), set(source["video_sha256"].values()))
            (directory / "ratings.jsonl").write_text('{"rater":"anton"}\n')
            with self.assertRaises(FileExistsError):
                render.prepare_pairs(root, {}, candidates, renderer=FakeRenderer())
            self.assertEqual((directory / "ratings.jsonl").read_text(), '{"rater":"anton"}\n')

    def test_frozen_pairs_reject_missing_model_provenance_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            candidates = [{"canonical_id": str(i), "audio_path": audio, "offset_s": 0, "duration_s": 1,
                           "arms": {"a": dmx(1), "b": dmx(2)}, "comparisons": [["a", "b"], ["b", "a"]]}
                          for i in range(2)]
            with self.assertRaisesRegex(ValueError, "per-arm checkpoint provenance"):
                render.prepare_pairs(root, {"frozen": True}, candidates, n_songs=2, renderer=FakeRenderer())
            self.assertFalse((root / "results").exists())

    def test_duplicate_song_candidate_is_rejected_before_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "source.wav"
            audio.write_bytes(b"audio")
            candidate = {"canonical_id": "same", "audio_path": audio, "offset_s": 0, "duration_s": 1,
                         "arms": {"a": dmx(1), "b": dmx(2)}, "comparisons": [["a", "b"], ["b", "a"]]}
            with self.assertRaisesRegex(ValueError, "unique"):
                render.prepare_pairs(root, {}, [candidate, candidate], n_songs=2, renderer=FakeRenderer())
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
