"""Failure-mode tests for versioned Stage 11 preparation (CPU only)."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch

import numpy as np

from lightsem.experiments.stage11.stage11_data import (
    active_fingerprint, active_span, atomic_copy, canonicalize, fingerprint_candidates, fingerprint_score,
    load_manifest, normalize_scene, normalize_title, reduce_histograms, _verify_artifacts,
    recording_mask_dilation, resize_mask, sha256,
)


class Stage11DataTests(unittest.TestCase):
    def test_interrupted_cache_copy_preserves_completed_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp)/'source', Path(tmp)/'destination'
            source.write_bytes(b'complete new data')
            destination.write_bytes(b'previous complete data')
            def failed_copy(src, dst):
                Path(dst).write_bytes(b'partial')
                raise OSError('interrupted copy')
            with patch('lightsem.experiments.stage11.stage11_data.shutil.copyfile', failed_copy):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    atomic_copy(source, destination)
            self.assertEqual(destination.read_bytes(), b'previous complete data')
            atomic_copy(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_active_offset_and_coach_coverage(self):
        k = np.full((100, 2, 17, 2), np.nan)
        k[20:90, 0] = 1
        k[40:90, 1] = 2
        first, last, offset, duration, cov = active_span(k, np.arange(100)/20 + .05, 20)
        self.assertEqual((first, last), (20, 90))
        self.assertAlmostEqual(offset, 1.05)
        self.assertAlmostEqual(duration, 3.5)
        np.testing.assert_allclose(cov, [1., 50/70])

    def test_scene_overlap_is_recording_relative(self):
        rng = np.random.default_rng(1)
        raw = rng.uniform(size=(300, 6)).astype(np.float32)
        y, mean, scale = normalize_scene(raw)
        np.testing.assert_allclose(y[:150][100:150], y[100:250][:50])
        np.testing.assert_array_equal(y[:, [0, 1, 2, 5]], raw[:, [0, 1, 2, 5]])
        np.testing.assert_allclose(y*scale+mean, raw, atol=1e-7)
        np.testing.assert_allclose(y[:, 3:5].mean(0), 0., atol=2e-7)
        constant, _, scale = normalize_scene(np.ones((40, 6), np.float32))
        np.testing.assert_array_equal(constant[:, 3:5], 0.)
        np.testing.assert_allclose(scale[3:5], 1/255)

    def test_pose_box_coordinates_are_original_resolution(self):
        boxes = np.array([[1920., 1080., 2880., 1620.]])
        mask = resize_mask(boxes, (3840, 2160), (640, 360), include_ui=False)
        self.assertEqual(mask[200, 400], 0)
        self.assertEqual(mask[50, 50], 255)
        # An anisotropic test would catch accidentally using width for both axes.
        mask = resize_mask(np.array([[50., 20., 100., 40.]]), (200, 100), (100, 100), include_ui=False)
        self.assertEqual(mask[30, 40], 0)
        self.assertEqual(mask[10, 40], 255)

    def test_entropy_concentration_is_preserved(self):
        hist = np.zeros((2, 3, 256), np.float32)
        hist[0, 0, 0] = 180
        hist[1, 0, :180] = 1
        hist[:, 1, 255] = 180
        hist[:, 2, 128] = 180
        y = reduce_histograms(hist)
        np.testing.assert_allclose(y[:, 5], [1., 0.], atol=1e-6)
        np.testing.assert_allclose(y[:, 2], 1.)
        np.testing.assert_allclose(y[:, 3:5], 128/255)

    def test_wide_coach_uses_fixed_recording_padding(self):
        boxes = np.array([[[37., 29.5, 625.5, 480.]], [[150., 30., 500., 480.]]])
        dilation, visible = recording_mask_dilation(boxes, np.array([0., .1]), np.array([0., .1]),
                                                     (640, 480), (640, 480))
        self.assertEqual(dilation, 0.)
        self.assertGreaterEqual(visible, 100)
        # Removing padding never unmasks any part of a detected coach box.
        mask = resize_mask(boxes[0], (640, 480), (640, 480), dilation=dilation)
        self.assertFalse(mask[30:480, 37:626].any())

    def test_no_observable_background_fails_before_decode(self):
        with self.assertRaisesRegex(ValueError, 'fewer than 100'):
            recording_mask_dilation(np.array([[[0., 0., 640., 480.]]]), np.array([0.]),
                                     np.array([0.]), (640, 480), (640, 480))

    def test_fingerprint_alignment_tolerates_menu_offsets(self):
        rng = np.random.default_rng(4)
        shared = rng.integers(0, 2**32, 900, dtype=np.uint32)
        a = np.r_[rng.integers(0, 2**32, 80, dtype=np.uint32), shared]
        b = np.r_[rng.integers(0, 2**32, 11, dtype=np.uint32), shared]
        scores = fingerprint_candidates({'a': a, 'b': b})
        self.assertIn(('a', 'b'), scores)
        self.assertLessEqual(abs(scores[('a', 'b')][0]-69), 2)
        self.assertEqual(fingerprint_score(a, b, 69)['bit_error'], 0.)
        unrelated = rng.integers(0, 2**32, 900, dtype=np.uint32)
        self.assertGreater(fingerprint_score(shared, unrelated, 0)['bit_error'], .4)
        self.assertEqual(fingerprint_score(shared[:30], shared, 0)['bit_error'], 1.)

    def test_fingerprints_exclude_shared_menu_audio(self):
        fp = np.arange(1000, dtype=np.uint32)
        selected = active_fingerprint(fp, 20., 80.)
        self.assertGreaterEqual(int(selected[0])*1365/11025, 20.)
        self.assertLessEqual(int(selected[-1])*1365/11025+28666/11025, 80.)

    def test_duplicate_group_cannot_cross_old_new_split(self):
        rng = np.random.default_rng(9)
        rows = [
            {'recording_id': 'a-old', 'source_path': '/x/Unknown OLD.mp4', 'is_old': True, 'coaches': [{}]},
            {'recording_id': 'b-new', 'source_path': '/x/Bad Boy.mp4', 'is_old': False, 'coaches': [{}]},
            {'recording_id': 'c-new', 'source_path': '/x/Drum Go Dum.mp4', 'is_old': False, 'coaches': [{}, {}]},
            {'recording_id': 'd-new', 'source_path': '/x/Blinding Lights.mp4', 'is_old': False, 'coaches': [{}]},
        ]
        fp = {r['recording_id']: rng.integers(0, 2**32, 500, dtype=np.uint32) for r in rows}
        fp['b-new'] = fp['a-old'].copy()
        result = canonicalize(rows, {}, fp, test_count=1, val_count=1)
        self.assertEqual(result['records']['a-old']['canonical_id'], result['records']['b-new']['canonical_id'])
        self.assertEqual(result['records']['b-new']['partition'], 'train')
        self.assertEqual({result['records'][r]['partition'] for r in ('c-new', 'd-new')}, {'test', 'val'})
        reversed_result = canonicalize(list(reversed(rows)), {}, dict(reversed(list(fp.items()))), test_count=1, val_count=1)
        self.assertEqual(result['test'], reversed_result['test'])
        self.assertEqual(result['val'], reversed_result['val'])

    def test_explicit_protected_partition_propagates_through_duplicate(self):
        rng = np.random.default_rng(10)
        rows = [
            {'recording_id': 'old-val', 'source_path': '/x/Bad Boy.mp4', 'is_old': True,
             'partition': 'val', 'coaches': [{}]},
            {'recording_id': 'new-copy', 'source_path': '/x/Bad Boy alt.mp4', 'is_old': False,
             'coaches': [{}]},
            {'recording_id': 'new-test', 'source_path': '/x/Drum Go Dum.mp4', 'is_old': False,
             'coaches': [{}]},
            {'recording_id': 'new-val', 'source_path': '/x/Blinding Lights.mp4', 'is_old': False,
             'coaches': [{}]},
        ]
        fp = {r['recording_id']: rng.integers(0, 2**32, 500, dtype=np.uint32) for r in rows}
        fp['new-copy'] = fp['old-val'].copy()
        result = canonicalize(rows, {}, fp, test_count=1, val_count=1)
        self.assertEqual(result['records']['old-val']['partition'], 'val')
        self.assertEqual(result['records']['new-copy']['partition'], 'val')
        self.assertEqual(result['protected_group_counts']['val'], 1)

    def test_title_containment_requires_audio_evidence(self):
        rng = np.random.default_rng(11)
        rows = [
            {'recording_id': 'a', 'source_path': '/x/Make Me Feel.mp4', 'is_old': True, 'coaches': [{}]},
            {'recording_id': 'b', 'source_path': '/x/You Make Me Feel.mp4', 'is_old': True, 'coaches': [{}]},
            {'recording_id': 'c', 'source_path': '/x/Bad Boy.mp4', 'is_old': False, 'coaches': [{}]},
            {'recording_id': 'd', 'source_path': '/x/Drum Go Dum.mp4', 'is_old': False, 'coaches': [{}]},
        ]
        fp = {r['recording_id']: rng.integers(0, 2**32, 500, dtype=np.uint32) for r in rows}
        result = canonicalize(rows, {}, fp, test_count=1, val_count=1)
        self.assertNotEqual(result['records']['a']['canonical_id'], result['records']['b']['canonical_id'])
        self.assertFalse(any(e['reason'] == 'conservative_title_containment' for e in result['edges']))

    def test_resume_artifact_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'artifact.bin'
            path.write_bytes(b'original')
            expected = sha256(path)
            _verify_artifacts({str(path): expected})
            path.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                _verify_artifacts({str(path): expected})

    def test_inadequate_verified_holdout_fails(self):
        rows = [{'recording_id': 'unknown', 'source_path': '/x/Something unclear.mp4',
                 'is_old': False, 'coaches': [{}]}]
        with self.assertRaisesRegex(ValueError, 'require 80'):
            canonicalize(rows, {}, {'unknown': np.arange(500, dtype=np.uint32)})

    def test_frozen_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            data.mkdir()
            path = data/'manifest.json'
            path.write_text(json.dumps({'schema_version': 1, 'recordings': []}))
            (data/'manifest.sha256.json').write_text(json.dumps({'sha256': sha256(path)}))
            self.assertEqual(load_manifest(root)['recordings'], [])
            path.write_text(json.dumps({'schema_version': 1, 'recordings': [1]}))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_manifest(root)

    def test_missing_manifest_checksum_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data').mkdir()
            (root/'data'/'manifest.json').write_text(json.dumps({'schema_version': 1, 'recordings': []}))
            with self.assertRaisesRegex(ValueError, 'checksum sidecar is missing'):
                load_manifest(root)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_decoder_clock_after_fractional_seek(self):
        # Pixel intensity encodes source frame number. Nearest FPS rounding
        # drifts by 43 ms here; upward rounding keeps each sample within a frame.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'clock.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'nullsrc=s=32x32:r=30:d=2', '-vf', 'geq=r=2*N:g=2*N:b=2*N',
                            '-c:v', 'ffv1', '-pix_fmt', 'gbrp', '-threads', '1', str(path)], check=True)
            offset = .09
            result = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-ss', str(offset),
                                     '-i', str(path), '-t', '1', '-an', '-vf',
                                     'fps=10:round=up:start_time=0,scale=32:32', '-threads', '1',
                                     '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
                                    capture_output=True, check=True)
            frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, 32, 32, 3)
            source_times = frames[:, 0, 0, 0]/2/30
            expected = offset+np.arange(len(frames))/10
            self.assertLessEqual(float(np.max(abs(source_times-expected))), 1/30)

    def test_title_normalization_removes_only_boilerplate(self):
        title = normalize_title('Just Dance 2020 - Blinding Lights (Extreme) [ABC123]')
        self.assertEqual(title, 'blinding lights')


if __name__ == '__main__':
    unittest.main()
