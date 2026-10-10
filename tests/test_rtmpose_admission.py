import numpy as np
import pytest

from lightsem.experiments.native_dynamics.full_fit_admission import (
    iter_native_tracks, native_metadata,
)
from lightsem.experiments.native_dynamics.rtmpose_cache import CoachTracker, masked_track
from scripts.data.extract_rtmpose_native import dedupe_geometry


def test_bf16_batch_uses_validated_single_crop_contract(monkeypatch):
    import lightsem.experiments.native_dynamics.rtmpose_cache as cache

    class FakeBackend(cache.Backend):
        def __init__(self):
            self.seen = []

        def __call__(self, frame):
            self.seen.append(frame)
            return frame

    monkeypatch.setattr(cache, 'POSE_PROFILE', 'bf16')
    backend = FakeBackend()
    assert backend.batch(['a', 'b']) == ['a', 'b']
    assert backend.seen == ['a', 'b']


def test_geometry_dedupe_does_not_use_confidence():
    boxes = np.array([[10, 10, 110, 310], [12, 12, 108, 308], [300, 10, 400, 310]], float)
    keep = dedupe_geometry(boxes)
    assert keep.tolist() == [0, 2]


def test_tracker_keeps_people_separate_and_continues_after_gap():
    tracker = CoachTracker(maximum_gap_s=.5)
    styles = [np.zeros(48), np.zeros(48)]
    first = tracker.step(0., np.array([[10, 10, 110, 310], [300, 10, 400, 310]], float), styles)
    second = tracker.step(.1, np.array([[14, 10, 114, 310], [296, 10, 396, 310]], float), styles)
    after_gap = tracker.step(.3, np.array([[18, 10, 118, 310]], float), [styles[0]])
    assert first.tolist() == [0, 1]
    assert second.tolist() == [0, 1]
    assert after_gap.tolist() == [0]
    # A later observation after the identity timeout gets a new identity; no
    # averaging of the two people is possible through this state machine.
    fresh = tracker.step(1.0, np.array([[22, 10, 122, 310]], float), [styles[0]])
    assert fresh.tolist() == [2]


def test_tracker_splits_exactly_ambiguous_assignment():
    tracker = CoachTracker(maximum_gap_s=.5)
    style = np.zeros(48)
    boxes = np.array([[0, 0, 100, 300], [200, 0, 300, 300]], float)
    first = tracker.step(0., boxes, [style, style])
    # Centered detection has exactly equal geometric cost to both tracks.
    next_ids = tracker.step(.1, np.array([[100, 0, 200, 300]], float), [style])
    assert first.tolist() == [0, 1]
    assert next_ids[0] not in first


def test_masked_track_masks_finite_jump_and_does_not_forward_fill():
    times = np.arange(120, dtype=float) / 30.
    keypoints = np.zeros((120, 17, 2), np.float32)
    for i in range(120):
        keypoints[i, :, 0] = 100 + i
        keypoints[i, :, 1] = 300
        keypoints[i, [5, 6], 1] = 200
        keypoints[i, [11, 12], 1] = 400
    keypoints[60] += 500  # a finite-to-finite tracking jump
    boxes = np.tile(np.array([80, 10, 220, 500], np.float32), (120, 1))
    result, quality = masked_track(keypoints, boxes, times, 4., 720., confidence=np.ones((120, 17)))
    assert quality['joint_jump_edges'] > 0
    assert not result['mask'][60].any()
    assert result['mask'][59].sum() == 0 or result['mask'][61].sum() == 0
    assert np.all(result['pose'][60] == 0)
    assert np.allclose(result['time'], np.arange(len(result['time'])) / 30.)


def test_masked_track_can_use_explicit_confidence_gate_for_legacy_controls():
    times = np.arange(600, dtype=float) / 30.
    keypoints = np.zeros((600, 17, 2), np.float32)
    keypoints[..., 0] = np.arange(600)[:, None] * .1 + 100
    keypoints[..., 1] = 300
    keypoints[:, [5, 6], 1] = 200
    keypoints[:, [11, 12], 1] = 400
    boxes = np.tile(np.array([80, 10, 220, 500], np.float32), (600, 1))
    confidence = np.ones((600, 17))
    confidence[:, 9] = np.nan
    result, quality = masked_track(keypoints, boxes, times, 20., 720., confidence=confidence, confidence_gate=True)
    assert quality['confidence_filter'] is True
    assert quality['confidence_threshold'] == .60
    assert not result['mask'][:, 5].any()  # COCO index 9 maps to selected index 5
    assert result['mask'][:, 6].all()
    with pytest.raises(ValueError, match='torso'):
        masked_track(keypoints, boxes, times, 20., 720., confidence=np.zeros((600, 17)), confidence_gate=True)


def test_masked_track_default_keeps_scores_as_metadata_only():
    times = np.arange(600, dtype=float) / 30.
    keypoints = np.zeros((600, 17, 2), np.float32)
    keypoints[..., 0] = 100; keypoints[..., 1] = 300
    keypoints[:, [5, 6], 1] = 200; keypoints[:, [11, 12], 1] = 400
    boxes = np.tile(np.array([80, 10, 220, 500], np.float32), (600, 1))
    confidence = np.zeros((600, 17), np.float32)
    result, quality = masked_track(keypoints, boxes, times, 20., 720., confidence=confidence)
    assert quality['confidence_filter'] is False
    assert result['mask'][:, 5].all()


def _ragged_fixture(*, frames=(0, 2), ids=(4, 4), n_frames=4):
    rows = len(frames)
    keypoints = np.zeros((rows, 17, 2), np.float32)
    keypoints[..., 0] = np.arange(rows, dtype=np.float32)[:, None] + 100
    keypoints[..., 1] = 300
    boxes = np.tile(np.array([80, 10, 220, 500], np.float32), (rows, 1))
    return {
        't': np.arange(n_frames, dtype=np.float64) / 30,
        'K': keypoints,
        'C': np.ones((rows, 17), np.float32),
        'box': boxes,
        'track_id': np.asarray(ids, np.int32),
        'candidate': np.ones(rows, bool),
        'frame_index': np.asarray(frames, np.int32),
        'meta': np.asarray('{' + f'"n_frames": {n_frames}, "source_path": "/tmp/song.mkv", "source_sha256": "hash", "height": 720, "width": 1280' + '}'),
    }


def test_ragged_schema_uses_source_frames_and_detection_rows_separately():
    raw = _ragged_fixture()
    tracks = list(iter_native_tracks(raw))
    assert len(tracks) == 1
    track_id, keypoints, confidence, boxes = tracks[0]
    assert track_id == 4
    assert keypoints.shape == (4, 17, 2)
    assert boxes.shape == (4, 4)
    assert np.isfinite(keypoints[[0, 2]]).all()
    assert np.isnan(keypoints[[1, 3]]).all()
    assert np.isnan(confidence[[1, 3]]).all()
    assert np.isnan(boxes[[1, 3]]).all()
    assert native_metadata(raw)['source_path'] == '/tmp/song.mkv'


def test_ragged_schema_rejects_duplicate_track_frame_without_averaging():
    raw = _ragged_fixture(frames=(0, 0), ids=(4, 4), n_frames=2)
    with pytest.raises(ValueError, match='Multiple detections'):
        list(iter_native_tracks(raw))


def test_ragged_schema_preserves_missing_observation_and_continued_track():
    raw = _ragged_fixture(frames=(0, 2), ids=(4, 4), n_frames=3)
    track_id, keypoints, _, _ = next(iter(iter_native_tracks(raw)))
    assert track_id == 4
    # A later row with the same track ID is a continuation, while the absent
    # source frame remains missing. No value is copied into the gap.
    assert np.isnan(keypoints[1]).all()
    assert keypoints[0, 0, 0] != keypoints[2, 0, 0]


def test_ragged_schema_rejects_frame_index_outside_source_clock():
    raw = _ragged_fixture(frames=(0, 4), ids=(4, 4), n_frames=4)
    with pytest.raises(ValueError, match='Invalid frame/track assignment'):
        list(iter_native_tracks(raw))


def test_pose_audit_accepts_current_flat_metadata(tmp_path):
    from scripts.data.audit_pose_benchmark_20261009 import audit
    path = tmp_path / 'recording.npz'
    np.savez(path, **_ragged_fixture())
    report = audit(path)
    assert report['recording_id'] == 'recording'
    assert report['source_path'] == '/tmp/song.mkv'
    assert report['n_source_frames'] == 4
    assert report['n_detection_rows'] == 2
