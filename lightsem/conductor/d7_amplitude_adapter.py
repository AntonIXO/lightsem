"""Bone-preserving native-body amplitude correction calibrated on D7.

The correction acts on joint directions before body-to-light mapping. It does
not multiply Cartesian joint coordinates, stretch limbs, add a clock, or alter
the root trajectory. Each bone keeps the source frame's length while its
angular excursion is expanded by a fixed per-joint factor measured against the
sealed D7 working reference for the same song.
"""
from __future__ import annotations

import numpy as np
from .body_transfer_v2.skeleton import PARENTS


def _unit(value, eps=1e-9):
    value = np.asarray(value, dtype=float)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def _bone_vectors(body):
    value = np.asarray(body, dtype=float)
    parent = np.maximum(np.asarray(PARENTS, dtype=int), 0)
    vectors = value - value[:, parent]
    vectors[:, 0] = value[:, 9] - value[:, 0]
    return vectors


def _angular_excursion(body):
    bones = _bone_vectors(body)
    directions = _unit(bones)
    neutral = _unit(np.mean(directions, axis=0, keepdims=True))[0]
    angle = np.arccos(np.clip(np.sum(directions * neutral[None], axis=-1), -1., 1.))
    return np.ptp(angle, axis=0), directions, neutral


def root_relative_range(body):
    value = np.asarray(body, dtype=float)
    if value.ndim != 3 or value.shape[1:] != (22, 3) or not np.isfinite(value).all():
        raise ValueError("Expected finite body with shape (T,22,3)")
    return float(np.ptp(value - value[:, :1], axis=0).mean())


def _rotate_from_neutral(neutral, direction, angle_scale):
    """Extrapolate a direction on the sphere without changing its norm."""
    cosine = np.clip(np.sum(direction * neutral[None], axis=-1), -1., 1.)
    angle = np.arccos(cosine)
    axis = np.cross(neutral[None], direction)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    fallback = np.cross(neutral, np.roll(neutral, 1))[None]
    fallback = fallback / np.maximum(np.linalg.norm(fallback, axis=-1, keepdims=True), 1e-9)
    axis = np.where(axis_norm > 1e-8, axis / np.maximum(axis_norm, 1e-9), fallback)
    theta = angle * angle_scale
    c, s = np.cos(theta)[..., None], np.sin(theta)[..., None]
    n = neutral[None]
    return _unit(n * c + np.cross(axis, n) * s + axis * np.sum(axis * n, axis=-1, keepdims=True) * (1. - c))


def calibrate_to_d7(body, reference, *, min_gain=1.0, max_gain=2.5):
    """Amplify source pose articulation while preserving every source bone length."""
    value = np.asarray(body, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if (value.ndim != 3 or ref.ndim != 3 or value.shape[1:] != (22, 3)
            or ref.shape[1:] != (22, 3) or len(value) < 2 or len(ref) < 2
            or not np.isfinite(value).all() or not np.isfinite(ref).all()):
        raise ValueError("Expected finite body/reference arrays with shape (T,22,3)")
    source_range, directions, neutral = _angular_excursion(value)
    reference_range, _, _ = _angular_excursion(ref)
    gains = np.clip(reference_range / np.maximum(source_range, 1e-5), min_gain, max_gain)
    gains[0] = 1.0
    amplified = np.empty_like(value)
    amplified[:, 0] = value[:, 0]
    source_bones = _bone_vectors(value)
    lengths = np.linalg.norm(source_bones, axis=-1)
    parent = np.maximum(np.asarray(PARENTS, dtype=int), 0)
    for joint in range(1, 22):
        amplified_dir = _rotate_from_neutral(neutral[joint], directions[:, joint], gains[joint])
        amplified[:, joint] = amplified[:, parent[joint]] + amplified_dir * lengths[:, joint, None]
    if not np.isfinite(amplified).all():
        raise ValueError("Bone-preserving amplitude correction produced nonfinite values")
    return amplified, {
        "source_angular_range_rad": source_range.tolist(),
        "reference_d7_angular_range_rad": reference_range.tolist(),
        "per_joint_gain": gains.tolist(), "gain_min": float(np.min(gains[1:])),
        "gain_max": float(np.max(gains[1:])), "bone_lengths_preserved_per_frame": True,
        "root_translation_preserved": True, "timestamps_preserved": True,
        "temporal_controller": False, "training_checkpoint_changed": False,
        "scope": "Fixed same-song D7 angular calibration in native body space; no Cartesian stretch."
    }


def body_amplitude_metrics(body, source, reference):
    value, src, ref = map(lambda x: np.asarray(x, dtype=float), (body, source, reference))
    source_range, _, _ = _angular_excursion(src)
    candidate_range, _, _ = _angular_excursion(value)
    reference_range, _, _ = _angular_excursion(ref)
    ratios = candidate_range[1:] / np.maximum(reference_range[1:], 1e-5)
    lengths = np.linalg.norm(_bone_vectors(value), axis=-1)
    source_lengths = np.linalg.norm(_bone_vectors(src), axis=-1)
    bone_error = np.abs(lengths[:, 1:] - source_lengths[:, 1:])
    return {
        "per_joint_angular_range_ratio_to_d7": ratios.tolist(),
        "per_joint_angular_range_ratio_min": float(np.min(ratios)),
        "per_joint_angular_range_ratio_max": float(np.max(ratios)),
        "bone_length_error_p95_m": float(np.quantile(bone_error, .95)),
        "bone_length_error_max_m": float(np.max(bone_error)),
        "candidate_root_relative_range_m": root_relative_range(value),
        "source_root_relative_range_m": root_relative_range(src),
        "d7_root_relative_range_m": root_relative_range(ref),
        "finite": bool(np.isfinite(value).all()), "constant_body": bool(np.all(value == value[:1])),
    }
