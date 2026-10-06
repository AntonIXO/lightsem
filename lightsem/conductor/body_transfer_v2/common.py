"""CPU-only, clock-free contracts shared by the fifteen transfer experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .skeleton import PARENTS, NAMES, OFFSETS


def unit(v, fallback=None, eps=1e-8):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if fallback is None:
        fallback = np.array([0., 1., 0.])
    return np.where(norm > eps, v / np.maximum(norm, eps), np.broadcast_to(fallback, v.shape))


def derivative(value, fps):
    """Physical backward difference; a repeated pose has exactly zero derivative."""
    return np.diff(value, axis=0, prepend=value[:1]) * fps


def canonical_pose():
    result = np.zeros((22, 3), dtype=float)
    result[0, 1] = 1.
    for j in range(1, 22):
        result[j] = result[PARENTS[j]] + OFFSETS[j]
    return result


REFERENCE = canonical_pose()
BODY_PARTS = ((0, 3, 6, 9, 12, 15), (13, 16, 18, 20),
              (14, 17, 19, 21), (1, 4, 7, 10), (2, 5, 8, 11))
ROLE_ORDER = (20, 21, 7, 8, 15, 9, 18, 19, 4, 5, 16, 17, 0,
              10, 11, 1, 2, 3, 6, 12, 13, 14)


@dataclass(frozen=True)
class Features:
    positions: np.ndarray  # (T,22,3), only horizontal root travel removed
    reference: np.ndarray  # (22,3), fixed canonical pose, shared between songs
    displacement: np.ndarray  # positions-reference
    velocity: np.ndarray
    bones: np.ndarray  # (T,22,3); root uses torso vector
    reference_bones: np.ndarray
    fps: float


def prepare_features(body, fps=40.):
    body = np.asarray(body, dtype=float)
    if body.ndim != 3 or body.shape[1:] != (22, 3) or len(body) < 2 or not np.isfinite(body).all():
        raise ValueError('Expected finite body (T>=2,22,3)')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite fps required')
    p = body - body[:, :1] * np.array([1., 0., 1.])
    parent = np.maximum(PARENTS, 0)
    bones = p - p[:, parent]
    bones[:, 0] = p[:, 9] - p[:, 0]
    rb = REFERENCE - REFERENCE[parent]
    rb[0] = REFERENCE[9] - REFERENCE[0]
    return Features(p, REFERENCE.copy(), p - REFERENCE, derivative(p, fps),
                    bones, rb, float(fps))


@dataclass
class FieldResult:
    """22 anatomical field samples; coordinates are metre-like deformations.

    Each algorithm supplies a distinct spatial representation. No output is
    animated by index/time. attribution[j] lists source joints used by sample j.
    activity optionally supplies a bounded pose/speed-derived value in [0,1].
    """
    points: np.ndarray
    attribution: list[list[int]]
    meta: dict = field(default_factory=dict)
    activity: np.ndarray | None = None
    absolute_points: np.ndarray | None = None
    absolute_reference: np.ndarray | None = None
    shared_groups: tuple[tuple[int, ...], ...] = ()

    def validate(self, frames):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.shape != (frames, 22, 3) or not np.isfinite(self.points).all():
            raise ValueError('Mapper must return finite (T,22,3) field')
        if len(self.attribution) != 22 or any(not a or any(j not in range(22) for j in a) for a in self.attribution):
            raise ValueError('Every field sample requires valid source attribution')
        if self.activity is not None:
            self.activity = np.asarray(self.activity, dtype=float)
            if self.activity.shape != (frames, 22) or not np.isfinite(self.activity).all() or np.any((self.activity < 0) | (self.activity > 1)):
                raise ValueError('Activity must be finite (T,22) in [0,1]')
        if self.shared_groups:
            self.absolute_points = np.asarray(self.absolute_points, dtype=float)
            self.absolute_reference = np.asarray(self.absolute_reference, dtype=float)
            if self.absolute_points.shape != (frames, 22, 3) or self.absolute_reference.shape != (22, 3):
                raise ValueError('Shared structures require absolute points and a fixed reference')
            if not np.isfinite(self.absolute_points).all() or not np.isfinite(self.absolute_reference).all():
                raise ValueError('Shared structures must be finite')
            joints = [j for group in self.shared_groups for j in group]
            if len(joints) != len(set(joints)) or any(j not in range(22) for j in joints):
                raise ValueError('Shared anatomical groups must be disjoint valid joint sets')
        return self


def direct_attribution():
    # Horizontal recentering means every position depends on the root as well.
    return [sorted({0, j}) for j in range(22)]


def all_attribution():
    return [list(range(22)) for _ in range(22)]


def skeleton_laplacian():
    adj = np.zeros((22, 22))
    for j, parent in enumerate(PARENTS[1:], 1):
        adj[j, parent] = adj[parent, j] = 1.
    return np.diag(adj.sum(1)) - adj


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)
