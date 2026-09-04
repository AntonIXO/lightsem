"""COCO-17 skeleton geometry: body frame, normalisation, smoothing.

Everything downstream (descriptors, light targets, the audit) reads positions
through :func:`prepare`, so the invariances promised in PLAN.md 2.1 are
enforced in exactly one place:

* **size invariance** — positions are divided by the dancer's own torso length,
  so a tall and a short dancer performing the same move give the same numbers.
  Unit is TL (torso lengths); speeds are TL/s.
* **floor invariance** — the vertical axis is shifted so the floor (2nd
  percentile of ankle height) sits at 0. AIST++ keypoints are NOT floor-aligned
  (measured: ankle y runs 72-90 in a clip whose nose sits at 202), so jump and
  footfall thresholds computed on raw coordinates would be nonsense.
* **orientation invariance** — coordinates are rotated into a clip-stable body
  frame (lateral, up, forward) built from the dancer's mean hip axis, so
  "the dancer moved to their right" is a fact about the dancer and not about
  where the capture rig happened to point.

Filtering is zero-phase (``filtfilt``). That is deliberately non-causal and
deliberately fine: these are *training targets* computed offline from motion
capture. The causal constraint in this project applies to the audio side at
inference, never to the target extraction. Do not copy this filter into
``lightsem/audio/online.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

# --- COCO-17 joint indices (AIST++ keypoints3d / keypoints3d_optim layout) ---
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
N_JOINTS = 17

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]

MIRROR_PAIRS = (
    (LEYE, REYE), (LEAR, REAR), (LSHO, RSHO), (LELB, RELB),
    (LWRI, RWRI), (LHIP, RHIP), (LKNE, RKNE), (LANK, RANK),
)

HEAD = (NOSE, LEYE, REYE, LEAR, REAR)
HANDS = (LWRI, RWRI)
FEET = (LANK, RANK)
UPPER = (NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI)
LOWER = (LHIP, RHIP, LKNE, RKNE, LANK, RANK)

# Axis order inside a prepared Skeleton.
LAT, UP, FWD = 0, 1, 2


@dataclass
class Skeleton:
    """A prepared motion clip in the body-stage frame, in torso lengths."""

    pos: np.ndarray          # (T, 17, 3) world-ish: lateral / up / forward, TL
    fps: float
    torso_cm: float          # the divisor, kept so cm can be recovered
    right_axis: np.ndarray   # (3,) unit vector in source coords = dancer's right
    fwd_axis: np.ndarray     # (3,)
    up_axis: np.ndarray      # (3,)
    floor_src: float         # floor height in source units, before shifting
    name: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.pos.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps

    @property
    def root(self) -> np.ndarray:
        """Mid-hip trajectory, (T, 3)."""
        return 0.5 * (self.pos[:, LHIP] + self.pos[:, RHIP])

    @property
    def mid_shoulder(self) -> np.ndarray:
        return 0.5 * (self.pos[:, LSHO] + self.pos[:, RSHO])

    @property
    def local(self) -> np.ndarray:
        """Root-relative positions, (T, 17, 3) — translation removed."""
        return self.pos - self.root[:, None, :]


def torso_length(kp: np.ndarray) -> float:
    """Median mid-shoulder to mid-hip distance. The size normaliser."""
    mid_sh = 0.5 * (kp[:, LSHO] + kp[:, RSHO])
    mid_hip = 0.5 * (kp[:, LHIP] + kp[:, RHIP])
    d = np.linalg.norm(mid_sh - mid_hip, axis=-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("cannot measure torso length: no finite frames")
    return float(np.median(d))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("cannot normalise a zero-length axis")
    return v / n


def body_axes(kp: np.ndarray, up_hint: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip-stable (right, up, forward) axes in the source coordinate frame.

    ``right`` is the time-averaged hip axis (left hip -> right hip) projected
    onto the horizontal plane; ``forward`` completes a right-handed frame. The
    average is over the whole clip on purpose: a per-frame body frame would
    cancel exactly the turning motion the lights are supposed to follow.
    """
    up = np.zeros(3)
    up[up_hint] = 1.0
    hip_axis = kp[:, RHIP] - kp[:, LHIP]           # (T, 3)
    hip_axis = hip_axis - (hip_axis @ up)[:, None] * up[None, :]
    mean_axis = hip_axis.mean(axis=0)
    if np.linalg.norm(mean_axis) < 1e-6:
        # Dancer turned symmetrically over the clip: fall back to the dominant
        # horizontal direction of the hip axis (principal component).
        u, s, vt = np.linalg.svd(hip_axis - hip_axis.mean(axis=0), full_matrices=False)
        mean_axis = vt[0]
    right = _unit(mean_axis)
    fwd = _unit(np.cross(up, right))
    return right, up, fwd


def prepare(
    kp: np.ndarray,
    fps: float,
    smooth_hz: float = 12.0,
    up_hint: int = 1,
    name: str = "",
) -> Skeleton:
    """Raw (T, 17, 3) keypoints -> normalised :class:`Skeleton`.

    ``smooth_hz`` is a 4th-order zero-phase Butterworth low pass. 12 Hz keeps
    footfall impacts (which live around 5-10 Hz) while removing marker jitter
    that would otherwise dominate jerk-based Laban proxies.
    """
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1] != N_JOINTS or kp.shape[2] != 3:
        raise ValueError(f"expected (T,17,3) keypoints, got {kp.shape}")
    if not np.isfinite(kp).all():
        raise ValueError("non-finite keypoints: interpolate before prepare()")
    if kp.shape[0] < 8:
        raise ValueError(f"clip too short to filter: {kp.shape[0]} frames")

    right, up, fwd = body_axes(kp, up_hint=up_hint)
    R = np.stack([right, up, fwd], axis=0)          # rows are the new axes
    pos = kp @ R.T                                  # (T, 17, 3) in body frame

    floor = float(np.percentile(pos[:, FEET, UP], 2.0))
    pos[:, :, UP] -= floor

    tl = torso_length(kp)
    pos /= tl

    if smooth_hz and smooth_hz < fps / 2:
        nyq = fps / 2.0
        b, a = butter(4, smooth_hz / nyq, btype="low")
        pad = 3 * max(len(a), len(b))
        if pos.shape[0] > pad:
            flat = pos.reshape(pos.shape[0], -1)
            pos = filtfilt(b, a, flat, axis=0).reshape(pos.shape)

    return Skeleton(
        pos=pos, fps=float(fps), torso_cm=tl,
        right_axis=right, fwd_axis=fwd, up_axis=up, floor_src=floor, name=name,
    )


def mirror(sk: Skeleton) -> Skeleton:
    """Mirror a prepared clip: negate the lateral axis and swap L/R joints.

    Used as an augmentation and as the check that no descriptor secretly
    depends on which way a dancer happens to face.
    """
    pos = sk.pos.copy()
    pos[:, :, LAT] *= -1.0
    for a, b in MIRROR_PAIRS:
        pos[:, [a, b]] = pos[:, [b, a]]
    return Skeleton(
        pos=pos, fps=sk.fps, torso_cm=sk.torso_cm,
        right_axis=-sk.right_axis, fwd_axis=sk.fwd_axis, up_axis=sk.up_axis,
        floor_src=sk.floor_src, name=sk.name + "_mirror",
    )


def derivative(x: np.ndarray, fps: float, order: int = 1) -> np.ndarray:
    """Central-difference time derivative along axis 0, ``order`` times."""
    out = np.asarray(x, dtype=np.float64)
    for _ in range(order):
        out = np.gradient(out, 1.0 / fps, axis=0, edge_order=1)
    return out


def resample(x: np.ndarray, fps_in: float, fps_out: float, n_out: int | None = None) -> np.ndarray:
    """Linear resample of a (T, ...) series onto a new frame rate."""
    x = np.asarray(x, dtype=np.float64)
    t_in = np.arange(x.shape[0]) / fps_in
    if n_out is None:
        n_out = int(round(x.shape[0] / fps_in * fps_out))
    t_out = np.arange(n_out) / fps_out
    flat = x.reshape(x.shape[0], -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_out, t_in, flat[:, j])
    return out.reshape((n_out,) + x.shape[1:])


def smooth_ma(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Centred moving average with edge-reflected padding (window in seconds)."""
    n = max(int(round(window_s * fps)), 1)
    if n <= 1:
        return np.asarray(x, dtype=np.float64)
    if n % 2 == 0:
        n += 1
    pad = n // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="reflect")
    k = np.ones(n) / n
    return np.convolve(xp, k, mode="valid")
