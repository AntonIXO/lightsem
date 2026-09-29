"""Opt-in, stateless articulated floor/back-wall geometry candidate.

Synthetic engineering only. Gains below are fixed, not fitted on recordings.
This module does not smooth targets, control motors, or generate timed motion.
"""
from __future__ import annotations

import numpy as np

from .geometry import OFFSETS, PARENTS

# Rank is increasing canonical fixture x, never current body-side ordering.
REGIONS = (
    ("right_hand", (21,), (1.,)),
    ("right_leg", (11, 5), (.65, .35)),
    ("right_torso", (17, 15, 12), (.5, .3, .2)),
    ("left_torso", (16, 15, 12), (.5, .3, .2)),
    ("left_leg", (10, 4), (.65, .35)),
    ("left_hand", (20,), (1.,)),
)
NEUTRAL = np.zeros((22, 3), dtype=np.float64)
for _joint in range(1, 22):
    NEUTRAL[_joint] = NEUTRAL[PARENTS[_joint]] + OFFSETS[_joint]
NEUTRAL.setflags(write=False)

LANE_SCALE = .95
LANE_RESIDUAL_M = .65
LATERAL_GAIN = 2.8
FACING_X_M = .35
FACING_SURFACE_M = .35
ROOT_X_M = .25
HEIGHT_GAIN = 4.
DEPTH_GAIN = 3.
# u=0 is floor z=+1; u=7 is the floor/back-wall seam at z=-6.
BASE_UNFOLDED_M = (5.7, 3.2, 5.7)
SEAM_M = 7.
WALL_HEIGHT_M = 5.5


def _frame(hips, spine):
    """Orthonormal frame; reject degenerate anchors rather than invent facing."""
    norm = np.linalg.norm(hips, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate hip anchors; supply root_rotation")
    x = hips / norm
    up = spine - np.sum(spine*x, axis=-1, keepdims=True)*x
    norm = np.linalg.norm(up, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate spine anchor; supply root_rotation")
    y = up / norm
    return np.stack((x, y, np.cross(x, y)), axis=-1)


_REST_FRAME = _frame(NEUTRAL[1]-NEUTRAL[2], NEUTRAL[3])


def infer_root_rotation(body):
    """Recover Q from the three root-child offsets, including rest asymmetry.

    In this canonical FK, hip and spine1 *positions* depend only on the root
    rotation. Hip/spine articulation does not change these root-child offsets.
    Thus Q = observed_frame @ rest_frame.T, not a world-up yaw approximation.
    """
    body = np.asarray(body, dtype=np.float64)
    return _frame(body[:, 1]-body[:, 2], body[:, 3]-body[:, 0]) @ _REST_FRAME.T


def unfolded_targets(x, u):
    """Continuous finite surface chart, with a corner (not a jump) at u=7.

    Increasing u travels back across the floor, then up the back wall. Stage
    limits leave a margin inside the renderer's finite surfaces. For canonical
    head origins (above floor, in front of wall), these are first ray hits.
    """
    x, u = np.broadcast_arrays(np.asarray(x, float), np.asarray(u, float))
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError("Surface coordinates must be finite")
    u = np.clip(u, 0., SEAM_M + WALL_HEIGHT_M)
    return np.stack((np.clip(x, -7.5, 7.5), np.maximum(u-SEAM_M, 0.),
                     1.-np.minimum(u, SEAM_M)), axis=-1)


def stage_targets(body, fixture_x, rank, activity, cfg, *, root_rotation=None):
    """Return (T,3) surface points and root-rotation-corrected regional deltas.

    ``body`` is canonical world-space (T,22,3). Rank 0..5 keeps the six REGIONS
    fixed through all turns. Optional root_rotation is (3,3) or (T,3,3), acting
    on column vectors. Without it, derive Q from the hip/spine1 offsets.

    delta = regional_world - Q @ neutral_region = Q @ articulation_local.
    This removes the old (Q-I) @ rest term, not the world direction of gestures.
    Turns remain explicit: shared x = .35*forward_x, shared u = .35*(1-forward_z).
    Facing is bounded, continuous across yaw +/-pi, and never changes ownership.

    Canonical lane spacing is 1.9 m; bounded +/-0.65 m lateral articulation
    leaves >=0.6 m between lanes. Shared facing/root cues translate all lanes.
    These guarantees describe raw targets, not controller output or cone area.
    ``cfg.stage_base_unfolded_m`` optionally sets the three static role anchors.
    ``cfg.stage_lean_gain_m`` optionally restores rigid-body roll/pitch via the
    horizontal root-up vector. Zero preserves the prior candidate; positive
    values retain signed lean without bringing back amplified neutral offsets.
    Other geometric gains stay fixed; legacy surface_gain/torso_gain and timing
    options do not change this geometry.
    No sequence statistics, clock, periodic cue, or time-dependent fade is used.
    """
    body = np.asarray(body, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (22, 3) or not np.isfinite(body).all():
        raise ValueError("Expected finite canonical body (T,22,3)")
    if not isinstance(rank, (int, np.integer)) or not 0 <= rank < 6:
        raise ValueError("Fixture rank must be an integer in 0..5")
    if not np.isfinite(fixture_x) or abs(fixture_x) > 6.5:
        raise ValueError("Candidate supports fixture x within +/-6.5 m")
    activity = np.broadcast_to(np.asarray(activity, dtype=np.float64), (len(body),))
    if not np.isfinite(activity).all():
        raise ValueError("Activity must be finite")
    if root_rotation is None:
        q = infer_root_rotation(body)
    else:
        q = np.broadcast_to(np.asarray(root_rotation, float), (len(body), 3, 3))
        if (not np.isfinite(q).all() or
                not np.allclose(q @ np.swapaxes(q, -1, -2), np.eye(3), atol=1e-5) or
                not np.allclose(np.linalg.det(q), 1., atol=1e-5)):
            raise ValueError("root_rotation must contain proper rotation matrices")
    _, ids, weights = REGIONS[rank]
    weights = np.asarray(weights)
    root = body[:, 0]
    regional = np.sum((body[:, ids]-root[:, None])*weights[None, :, None], axis=1)
    rest = np.sum(NEUTRAL[list(ids)]*weights[:, None], axis=0)
    delta = regional - np.einsum("tij,j->ti", q, rest)
    amplitude = .25 + .75*np.clip(activity, 0., 1.)
    # Bounded articulation cannot reverse the canonical fixture lane order.
    lateral = LANE_RESIDUAL_M*np.tanh(amplitude*LATERAL_GAIN*delta[:, 0]/LANE_RESIDUAL_M)
    forward = q[:, :, 2]
    lean_gain = float(getattr(cfg, "stage_lean_gain_m", 0.))
    if not np.isfinite(lean_gain) or not 0 <= lean_gain <= 2.:
        raise ValueError("Stage lean gain must be finite and within [0,2] metres")
    up = q[:, :, 1]
    shared_x = (FACING_X_M*forward[:, 0] + ROOT_X_M*np.tanh(root[:, 0]/3.)
                + amplitude*lean_gain*up[:, 0])
    x = LANE_SCALE*fixture_x + shared_x + lateral
    pair = min(rank, 5-rank)
    base = np.asarray(getattr(cfg, "stage_base_unfolded_m", BASE_UNFOLDED_M),float)
    if base.shape != (3,) or not np.isfinite(base).all() or np.any((base<0)|(base>SEAM_M+WALL_HEIGHT_M)):
        raise ValueError("Expected three finite stage-role anchors within the chart")
    # Positive height and backward depth both advance toward/up the wall.
    # This is still a 3D->2D projection, not a complete pose representation.
    u = (base[pair] + FACING_SURFACE_M*(1.-forward[:, 2])
         + amplitude*(HEIGHT_GAIN*delta[:, 1] - DEPTH_GAIN*delta[:, 2])
         + .8*np.tanh((root[:, 1]-.95)/.4)
         - .4*np.tanh(root[:, 2]/3.) - amplitude*lean_gain*up[:, 2])
    return unfolded_targets(x, u), delta
