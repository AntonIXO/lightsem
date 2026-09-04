"""Canonical rig and fixture roles.

Targets are defined on THIS rig so they are venue-independent; a rig adaptor
(Stage 4) re-patches the same roles onto whatever a venue actually owns. The
roles come straight from the body-following table in PLAN.md section 3.1 —
bass drives the head and torso, so the big slow fixtures follow the head;
hats drive the hands, so the small fast fixtures follow the hands.
"""

from __future__ import annotations

import numpy as np

from ..dmx.fixtures import Rig, club_rig

#: role -> which body channels drive it
ROLES = {
    "big_slow": ("head_speed", "section_energy"),   # washes
    "beam": ("vert_bounce", "lateral_pos", "spread"),  # moving heads
    "fast": ("hand_speed", "hit"),                  # strobe
    "event": ("footfall", "jump"),                  # blinders
    "back": ("section_energy",),                    # projector
}

KIND_TO_ROLE = {
    "wash": "big_slow",
    "moving_head": "beam",
    "strobe": "fast",
    "blinder": "event",
    "projector": "back",
}


def canonical_rig() -> Rig:
    """The rig every target is written against (6 heads, 4 washes, 2 blinders,
    strobe, projector = 14 fixtures, 16 renderer light slots)."""
    return club_rig()


def role_index(rig: Rig) -> dict[str, np.ndarray]:
    """role -> integer indices into ``rig.fixtures``."""
    out: dict[str, list[int]] = {r: [] for r in ROLES}
    for i, f in enumerate(rig.fixtures):
        out[KIND_TO_ROLE[f.kind]].append(i)
    return {k: np.array(v, dtype=int) for k, v in out.items()}


def stage_x(rig: Rig) -> np.ndarray:
    """Each fixture's stage-left/right position, normalised to [-1, 1].

    This is what makes "the dancer moved right, so the right-hand lights come
    up" expressible as arithmetic instead of a special case.
    """
    xs = np.array([f.position[0] for f in rig.fixtures], dtype=np.float64)
    span = max(float(np.abs(xs).max()), 1e-6)
    return xs / span
