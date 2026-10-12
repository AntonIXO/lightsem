"""Capability-aware expansion of learned group state to actual DMX bytes."""
from __future__ import annotations

import numpy as np

from lightsem.dmx.encode import pack_universe
from lightsem.dmx.fixtures import Rig


def scene_to_dmx(scene: np.ndarray, body: np.ndarray, rig: Rig, event: np.ndarray | None = None) -> np.ndarray:
    """Expand ``(hue_cos,hue_sin,sat,value,p90,concentration)`` to a rig.

    This renderer is intentionally simple and deterministic.  Colour/value and
    accents come from the learned scene heads; moving-head aim is conditioned on
    the body so a frozen body cannot create temporal pan/tilt motion.  It is a
    transfer adapter, not pan/tilt ground truth from the public corpora.
    """
    scene, body = np.asarray(scene, float), np.asarray(body, float)
    if scene.ndim != 2 or scene.shape[1] != 6 or body.ndim != 3 or body.shape[2] != 3 or len(scene) != len(body):
        raise ValueError("scene/body shapes must be (T,6) and (T,J,3)")
    if not np.isfinite(scene).all() or not np.isfinite(body).all():
        raise ValueError("scene/body must be finite")
    T, J = len(scene), body.shape[1]
    if event is None: event = np.zeros(T)
    event = np.asarray(event, float).reshape(T)
    if not np.isfinite(event).all(): raise ValueError("event must be finite")
    hue = (np.arctan2(scene[:, 1], scene[:, 0]) / (2 * np.pi)) % 1.0
    sat = np.clip(scene[:, 2], 0., 1.)
    val = np.clip(scene[:, 3], 0., 1.)
    n = len(rig.fixtures)
    params = {k: np.zeros((T, n), np.float64) for k in ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom", "gobo")}
    params["hue"][:] = hue[:, None]; params["sat"][:] = sat[:, None]; params["val"][:] = val[:, None]
    params["dimmer"][:] = val[:, None]; params["strobe"][:] = np.clip(event[:, None], 0., 1.)
    params["pan"][:] = .5; params["tilt"][:] = .5; params["zoom"][:] = np.clip(1. - scene[:, 5, None], 0., 1.)
    # Body-conditioned aim.  The same assignment is used at every fixture count
    # and has no independent temporal oscillator.
    rel = body - body[:, :1]
    for i, fixture in enumerate(rig.fixtures):
        if fixture.kind != "moving_head":
            continue
        j = (2 * i) % max(J, 1)
        k = (j + 1) % max(J, 1)
        x = np.clip(.5 + .10 * np.tanh(rel[:, j, 0] + .5 * rel[:, k, 0]), 0., 1.)
        y = np.clip(.5 + .10 * np.tanh(rel[:, j, 1] + .5 * rel[:, k, 1]), 0., 1.)
        params["pan"][:, i], params["tilt"][:, i] = x, y
    params["gobo"][:] = 0.
    return pack_universe(rig, params)

