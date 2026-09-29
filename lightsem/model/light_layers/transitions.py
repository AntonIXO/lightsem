"""Causal fades between supplied channel-RGB targets; no new cue decisions.

Values are DMX channel proportions, not calibrated radiance or sRGB. Fades
start at the supplied change time, reach the target after duration_s, and
retarget from the current interpolated color when interrupted. They never
anticipate a cue. Call on the full record before cropping for review.
"""
from __future__ import annotations

import math
import numpy as np

from .overlay import _times, _track


def fade_rgb_targets(targets, times, *, duration_s=0.3, active=None):
    """Smoothstep channel-RGB fades, preserving inactive frames and inputs.

    A zero duration is an exact copy. Repeated identical targets do not restart
    a fade. The first frame initializes color; it does not invent a fade from
    black. Disabling then enabling starts from the last emitted inactive color.
    """
    if (isinstance(duration_s, (bool, np.bool_))
            or not isinstance(duration_s, (int, float, np.integer, np.floating))
            or not math.isfinite(duration_s) or duration_s < 0):
        raise ValueError("duration_s must be finite and nonnegative")
    times = _times(times)
    targets = _track(targets, (len(times), 3), "RGB targets", upper=1)
    if active is None:
        active = np.ones(len(times), dtype=bool)
    else:
        active = np.asarray(active)
        if active.dtype != bool or active.shape != (len(times),):
            raise ValueError("active must be a matching boolean timeline")
    out = targets.copy()
    if not len(times) or duration_s == 0:
        return out
    origin = targets[0].copy()
    target = origin.copy()
    began = times[0] - duration_s
    for i in range(1, len(times)):
        if not active[i]:
            origin = targets[i].copy()
            target = origin.copy()
            began = times[i] - duration_s
            continue
        u = float(np.clip((times[i] - began) / duration_s, 0., 1.))
        color = origin + (target - origin) * (u * u * (3. - 2. * u))
        if not np.array_equal(targets[i], target):
            origin = color
            target = targets[i].copy()
            began = times[i]
        out[i] = color
    return np.clip(out, 0., 1.)
