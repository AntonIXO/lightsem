"""The baseline every claim in this project is priced against.

Thirty lines of rules with no dance, no model and no structure analysis:
onset -> strobe, RMS -> dimmer, spectral jump -> colour, bar -> pan sweep.
This is what a competent engineer builds in an afternoon, and it is the thing
LightSem has to beat before any of the rest matters (PLAN.md Stage 0c ii).
It is deliberately not tuned per track.
"""

from __future__ import annotations

import numpy as np

from ..audio.online import AudioBus
from ..dmx.encode import pack_universe, slew_limit
from ..dmx.universe import DmxStream
from .rig import canonical_rig, role_index, stage_x

PALETTE = (0.60, 0.92, 0.08, 0.33, 0.75, 0.50)


def _n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def render_beat_trigger(bus: AudioBus, rig=None, fps: float = 40.0,
                        delay_s: float = 0.0) -> DmxStream:
    """Audio -> DMX with the dumbest rules that still look like a light show.

    ``delay_s`` shifts the whole output later, which is how Stage 0c (i)
    measures what a late light costs.
    """
    rig = rig or canonical_rig()
    roles, xs = role_index(rig), stage_x(rig)
    T, nf = len(bus.times), len(rig.fixtures)
    lvl = _n(bus.channels["rms"])
    onset = _n(bus.channels["onset"])
    accent = bus.channels["beat_accent"]
    bright = _n(bus.channels["centroid"])
    bar = bus.channels["bar_phase"]

    hue = np.zeros(T)
    jump = np.abs(np.diff(bright, prepend=bright[:1])) > 0.18
    k = 0
    for i in range(T):
        if jump[i] and (i == 0 or not jump[i - 1]):
            k += 1
        hue[i] = PALETTE[k % len(PALETTE)]

    P = {p: np.zeros((T, nf)) for p in
         ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom", "gobo")}
    P["val"][:] = 1.0
    P["hue"][:] = hue[:, None]
    P["sat"][:] = 0.85
    P["zoom"][:] = 0.5
    P["tilt"][:] = 0.5
    sweep = 0.5 + 0.35 * np.sin(2 * np.pi * bar)
    P["pan"][:] = sweep[:, None]

    w = roles["big_slow"]
    P["dimmer"][:, w] = np.clip(0.15 + 0.7 * lvl, 0, 1)[:, None]
    b = roles["beam"]
    chase = 0.5 + 0.5 * np.cos(2 * np.pi * (bar[:, None] - xs[None, b] * 0.25))
    P["dimmer"][:, b] = np.clip((0.25 + 0.75 * accent)[:, None] * chase * (0.35 + 0.65 * lvl[:, None]), 0, 1)
    e = roles["event"]
    P["dimmer"][:, e] = np.clip(onset ** 3, 0, 1)[:, None]
    P["sat"][:, e] = 0.0
    f = roles["fast"]
    P["strobe"][:, f] = np.clip(bright, 0, 1)[:, None]
    P["dimmer"][:, f] = np.clip(onset ** 2 * lvl, 0, 1)[:, None]
    P["sat"][:, f] = 0.0
    k2 = roles["back"]
    P["dimmer"][:, k2] = np.clip(0.2 + 0.6 * lvl, 0, 1)[:, None]

    P["pan"] = slew_limit(P["pan"], fps, 1.1)
    P["hue"] = slew_limit(P["hue"], fps, 0.45, wrap=True)
    P["dimmer"][:, w] = slew_limit(P["dimmer"][:, w], fps, 4.0)

    n = int(round(delay_s * fps))
    if n > 0:
        for key in P:
            P[key] = np.concatenate([np.repeat(P[key][:1], n, axis=0), P[key][:-n]], axis=0)

    frames = pack_universe(rig, P)
    return DmxStream(frames=frames, fps=fps, times=bus.times.copy(),
                     rig_summary=rig.summary(),
                     cues=[{"type": "baseline", "time_s": 0.0, "event": "beat_trigger",
                            "delay_s": delay_s}])
