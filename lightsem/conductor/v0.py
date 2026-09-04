"""Conductor v0 — the fixed body-to-light mapping of PLAN.md section 3.1.

This is the *oracle* path: real dance in, DMX out, no model and **no audio**.
That restriction is the whole point. Stage 0c asks whether light derived from a
dancing body beats a 30-line beat trigger; if the conductor were allowed to
peek at the beat grid, a win would prove nothing. The only audio-derived
quantity that reaches this class is the metrical level already baked into the
descriptors, and ``motion_only`` records the promise.

The mapping follows the body rather than the spectrum:

======================  =========================================
body channel            light
======================  =========================================
quantity of motion      master intensity, haze, saturation
vertical bounce         dimmer + tilt pulse on the beam layer
lateral position        which side of the rig is lit, pan sweep
head / torso speed      big slow fixtures (washes)
hand speed              small fast fixtures (strobe rate)
footfalls, jumps        blinder hits, tilt snaps
spread                  zoom / beam width
Laban Time              crossfade rate (slew limits)
section energy          back wall, palette changes
======================  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dmx.encode import pack_universe, slew_limit
from ..dmx.fixtures import Rig
from ..dmx.universe import DmxStream
from ..motion.descriptors import MotionDescriptors
from .rig import canonical_rig, role_index, stage_x
from .scalers import Squash, load_scalers

#: hue palette walked at motion-novelty boundaries (hue, saturation)
PALETTE = ((0.60, 0.85), (0.92, 0.75), (0.08, 0.90), (0.33, 0.70),
           (0.75, 0.80), (0.50, 0.65))


@dataclass
class ConductorConfig:
    fps: float = 40.0
    mirror_to_audience: bool = True     # dancer's right reads as audience left
    base_level: float = 0.12            # never fully black
    bounce_depth: float = 0.68
    spatial_sigma: float = 0.42
    spatial_floor: float = 0.16
    contrast_gain: float = 0.85     # tanh steepness; < 1.5 = more dynamic range
    blinder_decay_s: float = 0.22
    strobe_decay_s: float = 0.12
    novelty_window_s: float = 4.0
    slew: dict = field(default_factory=lambda: {
        "wash_dimmer": 4.0, "head_dimmer": 14.0, "pan": 1.1, "tilt": 1.1,
        "zoom": 1.5, "hue": 0.45, "sat": 1.0,
    })


def impulse_train(times: np.ndarray, strengths: np.ndarray, n: int, fps: float,
                  decay_s: float, attack_s: float = 0.02) -> np.ndarray:
    """Event times -> a (n,) envelope of attack/decay flashes."""
    out = np.zeros(n)
    if times is None or len(times) == 0:
        return out
    s = np.ones(len(times)) if strengths is None or len(strengths) != len(times) \
        else np.asarray(strengths, dtype=np.float64)
    if s.max() > 0:
        s = np.clip(s / np.percentile(s, 90), 0.0, 1.5)
    decay = np.exp(-np.arange(0, int(decay_s * 5 * fps) + 1) / (decay_s * fps))
    atk = max(int(attack_s * fps), 1)
    for t, a in zip(np.asarray(times), s):
        i = int(round(t * fps))
        if not (0 <= i < n):
            continue
        j = min(i + len(decay), n)
        out[i:j] = np.maximum(out[i:j], a * decay[: j - i])
        lo = max(i - atk, 0)
        out[lo:i] = np.maximum(out[lo:i], a * np.linspace(0, 1, i - lo, endpoint=False))
    return out


def novelty_segments(desc: MotionDescriptors, window_s: float = 4.0,
                     min_gap_s: float = 3.0) -> np.ndarray:
    """Frame indices where the body's state changes enough to warrant a look.

    Self-similarity novelty on the block-A channels: the dance's own phrase
    boundaries, which is where a human operator changes the look.
    """
    names = ["qom", "head_speed", "foot_speed", "vert_bounce_amp", "spread",
             "hand_speed", "weight"]
    X = np.stack([desc.channels[n] for n in names if n in desc.channels], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    fps = desc.fps
    w = max(int(window_s * fps), 4)
    n = X.shape[0]
    nov = np.zeros(n)
    for i in range(w, n - w):
        a, b = X[i - w:i].mean(0), X[i:i + w].mean(0)
        nov[i] = np.linalg.norm(a - b)
    if nov.max() > 0:
        nov /= nov.max()
    thr = max(float(np.percentile(nov, 88)), 0.25)
    peaks, last = [], -1e9
    for i in range(1, n - 1):
        if nov[i] > thr and nov[i] >= nov[i - 1] and nov[i] > nov[i + 1]:
            if (i - last) / fps >= min_gap_s:
                peaks.append(i)
                last = i
    return np.array([0] + peaks, dtype=int)


class ConductorV0:
    """Deterministic motion -> DMX mapping on the canonical rig."""

    motion_only = True

    def __init__(self, rig: Rig | None = None, cfg: ConductorConfig | None = None,
                 scalers: dict | None = None):
        self.rig = rig or canonical_rig()
        self.cfg = cfg or ConductorConfig()
        self.sq = Squash(scalers if scalers is not None else load_scalers())
        self.roles = role_index(self.rig)
        self.x = stage_x(self.rig)

    # -- the mapping -------------------------------------------------------
    def params(self, desc: MotionDescriptors) -> dict[str, np.ndarray]:
        cfg, sq = self.cfg, self.sq
        fps = cfg.fps
        if abs(desc.fps - fps) > 1e-6:
            raise ValueError(f"descriptors are at {desc.fps} fps, conductor at {fps}")
        c = desc.channels
        T = len(desc.times)
        nf = len(self.rig.fixtures)

        g = cfg.contrast_gain
        qom = sq("qom", c["qom"], gain=g)
        weight = sq("weight", c["weight"], gain=g)
        head = sq("head_speed", c["head_speed"], gain=g)
        hand = sq("hand_speed", c["hand_speed"], gain=g)
        section = sq("section_energy", c["section_energy"], gain=g)
        spread = sq("spread", c["spread"])
        time_eff = sq("time_eff", c["time_eff"])
        bounce = sq.signed("vert_bounce", c["vert_bounce"], gain=1.2)
        depth = cfg.bounce_depth * sq("vert_bounce_amp", c["vert_bounce_amp"])

        lateral = sq.signed("lateral_pos", c["lateral_pos"], gain=1.2)
        if cfg.mirror_to_audience:
            lateral = -lateral

        master = np.clip(cfg.base_level + 0.88 * (0.6 * qom + 0.4 * weight), 0, 1)

        # events
        ev_foot = impulse_train(desc.events.get("footfall"),
                                desc.event_strength.get("footfall"), T, fps,
                                cfg.blinder_decay_s)
        ev_jump = impulse_train(desc.events.get("jump"),
                                desc.event_strength.get("jump"), T, fps,
                                cfg.blinder_decay_s * 1.6)
        ev_hit = impulse_train(desc.events.get("hit"),
                               desc.event_strength.get("hit"), T, fps,
                               cfg.strobe_decay_s)

        # palette: walk one step per motion-novelty boundary
        bounds = novelty_segments(desc, cfg.novelty_window_s)
        hue = np.zeros(T)
        sat = np.zeros(T)
        for k, start in enumerate(bounds):
            end = bounds[k + 1] if k + 1 < len(bounds) else T
            h, s = PALETTE[k % len(PALETTE)]
            hue[start:end] = h
            sat[start:end] = s
        # Keep the palette saturated. Scaling saturation down by motion made
        # every dance-derived render read as grey haze next to the baseline.
        sat = np.clip(sat * (0.86 + 0.20 * qom), 0.0, 1.0)

        P = {k: np.zeros((T, nf)) for k in
             ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom", "gobo")}
        P["val"][:] = 1.0
        P["hue"][:] = hue[:, None]
        P["sat"][:] = sat[:, None]
        P["pan"][:] = 0.5
        P["tilt"][:] = 0.5
        P["zoom"][:] = 0.5

        # -- washes: big and slow, driven by head/torso and section energy --
        w = self.roles["big_slow"]
        if w.size:
            lvl = np.clip(0.06 + 0.88 * (0.55 * head + 0.45 * section) * master, 0, 1)
            side = self._spatial(lateral, w)
            P["dimmer"][:, w] = lvl[:, None] * (0.55 + 0.45 * side)
            P["zoom"][:, w] = np.clip(0.35 + 0.6 * spread, 0, 1)[:, None]
            P["hue"][:, w] = ((hue + 0.06) % 1.0)[:, None]

        # -- moving heads: the beam layer, bouncing and following the side ---
        b = self.roles["beam"]
        if b.size:
            pulse = 0.5 - 0.5 * bounce            # body low  -> light up
            lvl = master * (1.0 - depth + depth * pulse)
            side = self._spatial(lateral, b)
            P["dimmer"][:, b] = np.clip(lvl[:, None] * side, 0, 1)
            P["pan"][:, b] = np.clip(0.5 + 0.34 * lateral, 0, 1)[:, None]
            P["tilt"][:, b] = np.clip(0.5 + 0.11 * bounce, 0, 1)[:, None]
            P["zoom"][:, b] = np.clip(0.15 + 0.7 * spread, 0, 1)[:, None]
            P["strobe"][:, b] = 0.0

        # -- blinders: the event layer ---------------------------------------
        e = self.roles["event"]
        if e.size:
            hit = np.clip(0.85 * ev_jump + 0.5 * ev_foot, 0, 1)
            P["dimmer"][:, e] = hit[:, None]
            P["sat"][:, e] = 0.0

        # -- strobe: fast, follows the hands ---------------------------------
        f = self.roles["fast"]
        if f.size:
            P["strobe"][:, f] = np.clip(hand, 0, 1)[:, None]
            P["dimmer"][:, f] = np.clip(ev_hit * (0.35 + 0.65 * hand), 0, 1)[:, None]
            P["sat"][:, f] = 0.0

        # -- projector: the back wall ----------------------------------------
        k = self.roles["back"]
        if k.size:
            P["dimmer"][:, k] = np.clip(0.18 + 0.7 * section * master, 0, 1)[:, None]
            P["zoom"][:, k] = np.clip(0.3 + 0.5 * spread, 0, 1)[:, None]
            P["pan"][:, k] = np.clip(0.5 + 0.15 * lateral, 0, 1)[:, None]

        # -- Laban Time -> crossfade speed, then physical slew limits --------
        speed = 0.55 + 0.9 * time_eff.mean()
        sl = cfg.slew
        wash_idx = np.r_[w, k] if k.size else w
        dim = P["dimmer"].copy()
        if wash_idx.size:
            dim[:, wash_idx] = slew_limit(dim[:, wash_idx], fps, sl["wash_dimmer"] * speed)
        if b.size:
            dim[:, b] = slew_limit(dim[:, b], fps, sl["head_dimmer"] * speed)
        P["dimmer"] = dim
        P["pan"] = slew_limit(P["pan"], fps, sl["pan"])
        P["tilt"] = slew_limit(P["tilt"], fps, sl["tilt"])
        P["zoom"] = slew_limit(P["zoom"], fps, sl["zoom"])
        P["hue"] = slew_limit(P["hue"], fps, sl["hue"] * speed, wrap=True)
        P["sat"] = slew_limit(P["sat"], fps, sl["sat"])
        return P

    def _spatial(self, lateral: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """(T, len(idx)) weight: fixtures near the dancer's side get the light."""
        xs = self.x[idx][None, :]
        d = xs - lateral[:, None]
        w = np.exp(-(d ** 2) / (2 * self.cfg.spatial_sigma ** 2))
        return self.cfg.spatial_floor + (1 - self.cfg.spatial_floor) * w

    # -- output ------------------------------------------------------------
    def render(self, desc: MotionDescriptors) -> DmxStream:
        P = self.params(desc)
        frames = pack_universe(self.rig, P)
        cues = []
        for name in ("footfall", "jump", "hit"):
            for t in np.asarray(desc.events.get(name, []))[:400]:
                cues.append({"type": "motion_event", "time_s": round(float(t), 3),
                             "event": name})
        for i in novelty_segments(desc, self.cfg.novelty_window_s):
            cues.append({"type": "look_change", "time_s": round(i / self.cfg.fps, 3),
                         "event": "motion_novelty"})
        cues.sort(key=lambda c: c["time_s"])
        return DmxStream(frames=frames, fps=self.cfg.fps, times=desc.times.copy(),
                         rig_summary=self.rig.summary(), cues=cues)
