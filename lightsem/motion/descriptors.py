"""Motion descriptors: the three target blocks of PLAN.md section 3.

The split exists because the movement-science literature says audio determines
a *shared skeleton* of a dancing body and only a *distribution* over the rest
(Burger 2013/2014, Toiviainen 2010, Solberg & Jensenius 2017). Regressing all
of it with one loss blurs; sampling all of it with one flow reproduces a dance
prior with a music watermark. So every channel carries a block label:

* **A — shared skeleton**, predicted deterministically (how much motion, at
  which metrical level, bounce, head/foot response, energy contrast).
* **B — events**, a point process (footfalls, jumps, hits).
* **C — style residual**, sampled (hands, laterality sign and phase, travel,
  spread residual, effort qualities).

The assignment here is the literature's *prediction*, not a measurement. Stage
0b (``run_stage0b.py``) re-derives it from data and is allowed to overrule any
entry; ``BLOCKS`` is the provisional map until it does.

Units: torso lengths (TL) and TL/s throughout, from
:func:`lightsem.motion.skeleton.prepare`. Angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import (
    FEET, FWD, HANDS, LANK, LAT, LHIP, LSHO, LWRI, NOSE, RANK, RHIP, RSHO,
    RWRI, UP, Skeleton, derivative, resample, smooth_ma,
)

CTRL_FPS = 40.0          # control / DMX rate, PLAN.md section 4

#: Provisional block assignment (Stage 0b may move any channel).
BLOCKS: dict[str, tuple[str, ...]] = {
    "A": (
        "qom", "com_speed", "head_speed", "foot_speed", "hip_yaw_speed",
        "vert_bounce_amp", "vert_period_s", "metric_level", "weight",
        "time_eff", "footfall_rate", "section_energy",
    ),
    "B": ("footfall", "jump", "hit"),
    "C": (
        "hand_speed", "hand_spread", "lateral_pos", "lateral_vel",
        "travel_speed", "travel_sin", "travel_cos", "spread", "symmetry",
        "flow_eff", "space_eff", "jerk", "vert_bounce",
    ),
}

CHANNEL_BLOCK: dict[str, str] = {
    ch: blk for blk, chans in BLOCKS.items() for ch in chans if blk != "B"
}


@dataclass
class MotionDescriptors:
    """Frame-aligned motion channels plus event times for one clip."""

    fps: float
    times: np.ndarray                                   # (T,) seconds
    channels: dict[str, np.ndarray]                     # each (T,)
    events: dict[str, np.ndarray]                       # event times, seconds
    event_strength: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.times[-1] + 1.0 / self.fps) if self.times.size else 0.0

    def matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        names = list(names or sorted(self.channels))
        return np.stack([self.channels[n] for n in names], axis=1), names

    def at(self, name: str, t: float | np.ndarray) -> np.ndarray:
        return np.interp(t, self.times, self.channels[name])

    def smoothed(self, name: str, window_s: float) -> np.ndarray:
        return smooth_ma(self.channels[name], self.fps, window_s)


# --------------------------------------------------------------------------
# periodicity
# --------------------------------------------------------------------------

def windowed_period(
    x: np.ndarray,
    fps: float,
    win_s: float = 4.0,
    hop_s: float = 0.25,
    lo_s: float = 0.15,
    hi_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dominant period of ``x`` in sliding windows, by autocorrelation.

    Returns ``(centre_times, period_s, strength)``. ``period_s`` is NaN where
    no peak clears the lag range, ``strength`` is the normalised
    autocorrelation height at the peak (0..1) and doubles as a confidence.

    Why autocorrelation and not an FFT peak: dance bounce is not sinusoidal,
    and the tactus-vs-half-tactus decision (PLAN.md section 2) needs the lag
    domain where octave errors are visible as competing peaks.
    """
    x = np.asarray(x, dtype=np.float64)
    n_win = max(int(round(win_s * fps)), 8)
    hop = max(int(round(hop_s * fps)), 1)
    lo, hi = max(int(round(lo_s * fps)), 1), int(round(hi_s * fps))
    starts = np.arange(0, max(len(x) - n_win, 0) + 1, hop)
    if starts.size == 0:
        starts = np.array([0])
        n_win = len(x)
    hi = min(hi, n_win - 2)
    times, periods, strengths = [], [], []
    for s in starts:
        seg = x[s:s + n_win]
        seg = seg - seg.mean()
        denom = float(seg @ seg)
        times.append((s + n_win / 2.0) / fps)
        if denom < 1e-12 or hi <= lo:
            periods.append(np.nan); strengths.append(0.0); continue
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        ac = ac / denom
        band = ac[lo:hi + 1]
        k = int(np.argmax(band))
        if band[k] <= 0.05:
            periods.append(np.nan); strengths.append(float(max(band[k], 0.0))); continue
        lag = lo + k
        # parabolic refinement around the integer-lag peak
        if 0 < k < len(band) - 1:
            a, b, c = band[k - 1], band[k], band[k + 1]
            denom2 = (a - 2 * b + c)
            if abs(denom2) > 1e-12:
                lag = lag + 0.5 * (a - c) / denom2
        periods.append(lag / fps)
        strengths.append(float(band[k]))
    return np.asarray(times), np.asarray(periods), np.asarray(strengths)


def _highpass(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Remove slow drift: x minus its moving average."""
    return np.asarray(x, dtype=np.float64) - smooth_ma(x, fps, window_s)


def _rms(x: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    return np.sqrt(np.maximum(smooth_ma(np.asarray(x) ** 2, fps, window_s), 0.0))


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------

def detect_footfalls(
    sk: Skeleton,
    down_margin: float = 0.05,
    up_margin: float = 0.12,
    refractory_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Foot-strike times and impact strengths, by contact hysteresis.

    A strike is a **down-crossing after a genuine lift**: the ankle must first
    rise above ``rest + up_margin`` torso lengths and then fall back below
    ``rest + down_margin``. The earlier local-minimum version fired 2.1 times
    per musical beat on this corpus (~4 Hz) because a planted foot's marker
    jitter produces a minimum every few frames; hysteresis is what separates a
    step from a wobble. Strength is the downward speed at contact — the
    quantity a lighting desk would map to a hit.
    """
    fps = sk.fps
    times, strengths = [], []
    for ank in (LANK, RANK):
        h = sk.pos[:, ank, UP]
        v = derivative(h, fps)
        rest = float(np.percentile(h, 15))
        up_thr, down_thr = rest + up_margin, rest + down_margin
        armed = False
        last = -1e9
        for i in range(1, len(h)):
            if h[i] > up_thr:
                armed = True
            elif armed and h[i] <= down_thr and h[i - 1] > down_thr:
                t = i / fps
                if t - last >= refractory_s:
                    times.append(t)
                    lo = max(i - int(0.05 * fps), 0)
                    strengths.append(float(max(-np.min(v[lo:i + 1]), 0.0)))
                    last = t
                armed = False
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(strengths)[order]


def detect_jumps(
    sk: Skeleton,
    clearance: float = 0.18,
    min_air_s: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Landing times and impact strengths for airborne phases.

    Airborne = both ankles above their own resting height by ``clearance``
    torso lengths. The event is emitted at the LANDING because that is the
    moment a light is expected to hit.
    """
    fps = sk.fps
    hl = sk.pos[:, LANK, UP]
    hr = sk.pos[:, RANK, UP]
    rest_l = float(np.percentile(hl, 20))
    rest_r = float(np.percentile(hr, 20))
    air = (hl > rest_l + clearance) & (hr > rest_r + clearance)
    root_v = derivative(sk.root[:, UP], fps)
    times, strengths = [], []
    if air.any():
        d = np.diff(air.astype(int))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if air[0]:
            starts = np.r_[0, starts]
        if air[-1]:
            ends = np.r_[ends, len(air) - 1]
        for s, e in zip(starts, ends):
            if (e - s) / fps < min_air_s:
                continue
            times.append(e / fps)
            lo = max(e - int(0.08 * fps), 0)
            strengths.append(float(-np.min(root_v[lo:e + 1])))
    return np.asarray(times), np.asarray(strengths)


def detect_hits(
    jerk: np.ndarray,
    fps: float,
    percentile: float = 92.0,
    refractory_s: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Accent times: peaks of whole-body jerk above a per-clip percentile."""
    thr = float(np.percentile(jerk, percentile))
    cand = np.flatnonzero(
        (jerk[1:-1] >= jerk[:-2]) & (jerk[1:-1] > jerk[2:]) & (jerk[1:-1] > thr)
    ) + 1
    times, strengths, last = [], [], -1e9
    for i in cand:
        t = i / fps
        if t - last < refractory_s:
            continue
        last = t
        times.append(t)
        strengths.append(float(jerk[i]))
    s = np.asarray(strengths)
    if s.size:
        s = s / max(float(s.max()), 1e-9)
    return np.asarray(times), s


def _rate_from_events(event_times: np.ndarray, times: np.ndarray, fps: float,
                      window_s: float = 2.0) -> np.ndarray:
    """Smoothed event rate (events/s) on the frame grid."""
    counts = np.zeros_like(times)
    if event_times.size:
        idx = np.clip(np.round(event_times * fps).astype(int), 0, len(times) - 1)
        np.add.at(counts, idx, 1.0)
    return smooth_ma(counts, fps, window_s) * fps


# --------------------------------------------------------------------------
# main extractor
# --------------------------------------------------------------------------

def extract(
    sk: Skeleton,
    beats: np.ndarray | None = None,
    tempo: float | None = None,
    fps_out: float = CTRL_FPS,
) -> MotionDescriptors:
    """Compute every descriptor channel and event stream for one clip.

    Channels are computed at the capture rate (so impacts survive) and then
    linearly resampled to ``fps_out``; event times stay in seconds and are
    never resampled.
    """
    fps = sk.fps
    pos, loc = sk.pos, sk.local
    root = sk.root

    v_loc = derivative(loc, fps)                     # (T,17,3) limb velocity
    a_loc = derivative(v_loc, fps)
    j_loc = derivative(a_loc, fps)
    sp_loc = np.linalg.norm(v_loc, axis=-1)          # (T,17)
    acc_loc = np.linalg.norm(a_loc, axis=-1)
    jerk_loc = np.linalg.norm(j_loc, axis=-1)

    centroid = pos.mean(axis=1)
    v_centroid = derivative(centroid, fps)

    ch: dict[str, np.ndarray] = {}

    # --- block A: how much, where, how fast ------------------------------
    ch["qom"] = sp_loc.mean(axis=1)
    ch["com_speed"] = np.linalg.norm(v_centroid, axis=-1)
    ch["head_speed"] = sp_loc[:, NOSE]
    ch["foot_speed"] = sp_loc[:, list(FEET)].mean(axis=1)
    ch["hand_speed"] = sp_loc[:, list(HANDS)].mean(axis=1)

    hip_vec = pos[:, RHIP] - pos[:, LHIP]
    sho_vec = pos[:, RSHO] - pos[:, LSHO]
    hip_yaw = np.unwrap(np.arctan2(hip_vec[:, FWD], hip_vec[:, LAT]))
    sho_yaw = np.unwrap(np.arctan2(sho_vec[:, FWD], sho_vec[:, LAT]))
    ch["hip_yaw_speed"] = np.abs(derivative(hip_yaw, fps))
    ch["twist"] = np.abs(((sho_yaw - hip_yaw + np.pi) % (2 * np.pi)) - np.pi)

    # --- vertical bounce: the tactus channel -----------------------------
    vert = root[:, UP]
    ch["vert_root"] = vert
    bounce = _highpass(vert, fps, 1.0)
    ch["vert_bounce"] = bounce
    ch["vert_bounce_amp"] = _rms(bounce, fps, 2.0)

    pt, per, strg = windowed_period(bounce, fps, win_s=4.0, hop_s=0.25)
    t_frames = np.arange(len(vert)) / fps
    good = np.isfinite(per)
    if good.sum() >= 2:
        period_f = np.interp(t_frames, pt[good], per[good])
        strength_f = np.interp(t_frames, pt, strg)
    else:
        period_f = np.full_like(t_frames, np.nan)
        strength_f = np.zeros_like(t_frames)
    ch["vert_period_s"] = period_f
    ch["vert_period_conf"] = strength_f

    # metrical level = log2(bounce period / beat period): 0 beat, 1 two-beat,
    # 2 bar, -1 half-beat. Needs the music's own beat grid.
    if beats is not None and np.asarray(beats).size >= 2:
        b = np.asarray(beats, dtype=np.float64)
        ibi = np.diff(b)
        beat_period_f = np.interp(t_frames, b[:-1] + ibi / 2, ibi)
    elif tempo:
        beat_period_f = np.full_like(t_frames, 60.0 / float(tempo))
    else:
        beat_period_f = np.full_like(t_frames, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ch["metric_level"] = np.log2(period_f / beat_period_f)
    ch["beat_period_s"] = beat_period_f

    # --- Laban effort proxies --------------------------------------------
    # Weight  = kinetic energy per unit mass (strong vs light)
    # Time    = acceleration magnitude (sudden vs sustained)
    # Flow    = jerk magnitude (bound/agitated vs free/smooth) -- unsigned
    # Space   = directness of the hands' path over 1 s (direct vs indirect)
    ch["weight"] = smooth_ma((sp_loc ** 2).mean(axis=1), fps, 0.5)
    ch["time_eff"] = smooth_ma(acc_loc.mean(axis=1), fps, 0.5)
    ch["jerk"] = jerk_loc.mean(axis=1)
    ch["flow_eff"] = smooth_ma(ch["jerk"], fps, 0.5)

    hand_path = 0.5 * (loc[:, LWRI] + loc[:, RWRI])
    step = np.linalg.norm(derivative(hand_path, fps), axis=-1) / fps
    win = max(int(round(1.0 * fps)), 3)
    kern = np.ones(win)
    pad = win // 2
    path_len = np.convolve(np.pad(step, pad, mode="reflect"), kern, mode="valid")[:len(step)]
    net = np.linalg.norm(
        hand_path[np.clip(np.arange(len(step)) + pad, 0, len(step) - 1)]
        - hand_path[np.clip(np.arange(len(step)) - pad, 0, len(step) - 1)], axis=-1)
    ch["space_eff"] = net / np.maximum(path_len, 1e-6)

    # --- shape / laterality / travel (block C) ---------------------------
    radial = np.linalg.norm(loc[:, :, [LAT, FWD]], axis=-1)      # (T,17)
    ch["spread"] = radial.mean(axis=1)
    ch["hand_spread"] = np.linalg.norm(loc[:, LWRI] - loc[:, RWRI], axis=-1)

    lat = root[:, LAT] - root[:, LAT].mean()
    ch["lateral_pos"] = lat
    ch["lateral_vel"] = derivative(lat, fps)
    tv = derivative(root[:, [LAT, FWD]], fps)
    speed_h = np.linalg.norm(tv, axis=-1)
    ch["travel_speed"] = speed_h
    ang = np.arctan2(tv[:, 1], tv[:, 0])
    ch["travel_sin"] = np.sin(ang) * np.tanh(speed_h)
    ch["travel_cos"] = np.cos(ang) * np.tanh(speed_h)

    left = sp_loc[:, [LSHO, LWRI, LHIP, LANK]].mean(axis=1)
    right = sp_loc[:, [RSHO, RWRI, RHIP, RANK]].mean(axis=1)
    ch["symmetry"] = 1.0 - np.abs(left - right) / np.maximum(left + right, 1e-6)
    ch["laterality"] = (right - left) / np.maximum(right + left, 1e-6)

    # --- events -----------------------------------------------------------
    ff_t, ff_s = detect_footfalls(sk)
    jp_t, jp_s = detect_jumps(sk)
    ht_t, ht_s = detect_hits(ch["jerk"], fps)
    ch["footfall_rate"] = _rate_from_events(ff_t, t_frames, fps, 2.0)
    ch["section_energy"] = smooth_ma(ch["qom"], fps, 8.0)

    # --- resample channels to the control rate ---------------------------
    n_out = int(round(len(t_frames) / fps * fps_out))
    times_out = np.arange(n_out) / fps_out
    out: dict[str, np.ndarray] = {}
    for k, v in ch.items():
        v = np.asarray(v, dtype=np.float64)
        if not np.isfinite(v).all():
            finite = np.isfinite(v)
            if finite.any():
                v = np.interp(t_frames, t_frames[finite], v[finite])
            else:
                v = np.zeros_like(v)
        out[k] = resample(v, fps, fps_out, n_out=n_out)

    return MotionDescriptors(
        fps=fps_out,
        times=times_out,
        channels=out,
        events={"footfall": ff_t, "jump": jp_t, "hit": ht_t},
        event_strength={"footfall": ff_s, "jump": jp_s, "hit": ht_s},
        meta={
            "name": sk.name,
            "src_fps": fps,
            "torso_cm": sk.torso_cm,
            "duration_s": len(t_frames) / fps,
            "n_footfalls": int(ff_t.size),
            "n_jumps": int(jp_t.size),
            "n_hits": int(ht_t.size),
        },
    )
