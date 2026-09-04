"""Per-clip light features, computed on DELTAS rather than levels.

PLAN.md section 7 is blunt about why: a distance computed on light *levels*
rewards a system that finds one pretty look and holds it for four minutes,
which is the exact failure mode a "learned" lighting model falls into. Every
feature here is a rate, a change, a spread or a spectrum. The two level
features that are computed (mean and 95th-percentile brightness) are returned
separately and are used only to brightness-match A/B pairs, never inside
FID_light.
"""

from __future__ import annotations

import numpy as np

from ..dmx.decode import brightness, decode_stream, emitted_light
from ..dmx.fixtures import Rig

FEATURE_NAMES = (
    "event_rate", "delta_mean", "delta_sd", "delta_p90",
    "hue_change_rate", "hue_change_mean", "sat_sd",
    "pan_vel_mean", "tilt_vel_mean", "zoom_vel_mean",
    "strobe_frac", "strobe_mean",
    "band_0_1", "band_1_2", "band_2_4", "band_4_8", "band_8_20",
    "ac_peak", "ac_lag_s", "iei_cv",
    "spatial_sd", "lateral_centroid_sd", "contrast", "on_frac",
)


#: Summed positive change in emitted intensity that counts as a visible accent.
#: ABSOLUTE, not a per-clip percentile: a percentile threshold forces every
#: stream to emit the same number of events per second, which silently turns
#: "how often does it flash" into a constant and makes the event rate feature
#: and the timing metric measure nothing. Units: fixtures at full output, so
#: 0.30 is roughly one third of one fixture snapping on within 25 ms.
VISIBLE_DELTA = 0.30


def light_onsets(frames: np.ndarray, rig: Rig, fps: float,
                 abs_thr: float = VISIBLE_DELTA, refractory_s: float = 0.10,
                 min_rel: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Times and sizes of visible light events (a "light beat").

    The signal is the summed positive change in EMITTED intensity across the
    rig, so a cross-fade that keeps total output constant is correctly not an
    event, and a blackout-to-blinder is a big one.
    """
    dec = decode_stream(rig, frames)
    emit = emitted_light(dec)
    d = np.maximum(np.diff(emit, axis=0), 0.0).sum(axis=1)
    if d.size < 3 or not np.any(d > 0):
        return np.array([]), np.array([])
    thr = max(abs_thr, min_rel * float(d.max()))
    times, sizes, last = [], [], -1e9
    for i in range(1, len(d) - 1):
        if d[i] > thr and d[i] >= d[i - 1] and d[i] > d[i + 1]:
            t = (i + 1) / fps
            if t - last >= refractory_s:
                times.append(t)
                sizes.append(float(d[i]))
                last = t
    return np.asarray(times), np.asarray(sizes)


def _band_power(x: np.ndarray, fps: float, bands) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.size < 8 or np.allclose(x, 0):
        return [0.0] * len(bands)
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    p = np.abs(np.fft.rfft(x)) ** 2
    tot = float(p.sum()) + 1e-12
    return [float(p[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in bands]


def delta_features(frames: np.ndarray, rig: Rig, fps: float) -> np.ndarray:
    """One clip -> the FID/diversity feature vector (len(FEATURE_NAMES),)."""
    dec = decode_stream(rig, frames)
    emit = emitted_light(dec)
    bright = brightness(dec)
    T = emit.shape[0]
    d_all = np.maximum(np.diff(emit, axis=0), 0.0).sum(axis=1) if T > 1 else np.zeros(1)

    ev_t, ev_s = light_onsets(frames, rig, fps)
    dur = T / fps
    iei = np.diff(ev_t) if ev_t.size > 2 else np.array([])

    hue = dec["hue"]
    dh = np.abs(((np.diff(hue, axis=0) + 0.5) % 1.0) - 0.5) if T > 1 else np.zeros((1, hue.shape[1]))
    hue_change = dh.mean(axis=1)

    pan_v = np.abs(np.diff(dec["pan"], axis=0)).mean() * fps if T > 1 else 0.0
    tilt_v = np.abs(np.diff(dec["tilt"], axis=0)).mean() * fps if T > 1 else 0.0
    zoom_v = np.abs(np.diff(dec["zoom"], axis=0)).mean() * fps if T > 1 else 0.0

    bands = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 20))
    bp = _band_power(bright, fps, bands)

    x = bright - bright.mean()
    if x.size > 16 and float(x @ x) > 1e-12:
        ac = np.correlate(x, x, "full")[x.size - 1:] / float(x @ x)
        lo, hi = int(0.15 * fps), min(int(2.5 * fps), ac.size - 1)
        if hi > lo:
            k = int(np.argmax(ac[lo:hi])) + lo
            ac_peak, ac_lag = float(ac[k]), k / fps
        else:
            ac_peak, ac_lag = 0.0, 0.0
    else:
        ac_peak, ac_lag = 0.0, 0.0

    fx_mean = emit.mean(axis=0)
    xs = np.array([f.position[0] for f in rig.fixtures])
    xs = xs / max(np.abs(xs).max(), 1e-6)
    w = emit / np.maximum(emit.sum(axis=1, keepdims=True), 1e-9)
    lat_centroid = (w * xs[None, :]).sum(axis=1)

    vals = [
        ev_t.size / max(dur, 1e-6),
        float(d_all.mean()), float(d_all.std()), float(np.percentile(d_all, 90)),
        float((hue_change > 1e-3).mean() * fps), float(hue_change.mean()),
        float(dec["sat"].std()),
        pan_v, tilt_v, zoom_v,
        float((dec["strobe"] > 0.05).mean()), float(dec["strobe"].mean()),
        *bp,
        ac_peak, ac_lag,
        float(iei.std() / iei.mean()) if iei.size > 2 and iei.mean() > 0 else 0.0,
        float(fx_mean.std()), float(lat_centroid.std()),
        float(bright.std() / max(bright.mean(), 1e-6)),
        float((bright > 0.05).mean()),
    ]
    return np.asarray(vals, dtype=np.float64)


def block_shuffle(frames: np.ndarray, fps: float, block_s: float = 1.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Shuffle whole blocks of a light stream in time.

    The null FID_light is measured against, replacing the circular shift named
    in PLAN.md: almost every delta feature is shift-invariant by construction,
    so a circular shift moves FID by nothing and would "pass" a sensitivity
    gate that tests nothing. Block shuffling keeps the marginal distribution of
    levels and destroys the temporal structure, which is exactly the thing the
    features are supposed to see.
    """
    rng = rng or np.random.default_rng(0)
    n = int(round(block_s * fps))
    if n < 2 or frames.shape[0] < 2 * n:
        return frames.copy()
    k = frames.shape[0] // n
    blocks = frames[: k * n].reshape(k, n, frames.shape[1])
    out = blocks[rng.permutation(k)].reshape(k * n, frames.shape[1])
    return np.concatenate([out, frames[k * n:]], axis=0)


def level_features(frames: np.ndarray, rig: Rig) -> dict:
    """Brightness statistics — for A/B matching only, never for FID."""
    b = brightness(decode_stream(rig, frames))
    return {"mean": float(b.mean()), "p95": float(np.percentile(b, 95)),
            "max": float(b.max())}
