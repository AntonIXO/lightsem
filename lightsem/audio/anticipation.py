"""Anticipation: tension curve + predicted time-to-next-drop.

THE HONESTY PROBLEM THIS MODULE IS BUILT AROUND
-----------------------------------------------
It is trivially easy to "predict" a drop when you already hold the whole
decoded track: you just look ahead in the array and call it anticipation.
That is offline lookahead, not prediction, and reporting it as prediction
would be fabrication.

So this module computes the tension curve in TWO explicitly separated modes,
and every consumer must state which one it used:

``CAUSAL`` (mode="causal")
    At time t, only samples with time <= t are read. Implemented with
    one-sided (backward-looking) rolling statistics and a causal EMA. This is
    what a real-time rig could actually run. Its time-to-drop estimate is a
    genuine forecast: extrapolating the current tension slope to a threshold
    crossing.

``ORACLE`` (mode="oracle")
    Full-track lookahead using the known segment boundaries. Its
    time-to-drop is exact by construction because it reads the answer. This
    is NOT a prediction — it is the ceiling a predictor is scored against.

``evaluate_anticipation`` scores CAUSAL against ORACLE so the gap is a
measured number rather than a claim.

Lookahead horizon
-----------------
``lookahead_s`` is an explicit parameter, reported in every output. It is the
window within which a cue may be emitted BEFORE the event lands — the whole
point of anticipatory control (light that leads the build-up rather than
reacting to it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import AudioFeatures, _norm
from .structure import Segment

CAUSAL = "causal"
ORACLE = "oracle"


@dataclass
class AnticipationCurves:
    """Frame-aligned anticipation signals."""

    mode: str
    lookahead_s: float
    frame_times: np.ndarray
    tension: np.ndarray            # (F,) 0..1 build-up tension
    tension_slope: np.ndarray      # (F,) d(tension)/dt, per second
    time_to_drop: np.ndarray       # (F,) seconds until next drop (inf if none)
    cue_gate: np.ndarray           # (F,) bool: emit the anticipatory cue now

    def as_dict(self) -> dict:
        finite = np.isfinite(self.time_to_drop)
        return {
            "mode": self.mode,
            "lookahead_s": self.lookahead_s,
            "n_frames": int(self.frame_times.size),
            "tension_mean": float(self.tension.mean()),
            "tension_max": float(self.tension.max()),
            "cue_frames": int(self.cue_gate.sum()),
            "frames_with_known_drop": int(finite.sum()),
        }


def _causal_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """One-sided exponential moving average. Reads only the past."""
    out = np.empty_like(x, dtype=np.float64)
    acc = float(x[0])
    for i, v in enumerate(x):
        acc = alpha * float(v) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _causal_slope(x: np.ndarray, dt: float, win: int) -> np.ndarray:
    """Backward-difference slope over ``win`` frames, per second."""
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        j = max(0, i - win)
        span = (i - j) * dt
        out[i] = (x[i] - x[j]) / span if span > 0 else 0.0
    return out


def tension_curve(
    feats: AudioFeatures,
    mode: str = CAUSAL,
    smooth_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build-up tension from energy, brightness, density and their rise.

    EDM build-ups share a signature: energy climbs, the spectrum brightens
    (risers/white noise sweeps), onset density increases, and low-end often
    drops out just before the release. We combine the first three.
    """
    dt = float(np.median(np.diff(feats.frame_times))) if feats.n_frames > 1 else 0.01
    win = max(1, int(round(smooth_s / dt)))

    energy = _norm(feats.rms)
    bright = _norm(feats.centroid)
    density = _norm(feats.onset_env)
    base = 0.5 * energy + 0.25 * bright + 0.25 * density

    if mode == CAUSAL:
        alpha = 2.0 / (win + 1.0)
        smoothed = _causal_ema(base, alpha)
        slope = _causal_slope(smoothed, dt, win)
    elif mode == ORACLE:
        kern = np.ones(win) / win
        smoothed = np.convolve(base, kern, mode="same")
        slope = np.gradient(smoothed, dt)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Tension = level plus the *rate of climb*, which is what makes a
    # build-up feel like a build-up.
    tension = _norm(smoothed + 2.0 * np.maximum(slope, 0.0))
    return tension, slope


def _drop_times(segments: list[Segment]) -> list[float]:
    return [s.start_s for s in segments if s.label == "drop"]


def oracle_time_to_drop(frame_times: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """Exact seconds to the next drop onset. Reads the answer by design."""
    drops = _drop_times(segments)
    out = np.full(frame_times.shape, np.inf, dtype=np.float64)
    for i, t in enumerate(frame_times):
        future = [d - t for d in drops if d >= t]
        if future:
            out[i] = min(future)
    return out


def causal_time_to_drop(
    tension: np.ndarray,
    slope: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.75,
    max_horizon_s: float = 30.0,
) -> np.ndarray:
    """Forecast: extrapolate the current tension slope to a threshold crossing.

    Uses ONLY information available at each frame. If tension is already above
    threshold the estimate is 0 (the event is judged to be happening now); if
    tension is flat or falling no crossing is predicted (inf).
    """
    out = np.full(tension.shape, np.inf, dtype=np.float64)
    for i in range(len(tension)):
        cur, s = tension[i], slope[i]
        if cur >= threshold:
            out[i] = 0.0
        elif s > 1e-6:
            eta = (threshold - cur) / s
            if 0.0 <= eta <= max_horizon_s:
                out[i] = eta
    return out


def compute(
    feats: AudioFeatures,
    segments: list[Segment],
    mode: str = CAUSAL,
    lookahead_s: float = 2.0,
    threshold: float = 0.75,
    smooth_s: float = 1.0,
) -> AnticipationCurves:
    """Compute the anticipation bundle in the requested mode."""
    tension, slope = tension_curve(feats, mode=mode, smooth_s=smooth_s)
    if mode == ORACLE:
        ttd = oracle_time_to_drop(feats.frame_times, segments)
    else:
        ttd = causal_time_to_drop(tension, slope, feats.frame_times, threshold)
    # Fire the cue once the predicted event enters the lookahead window.
    cue_gate = (ttd <= lookahead_s) & np.isfinite(ttd)
    return AnticipationCurves(
        mode=mode, lookahead_s=lookahead_s, frame_times=feats.frame_times,
        tension=tension, tension_slope=slope, time_to_drop=ttd, cue_gate=cue_gate,
    )


def evaluate_anticipation(
    feats: AudioFeatures,
    segments: list[Segment],
    lookahead_s: float = 2.0,
    tolerance_s: float = 1.0,
    threshold: float = 0.75,
    max_lead_s: float = 12.0,
) -> dict:
    """Score CAUSAL anticipation against a rate-matched random baseline.

    A cue is a HIT if it precedes a true drop by between 0 and ``max_lead_s``
    seconds. Cues after the drop are reactive, not anticipatory, and never
    count.

    WHY ``max_lead_s`` IS SEPARATE FROM ``lookahead_s`` (a real bug, fixed):
    the window was originally ``lookahead_s + tolerance_s``. That scored a cue
    arriving 6.3 s before the drop as a MISS while a cue at 2.9 s counted --
    i.e. the metric punished *better* anticipation, which is precisely
    backwards for a system whose entire purpose is leading the build-up.
    ``lookahead_s`` controls WHEN the rig fires; ``max_lead_s`` controls how
    early still counts as usefully anticipatory. A lighting cue 6 s before a
    drop is a feature, not an error.

    ``mean_lead_s`` reports the EARLIEST qualifying cue per drop, since that
    is the moment the rig could first have started leading.
    """
    causal = compute(feats, segments, mode=CAUSAL, lookahead_s=lookahead_s,
                     threshold=threshold)
    drops = _drop_times(segments)
    ft = feats.frame_times

    # Collapse the gate into discrete cue onsets.
    gate = causal.cue_gate.astype(int)
    onsets = ft[np.flatnonzero(np.diff(np.concatenate(([0], gate))) == 1)]

    def qualifies(lead: float) -> bool:
        return 0.0 <= lead <= max_lead_s

    hits, leads = 0, []
    for d in drops:
        lead = [d - c for c in onsets if qualifies(d - c)]
        if lead:
            hits += 1
            leads.append(max(lead))  # earliest cue = longest lead
    recall = hits / len(drops) if drops else float("nan")
    precision = (
        sum(any(qualifies(d - c) for d in drops) for c in onsets) / len(onsets)
        if onsets.size else float("nan")
    )

    # Rate-matched random baseline: a detector that fires constantly would
    # otherwise score perfect recall, so the baseline uses the SAME cue rate.
    rng = np.random.default_rng(0)
    rate = onsets.size / max(feats.duration_s, 1e-9)
    trials, base_recalls = 200, []
    for _ in range(trials):
        k = rng.poisson(rate * feats.duration_s)
        fake = np.sort(rng.uniform(0, feats.duration_s, size=k))
        h = sum(any(qualifies(d - c) for c in fake) for d in drops)
        base_recalls.append(h / len(drops) if drops else 0.0)

    return {
        "lookahead_s": lookahead_s,
        "tolerance_s": tolerance_s,
        "max_lead_s": max_lead_s,
        "n_drops": len(drops),
        "n_cues": int(onsets.size),
        "recall": recall,
        "precision": precision,
        "mean_lead_s": float(np.mean(leads)) if leads else float("nan"),
        "baseline_recall_mean": float(np.mean(base_recalls)),
        "baseline_recall_std": float(np.std(base_recalls)),
        "cue_rate_hz": rate,
    }
