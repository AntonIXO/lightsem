"""Beat alignment for light, priced against the two nulls that matter.

Light-BAS on its own is a broken metric and PLAN.md section 7 says so: a
metronome flashing on every beat scores near the ceiling while being the
single least musical lighting cue there is. So nothing here returns a bare
number. :func:`bas_report` always carries

* ``metronome`` — the UPPER reference (a perfect, dumb, on-beat trigger);
* ``null_mean``/``null_sd`` — the LOWER reference from circular time shifts of
  the same event train, which preserves its rate and burstiness and destroys
  only its phase relationship to the music.

Report ``ratio_metronome`` (how much of the dumb ceiling is reached) and ``z``
(how far above chance), never ``bas`` alone.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIGMA = 0.1          # seconds; ~1 beat/8 at 130 BPM


def beat_alignment(events: np.ndarray, beats: np.ndarray,
                   sigma: float = DEFAULT_SIGMA) -> float:
    """Mean Gaussian closeness of each event to its nearest beat (0..1)."""
    events = np.asarray(events, dtype=np.float64)
    beats = np.asarray(beats, dtype=np.float64)
    if events.size == 0 or beats.size == 0:
        return float("nan")
    d = np.abs(events[:, None] - beats[None, :]).min(axis=1)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)).mean())


def metronome_events(beats: np.ndarray, duration_s: float) -> np.ndarray:
    """The dumb upper reference: one event on every beat."""
    beats = np.asarray(beats, dtype=np.float64)
    return beats[(beats >= 0) & (beats < duration_s)]


def circular_shift_null(events: np.ndarray, beats: np.ndarray, duration_s: float,
                        n: int = 200, sigma: float = DEFAULT_SIGMA,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """BAS of the same event train under random circular time shifts."""
    rng = rng or np.random.default_rng(0)
    events = np.asarray(events, dtype=np.float64)
    if events.size == 0 or duration_s <= 0:
        return np.array([np.nan])
    shifts = rng.uniform(0, duration_s, size=n)
    return np.array([
        beat_alignment(np.mod(events + s, duration_s), beats, sigma) for s in shifts
    ])


def bas_report(events: np.ndarray, beats: np.ndarray, duration_s: float,
               sigma: float = DEFAULT_SIGMA, n_null: int = 200,
               rng: np.random.Generator | None = None) -> dict:
    """BAS plus its metronome ceiling and circular-shift floor."""
    bas = beat_alignment(events, beats, sigma)
    met = beat_alignment(metronome_events(beats, duration_s), beats, sigma)
    null = circular_shift_null(events, beats, duration_s, n=n_null, sigma=sigma, rng=rng)
    nm, ns = float(np.nanmean(null)), float(np.nanstd(null))
    return {
        "bas": bas,
        "metronome": met,
        "null_mean": nm,
        "null_sd": ns,
        "ratio_metronome": bas / met if met and np.isfinite(met) and met > 0 else float("nan"),
        "z": (bas - nm) / ns if ns > 1e-9 else float("nan"),
        "n_events": int(np.asarray(events).size),
        "sigma": sigma,
    }


def offset_scan(events: np.ndarray, beats: np.ndarray, duration_s: float,
                offsets: np.ndarray, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """BAS as a function of a constant time offset applied to ``events``.

    Used to verify corpus pairing (the peak must sit at 0 if the dance clip
    really starts at the track's first sample) and, later, to measure how much
    a late light costs.
    """
    return np.array([beat_alignment(np.asarray(events) + o, beats, sigma)
                     for o in offsets])


def circular_phase(events: np.ndarray, beats: np.ndarray) -> dict:
    """Circular statistics of event phase within the beat.

    Measured on AIST++ (2026-09-04): footfalls lock to the beat PERIOD with
    per-clip concentration R = 0.55 and median Rayleigh z = 5.2, but the
    preferred PHASE is a property of the choreography, not of the music —
    dancers performing the same choreography agree (R = 0.75 within a group)
    while different choreographies sit at different phases (R = 0.17 across
    groups). A metric that measures distance to the nearest beat therefore
    scores real, tightly locked dance at the shift-null. Report ``R`` for
    "is it locked" and ``offset_s`` for "locked to what".
    """
    events = np.asarray(events, dtype=np.float64)
    beats = np.asarray(beats, dtype=np.float64)
    if events.size < 3 or beats.size < 3:
        return {"R": float("nan"), "z": float("nan"), "mu": float("nan"),
                "offset_s": 0.0, "n": int(events.size)}
    ib = float(np.diff(beats).mean())
    inside = events[(events >= beats[0]) & (events < beats[-1])]
    if inside.size < 3:
        return {"R": float("nan"), "z": float("nan"), "mu": float("nan"),
                "offset_s": 0.0, "n": int(inside.size)}
    ph = np.mod((inside - beats[0]) / ib, 1.0)
    a = 2 * np.pi * ph
    C, S = float(np.cos(a).mean()), float(np.sin(a).mean())
    R = float(np.hypot(C, S))
    mu = float((np.arctan2(S, C) / (2 * np.pi)) % 1.0)
    signed = mu if mu <= 0.5 else mu - 1.0          # nearest beat, signed
    return {"R": R, "z": float(inside.size * R * R), "mu": mu,
            "offset_s": signed * ib, "n": int(inside.size), "beat_s": ib}
