"""Constant-tempo beat grid by comb-filter search.

Why not just use ``librosa.beat.beat_track``: measured on this corpus its
tempo estimate has octave errors on a large minority of tracks (mBR0 came back
at 161.5 BPM for a track whose siblings run 90-130), and an octave error moves
every "is the dancer on the beat" number by more than the effect being
measured. AIST's music is loop-based and metronomic (inter-beat-interval
standard deviation 9-17 ms under librosa's own tracker), so the right model is
a *constant* tempo and phase, fitted by maximising onset energy on the grid.

That model is wrong for live concert audio with a human drummer, and it is not
what ships: PLAN.md section 4 puts an online tracker (PLP / BeatNet) on the
inference path. This is a corpus tool for building trustworthy targets.
"""

from __future__ import annotations

import numpy as np

PREFERRED_BPM = (75.0, 155.0)      # where dance music's tactus actually lives
PRIOR_CENTER_BPM = 110.0
PRIOR_SIGMA_OCTAVES = 0.7


def tempo_prior(bpm: np.ndarray | float) -> np.ndarray:
    """Log-normal tempo prior, centred on 120 BPM.

    A comb score is a MEAN over grid points, so a grid at half the true tempo
    scores well by skipping the weaker backbeats: without a prior, 8 of this
    corpus's 60 tracks came back at 60-65 BPM. The prior is the standard fix
    (librosa uses the same log-normal shape) and it is stated here rather than
    hidden in a tolerance constant.
    """
    return np.exp(-0.5 * (np.log2(np.asarray(bpm, dtype=float) / PRIOR_CENTER_BPM)
                          / PRIOR_SIGMA_OCTAVES) ** 2)


def comb_score(onset: np.ndarray, fps: float, period_s: float,
               n_phase: int = 64) -> tuple[float, float]:
    """Best (score, phase) for one candidate beat period.

    Score is the mean onset strength sampled on the grid, normalised by the
    signal's own mean so tracks of different loudness are comparable.
    """
    n = len(onset)
    dur = n / fps
    if period_s <= 0 or period_s > dur / 3:
        return 0.0, 0.0
    t = np.arange(n) / fps
    base = float(np.mean(onset)) + 1e-9
    phases = np.linspace(0.0, period_s, n_phase, endpoint=False)
    grid_k = np.arange(0, int(dur / period_s))
    best, best_ph = -np.inf, 0.0
    for ph in phases:
        gt = ph + grid_k * period_s
        gt = gt[gt < dur]
        if gt.size < 4:
            continue
        s = float(np.interp(gt, t, onset).mean()) / base
        if s > best:
            best, best_ph = s, ph
    return best, best_ph


def fit_constant_grid(
    onset: np.ndarray,
    fps: float,
    phase_onset: np.ndarray | None = None,
    bpm_lo: float = 60.0,
    bpm_hi: float = 200.0,
    n_bpm: int = 141,
    octave_tolerance: float = 0.93,
) -> dict:
    """Fit (tempo, phase) to an onset envelope; return grid + diagnostics.

    Octave handling: the raw argmax is compared against its half and double.
    A candidate inside :data:`PREFERRED_BPM` wins if it scores at least
    ``octave_tolerance`` of the best — the usual "a comb at 2x the tempo hits
    every beat plus the gaps" failure, resolved by a stated prior instead of
    silently.
    """
    onset = np.asarray(onset, dtype=np.float64)
    bpms = np.linspace(bpm_lo, bpm_hi, n_bpm)
    scores, phases = np.zeros(n_bpm), np.zeros(n_bpm)
    for i, bpm in enumerate(bpms):
        scores[i], phases[i] = comb_score(onset, fps, 60.0 / bpm)
    posterior = scores * tempo_prior(bpms)
    i0 = int(np.argmax(posterior))
    best_bpm, best_score = float(bpms[i0]), float(scores[i0])

    # Safety net for the residual octave case: an in-range candidate that
    # nearly ties the winner on raw comb score wins.
    cands = [(best_bpm, best_score * float(tempo_prior(best_bpm)))]
    for factor in (0.5, 2.0, 1 / 3.0, 3.0):
        b = best_bpm * factor
        if bpm_lo <= b <= bpm_hi:
            s_, _ = comb_score(onset, fps, 60.0 / b)
            cands.append((b, s_ * float(tempo_prior(b))))
    in_range = [(b, s_) for b, s_ in cands
                if PREFERRED_BPM[0] <= b <= PREFERRED_BPM[1]
                and s_ >= octave_tolerance * max(c[1] for c in cands)]
    bpm = max(in_range, key=lambda x: x[1])[0] if in_range else best_bpm

    # PERIOD from the broadband onset, PHASE from the kick band when it is
    # supplied. Measured failure this fixes: on mHO2 the broadband comb locked
    # the grid exactly half a beat late (299 ms vs 47 ms) because the snare and
    # hats carry more spectral flux than the kick, and every "is the dancer on
    # the beat" number inherited that half-beat error.
    period = 60.0 / bpm
    ph_src = onset if phase_onset is None else np.asarray(phase_onset, dtype=np.float64)
    score, phase = comb_score(ph_src, fps, period, n_phase=512)
    dur = len(onset) / fps
    beats = np.arange(phase, dur, period)

    # downbeat: the bar phase (of 4) whose beats carry the most onset energy
    t = np.arange(len(ph_src)) / fps
    strength = np.interp(beats, t, ph_src)
    bar_scores = [float(strength[p::4].sum()) for p in range(4)] if beats.size >= 8 else [0]
    p0 = int(np.argmax(bar_scores))
    return {
        "bpm": bpm,
        "period_s": period,
        "phase_s": phase,
        "beats": beats,
        "downbeats": beats[p0::4],
        "comb_score": score,
        "raw_argmax_bpm": best_bpm,
        "octave_corrected": bool(abs(bpm - best_bpm) > 1e-6),
        "bar_phase": p0,
        "phase_from_kick": phase_onset is not None,
    }
