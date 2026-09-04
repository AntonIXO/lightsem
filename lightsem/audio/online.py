"""Causal audio feature bus — the only thing the product may read at inference.

PLAN.md section 4 fixes the conditioning: log-mel + RMS + spectral flux in
SEPARATE low and high bands + onset + beat phase. The band split is not
cosmetic: Burger et al. 2013 measured that low-frequency flux (50-100 Hz,
kick/bass) drives *head* speed (r = 0.73) while high-frequency flux
(6.4-12.8 kHz, hats) drives *hand* speed (r = 0.65). A single broadband flux
channel throws away the one distinction the rig mapping is built on, so the
bands are first-class channels here.

Sample rate is 44.1 kHz, not the 22.05 kHz used elsewhere in this repo, for
exactly one reason: at 22.05 kHz the Nyquist limit is 11 kHz and the top of
Burger's high band (12.8 kHz) does not exist.

Causality. Every channel here is computable from past samples only:

* flux/onset use frame t and t-1;
* smoothing is a one-pole EMA (never a centred window) when ``causal=True``;
* pulse clarity uses a TRAILING autocorrelation window;
* ``beat_*`` channels come from an OFFLINE beat tracker and are therefore
  **not** causal. They are marked ``oracle_beat`` in ``AudioBus.flags`` and
  exist so target extraction and the Stage 0b audit can use a correct beat
  grid. Stage 4 replaces them with an online tracker (PLP/BeatNet). Anything
  that reports a product-facing number must check the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .beatgrid import fit_constant_grid

SR_BUS = 44100
HOP_BUS = 1024                     # 43.07 fps analysis grid
CTRL_FPS = 40.0

#: Channels derived from the offline constant-tempo grid. The grid is fitted
#: over the whole track, so these see the entire future by construction; the bus
#: reports them under flags["oracle_beat"] and a deployment has to replace them
#: with a running tracker. Everything NOT in this set is strictly causal.
ORACLE_CHANNELS = ("beat_phase_sin", "beat_phase_cos", "beat_period_s",
                   "beat_accent", "bar_phase")

#: (low_hz, high_hz) per band. LF and HF follow Burger et al. 2013.
BANDS: dict[str, tuple[float, float]] = {
    "lf": (50.0, 100.0),
    "lmid": (100.0, 400.0),
    "mid": (400.0, 1600.0),
    "hmid": (1600.0, 6400.0),
    "hf": (6400.0, 12800.0),
}


@dataclass
class AudioBus:
    """Frame-aligned causal audio features at the control rate."""

    fps: float
    times: np.ndarray
    channels: dict[str, np.ndarray]
    mel: np.ndarray                      # (n_mels, T) log-mel, control rate
    beats: np.ndarray                    # offline beat grid, seconds
    downbeats: np.ndarray
    tempo: float
    duration_s: float
    flags: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        names = list(names or sorted(self.channels))
        return np.stack([self.channels[n] for n in names], axis=1), names

    def at(self, name: str, t: float | np.ndarray) -> np.ndarray:
        return np.interp(t, self.times, self.channels[name])


def ema(x: np.ndarray, fps: float, tau_s: float) -> np.ndarray:
    """Causal one-pole smoother with time constant ``tau_s``."""
    x = np.asarray(x, dtype=np.float64)
    if tau_s <= 0:
        return x
    a = float(np.exp(-1.0 / (tau_s * fps)))
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        out[i] = acc
    return out


def _resample(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray,
              causal: bool = True) -> np.ndarray:
    """Analysis grid -> control grid.

    Interpolating would read the analysis frame *after* the control frame, so
    the causal path holds the newest frame that has already finished instead.
    The cost is up to one analysis hop (23 ms) of staleness; the alternative is
    23 ms of look-ahead, which is the thing the whole bus is trying not to do.
    """
    x = np.asarray(x, dtype=np.float64)
    if not causal:
        return np.interp(t_out, t_in, x)
    j = np.searchsorted(t_in, t_out, side="right") - 1
    return x[np.clip(j, 0, len(x) - 1)]


def _percussive_ratio(S: np.ndarray, causal: bool = True,
                      kt: int = 31, kf: int = 31) -> np.ndarray:
    """Share of the spectrum that is transient rather than sustained.

    Same median-filter soft mask as librosa's HPSS, with one change: the
    harmonic estimate is a median over the *trailing* kt frames instead of a
    centred one. Centred, it reads 15 frames of the future -- 350 ms, and this
    channel's leak measured 15x its own standard deviation, the largest in the
    bus. The percussive estimate medians across frequency at a single frame and
    was already causal.
    """
    from scipy.ndimage import median_filter
    S = np.asarray(S, dtype=np.float64)
    # scipy's positive `origin` shifts the window toward earlier indices, so
    # origin=(kt-1)//2 makes output[i] the median of input[i-kt+1 .. i].
    org = (0, (kt - 1) // 2) if causal else (0, 0)
    harm = median_filter(S, size=(1, kt), origin=org, mode="nearest")
    perc = median_filter(S, size=(kf, 1), mode="nearest")
    p2, h2 = perc ** 2, harm ** 2
    mask_p = p2 / np.maximum(p2 + h2, 1e-12)
    return (S * mask_p).sum(axis=0) / np.maximum(S.sum(axis=0), 1e-9)


def trailing_pulse_clarity(
    onset: np.ndarray,
    fps: float,
    win_s: float = 4.0,
    lo_s: float = 0.25,
    hi_s: float = 2.0,
    hop: int = 4,
) -> np.ndarray:
    """Pulse clarity: peak of the trailing-window onset autocorrelation.

    MIRtoolbox's pulse clarity in the one form that is legal in a real-time
    system — the window ends at the current frame, so no future audio is read.
    """
    onset = np.asarray(onset, dtype=np.float64)
    n = len(onset)
    n_win = max(int(round(win_s * fps)), 16)
    lo, hi = max(int(round(lo_s * fps)), 1), int(round(hi_s * fps))
    out = np.zeros(n)
    idx = np.arange(0, n, hop)
    vals = np.zeros(len(idx))
    for k, i in enumerate(idx):
        s = max(i - n_win, 0)
        seg = onset[s:i + 1]
        if len(seg) < 3 * lo:
            continue
        seg = seg - seg.mean()
        denom = float(seg @ seg)
        if denom < 1e-12:
            continue
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:] / denom
        top = min(hi, len(ac) - 1)
        if top <= lo:
            continue
        vals[k] = float(np.clip(ac[lo:top + 1].max(), 0.0, 1.0))
    # Hold the last measurement rather than interpolating to the next one.
    # np.interp would mix in vals[k+1], which was measured from onset up to
    # idx[k+1] > j -- up to `hop` frames (~70 ms at the 1024-sample hop) of
    # future audio inside a channel this file promises is causal, and one that
    # AUDIO_SCALARS feeds to the model as an input. The cost of holding is that
    # the value is up to `hop` frames stale, which is nothing against a 4 s
    # analysis window.
    out = vals[np.searchsorted(idx, np.arange(n), side="right") - 1]
    return out


def extract_bus(
    path: str | Path | None = None,
    y: np.ndarray | None = None,
    sr: int = SR_BUS,
    hop: int = HOP_BUS,
    duration_s: float | None = None,
    fps_out: float = CTRL_FPS,
    n_mels: int = 64,
    causal: bool = True,
) -> AudioBus:
    """Compute the conditioning bus for one track."""
    if y is None:
        if path is None:
            raise ValueError("provide path or y")
        y, sr = librosa.load(str(path), sr=sr, duration=duration_s)
    y = np.asarray(y, dtype=np.float32)
    if y.size < hop * 4:
        raise ValueError("audio too short")

    # center=False, and every frame is timestamped at the END of its window.
    # librosa centres frames by default, which timestamps a 2048-sample window
    # at its midpoint and so hands every spectral channel 23 ms of the future --
    # in a bus whose docstring promises the opposite, and whose channels are
    # model inputs. With center=False the frame covering [i*hop, i*hop+n_fft) is
    # available once that last sample has arrived, which is what a streaming
    # implementation can actually produce.
    n_fft = 2 * hop
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=not causal))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    if causal:
        t_in = (np.arange(S.shape[1]) * hop + n_fft) / sr
    else:
        t_in = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    fps_in = sr / hop
    dur = float(len(y) / sr)

    n_out = int(round(dur * fps_out))
    t_out = np.arange(n_out) / fps_out

    ch: dict[str, np.ndarray] = {}

    rms = librosa.feature.rms(S=S, frame_length=2 * hop, hop_length=hop)[0]
    ch["rms"] = rms
    # top_db defaults to 80 and clamps against max() over the whole clip, which
    # is a future-dependent floor. There is no streaming version of that, so it
    # is turned off rather than approximated.
    ch["rms_db"] = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0, top_db=None)

    logS = np.log1p(S)
    dS = np.diff(logS, axis=1, prepend=logS[:, :1])
    flux_full = np.maximum(dS, 0.0)
    ch["flux"] = flux_full.sum(axis=0)
    for name, (f_lo, f_hi) in BANDS.items():
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not m.any():
            ch[f"flux_{name}"] = np.zeros(S.shape[1])
            ch[f"energy_{name}"] = np.zeros(S.shape[1])
            continue
        ch[f"flux_{name}"] = flux_full[m].sum(axis=0)
        ch[f"energy_{name}"] = S[m].mean(axis=0)

    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2, top_db=None),
                                         sr=sr, hop_length=hop)
    onset = onset[: S.shape[1]] if len(onset) >= S.shape[1] else np.pad(
        onset, (0, S.shape[1] - len(onset)))
    ch["onset"] = onset

    ch["centroid"] = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    ch["flatness"] = librosa.feature.spectral_flatness(S=S)[0]
    ch["rolloff"] = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]

    ch["percussive_ratio"] = _percussive_ratio(S, causal=causal)

    ch["pulse_clarity"] = trailing_pulse_clarity(onset, fps_in)

    # --- offline beat grid (NOT causal; flagged) --------------------------
    # Constant-tempo comb fit, not librosa's DP tracker: on this corpus the
    # latter returns octave-wrong tempi often enough to move every alignment
    # number more than the effect being measured (see audio/beatgrid.py).
    # The grid is fitted on a FINER hop than the feature bus: at hop 1024 one
    # analysis frame is 23 ms, which is a quarter of the flash-tone JND, and
    # the fitted phase moved by a full half beat between hop sizes.
    hop_g = 256
    S_g = np.abs(librosa.stft(y, n_fft=2 * hop_g, hop_length=hop_g))
    fps_g = sr / hop_g
    onset_g = librosa.onset.onset_strength(S=librosa.power_to_db(S_g ** 2), sr=sr,
                                           hop_length=hop_g)
    freqs_g = librosa.fft_frequencies(sr=sr, n_fft=2 * hop_g)
    logSg = np.log1p(S_g)
    dSg = np.maximum(np.diff(logSg, axis=1, prepend=logSg[:, :1]), 0.0)
    kick = dSg[(freqs_g >= 40.0) & (freqs_g < 120.0)].sum(axis=0)
    kick = kick[: len(onset_g)] if len(kick) >= len(onset_g) else np.pad(
        kick, (0, len(onset_g) - len(kick)))
    grid = fit_constant_grid(onset_g, fps_g, phase_onset=kick)
    tempo = float(grid["bpm"])
    beats = grid["beats"]
    downbeats = grid["downbeats"]

    if beats.size >= 2:
        ibi = np.diff(beats)
        beat_period = np.interp(t_out, beats[:-1] + ibi / 2, ibi)
        idx = np.searchsorted(beats, t_out, side="right") - 1
        idx_c = np.clip(idx, 0, beats.size - 1)
        nxt = np.where(idx_c + 1 < beats.size, beats[np.clip(idx_c + 1, 0, beats.size - 1)],
                       beats[idx_c] + beat_period)
        span = np.maximum(nxt - beats[idx_c], 1e-6)
        phase = np.clip((t_out - beats[idx_c]) / span, 0.0, 1.0)
        phase = np.where(idx < 0, 0.0, phase)
        beat_index = np.maximum(idx, 0)
    else:
        beat_period = np.full(n_out, 60.0 / max(tempo, 1e-6))
        phase = np.zeros(n_out)
        beat_index = np.zeros(n_out, dtype=int)

    # --- resample analysis-rate channels onto the control grid ------------
    out: dict[str, np.ndarray] = {}
    for k, v in ch.items():
        v = np.asarray(v, dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if causal and k in ("rms", "pulse_clarity", "percussive_ratio", "centroid",
                            "flatness", "rolloff"):
            v = ema(v, fps_in, 0.08)
        out[k] = _resample(v, t_in, t_out, causal=causal)

    out["beat_phase_sin"] = np.sin(2 * np.pi * phase)
    out["beat_phase_cos"] = np.cos(2 * np.pi * phase)
    out["beat_period_s"] = beat_period
    out["beat_accent"] = (1.0 - phase) ** 2.5
    out["bar_phase"] = (beat_index % 4 + phase) / 4.0
    for k in ("flux", "flux_lf", "flux_hf", "onset"):
        out[k + "_ema"] = ema(out[k], fps_out, 0.15)

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=n_mels), top_db=None)
    mel_out = np.stack([_resample(mel[i], t_in, t_out, causal=causal)
                        for i in range(mel.shape[0])])

    return AudioBus(
        fps=fps_out, times=t_out, channels=out, mel=mel_out,
        beats=beats, downbeats=downbeats, tempo=tempo, duration_s=dur,
        flags={"causal_spectral": causal, "oracle_beat": True},
        meta={"sr": sr, "hop": hop, "path": str(path) if path else None,
              "n_beats": int(beats.size),
              "comb_score": float(grid["comb_score"]),
              "octave_corrected": bool(grid["octave_corrected"]),
              "raw_argmax_bpm": float(grid["raw_argmax_bpm"]),
              "grid_hop": hop_g, "phase_s": float(grid["phase_s"])},
    )
