"""Low-level audio features: beats, downbeats, onsets, envelopes, spectra.

All frame-rate quantities share ``SR``/``HOP`` so every curve in the pipeline
lives on one common time grid (``frame_times``). Mixing grids is the single
easiest way to introduce a silent off-by-N drift between the light cue and
the music, so nothing here returns a curve without also being able to state
its time base.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .. import HOP, SR


@dataclass
class AudioFeatures:
    """Frame-aligned audio feature bundle for one clip."""

    y: np.ndarray
    sr: int
    hop: int
    duration_s: float
    frame_times: np.ndarray          # (F,) seconds
    rms: np.ndarray                  # (F,) broadband energy
    onset_env: np.ndarray            # (F,) spectral-flux novelty
    centroid: np.ndarray             # (F,) spectral centroid (Hz)
    rolloff: np.ndarray              # (F,) 95% spectral rolloff (Hz)
    flatness: np.ndarray             # (F,) 0=tonal 1=noisy
    bandwidth: np.ndarray            # (F,) spectral bandwidth
    contrast: np.ndarray             # (F,) mean spectral contrast
    mel: np.ndarray                  # (n_mels, F) log-mel
    chroma: np.ndarray               # (12, F)
    tempo: float                     # global BPM estimate
    beats: np.ndarray                # (B,) beat times, seconds
    downbeats: np.ndarray            # (D,) inferred downbeat times, seconds
    beats_per_bar: int

    @property
    def n_frames(self) -> int:
        return int(self.frame_times.shape[0])

    def at(self, curve: np.ndarray, t: float) -> float:
        """Sample a frame-aligned curve at time ``t`` seconds."""
        return float(np.interp(t, self.frame_times, curve))


def _norm(x: np.ndarray) -> np.ndarray:
    """Scale to [0,1]; a constant signal maps to zeros, never NaN."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def infer_downbeats(
    beats: np.ndarray,
    onset_env: np.ndarray,
    frame_times: np.ndarray,
    beats_per_bar: int = 4,
) -> np.ndarray:
    """Pick the bar phase whose beats carry the most onset energy.

    librosa has no downbeat tracker. For 4/4 EDM the bar phase is usually the
    one where the kick lands, so we score each of the ``beats_per_bar``
    candidate phases by summed onset strength and take the argmax.

    This is a HEURISTIC, not a trained downbeat model. It is the documented
    upgrade point (All-In-One / Beat This! / BeatNet).
    """
    if beats.size == 0:
        return beats
    strength = np.interp(beats, frame_times, onset_env)
    scores = [
        float(strength[phase::beats_per_bar].sum())
        for phase in range(beats_per_bar)
    ]
    best = int(np.argmax(scores))
    return beats[best::beats_per_bar]


def extract(
    y: np.ndarray | None = None,
    path: str | None = None,
    sr: int = SR,
    hop: int = HOP,
    duration_s: float | None = None,
    offset_s: float = 0.0,
    n_mels: int = 128,
    beats_per_bar: int = 4,
) -> AudioFeatures:
    """Compute the full feature bundle for a clip.

    ``duration_s`` clips the audio — REQUIRED when pairing with an AIST pose
    sequence, whose clip is far shorter than the full track (see
    docs/DATA_NOTES.md).
    """
    if y is None:
        if path is None:
            raise ValueError("provide either y or path")
        y, sr = librosa.load(path, sr=sr, offset=offset_s, duration=duration_s)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("empty audio")

    dur = float(len(y) / sr)
    S = np.abs(librosa.stft(y, hop_length=hop))
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    rms = librosa.feature.rms(S=S, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S**2), sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr).mean(axis=0)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=n_mels)
    )
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    downbeats = infer_downbeats(beats, onset_env, frame_times, beats_per_bar)

    # Guard: every frame-aligned curve must share one length.
    F = S.shape[1]
    for name, arr in [
        ("rms", rms), ("onset_env", onset_env), ("centroid", centroid),
        ("rolloff", rolloff), ("flatness", flatness), ("bandwidth", bandwidth),
        ("contrast", contrast),
    ]:
        if arr.shape[0] != F:
            raise AssertionError(f"{name} has {arr.shape[0]} frames, expected {F}")

    return AudioFeatures(
        y=y, sr=sr, hop=hop, duration_s=dur, frame_times=frame_times,
        rms=rms, onset_env=onset_env, centroid=centroid, rolloff=rolloff,
        flatness=flatness, bandwidth=bandwidth, contrast=contrast,
        mel=mel, chroma=chroma, tempo=tempo, beats=beats,
        downbeats=downbeats, beats_per_bar=beats_per_bar,
    )
