"""Put the Oslo stimulus audio back on the record's clock.

Zenodo 21945721 ships the two DJ mixes as features only -- the four commercial
recordings underneath are not deposited, and the README names them precisely so
that a user can obtain them. This module takes those four recordings, once
obtained, and reconstructs `DanceMix1` / `DanceMix2` on the same clock as
`audio_features/feature_series.tsv`, which is the clock the mocap is on.

The reconstruction is not a concatenation. Both mixes are beat-matched, so each
track was played off its native tempo -- `stimulus_tracks.tsv` lists Ladykiller
at 124 BPM in mix 1 and 127 in mix 2, one recording at two pitch settings. A DJ
pitch fader resamples, so the stretch here resamples too: it moves pitch with
tempo, which is what the room heard. The ratio is not taken from the nominal
BPMs, which are logged to the nearest integer and would leave a drift of a beat
per minute; it is fitted per track against the published onset envelope.

Nothing here decrypts a stream or touches a streaming account. `ingest_dir`
reads audio files the user already has, and `resolve_stimulus` asks Deezer's
public metadata API which master to look for.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/lightsem-data/oslo")
S2015 = ROOT / "Solberg2015_EDM"
AUDIO = ROOT / "audio"
SOURCES = AUDIO / "sources"
MIXES = AUDIO / "mixes"

SR = 44100
HOP = 512
FPS = SR / HOP          # 86.1328125, the record's analysis rate
MIX_NAME = {1: "DanceMix1", 2: "DanceMix2"}

#: The four recordings, as named in audio_features/README.md. `query` is what
#: goes to the metadata API; `aliases` are what a purchased filename tends to
#: look like once a store has had its way with it.
STIMULUS = {
    "Joyride": dict(
        artist="Rhode & Brown", featuring="Schegg", label="Toy Tonics", year=2013,
        query='artist:"Rhode & Brown" track:"Joyride"',
        aliases=["joyride", "rhode", "brown", "schegg"]),
    "Ladykiller": dict(
        artist="Vanilla Ace", version="Original Mix", label="Erase Records", year=2014,
        query='artist:"Vanilla Ace" track:"Ladykiller"',
        aliases=["ladykiller", "lady killer", "vanilla ace"]),
    "Unlock Down": dict(
        artist="NUAGE", version="Original Mix", label="Electronica Records", year=2014,
        query='artist:"NUAGE" track:"Unlock Down"',
        aliases=["unlock down", "unlockdown", "nuage"]),
    "Icarus": dict(
        artist="Madeon", version="Extended Mix", label="Popcultur", year=2012,
        query='artist:"Madeon" track:"Icarus"',
        aliases=["icarus", "madeon"]),
}


# ---------------------------------------------------------------- record side

def load_stimulus_tracks() -> list[dict]:
    with open(S2015 / "metadata" / "stimulus_tracks.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["dj_mix"] = int(r["dj_mix"])
        r["track_order"] = int(r["track_order"])
        r["bpm"] = float(r["bpm"])
        for k in ("start_s", "end_s", "breakdown_s", "buildup_s", "drop_s"):
            r[k] = float(r[k]) if r[k].strip() else None
    return rows


def load_feature_series(mix: int) -> dict[str, np.ndarray]:
    """The published per-frame features for one mix, at 86.13 fps."""
    name = MIX_NAME[mix]
    cols = ["t_s", "rms_dbfs", "spectral_centroid_hz", "spectral_bandwidth_hz",
            "rolloff85_hz", "spectral_flatness", "zcr", "bass_frac_below150hz",
            "onset_strength", "onset", "tempo_bpm_w8", "pulse_clarity_w8"]
    acc: dict[str, list] = {c: [] for c in cols}
    with open(S2015 / "audio_features" / "feature_series.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["track"] != name:
                continue
            for c in cols:
                v = r[c].strip()
                acc[c].append(float(v) if v not in ("", "nan") else np.nan)
    return {c: np.asarray(v, dtype=np.float64) for c, v in acc.items()}


def mix_duration(mix: int) -> float:
    with open(S2015 / "audio_features" / "acoustic_features.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["track"] == MIX_NAME[mix]:
                return float(r["duration_s"])
    raise KeyError(mix)


# ------------------------------------------------------------- our audio side

def _mono(y: np.ndarray) -> np.ndarray:
    """The record computes on the mean of the channels, so we do too."""
    return y if y.ndim == 1 else y.mean(axis=0)


def read_audio(path: Path, sr: int = SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=False)
    return _mono(np.atleast_2d(y)).astype(np.float64)


def onset_envelope(y: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa
    return librosa.onset.onset_strength(y=y.astype(np.float32), sr=sr, hop_length=HOP)


def frame_features(y: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    """Recompute the record's columns from audio, with the record's definitions.

    Same hop, same mono rule, same 85 % rolloff and same 150 Hz bass split, so a
    column here and the published column of the same name are comparable frame
    by frame. `onset_strength` is spectral flux and carries no absolute scale in
    either series, which is why every comparison downstream is a correlation.
    """
    import librosa
    yf = y.astype(np.float32)
    S = np.abs(librosa.stft(yf, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    power = S ** 2
    tot = power.sum(axis=0)
    bass = power[freqs < 150.0].sum(axis=0)
    rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=HOP)[0]
    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return {
        "rms_dbfs": rms_db,
        "spectral_centroid_hz": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "spectral_bandwidth_hz": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
        "rolloff85_hz": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "spectral_flatness": librosa.feature.spectral_flatness(S=S)[0],
        "zcr": librosa.feature.zero_crossing_rate(yf, frame_length=2048, hop_length=HOP)[0],
        "bass_frac_below150hz": bass / np.maximum(tot, 1e-20),
        "onset_strength": librosa.onset.onset_strength(S=librosa.power_to_db(power), sr=sr),
    }


#: Channels the alignment matches on. Onset flux alone does not discriminate
#: here: measured on this corpus its best correlation 30 s away from the answer
#: is 0.49-0.71, because a beat-matched EDM mix is four-on-the-floor at 122-127
#: BPM throughout and every bar's flux looks like every other bar's. Bass
#: fraction and brightness do discriminate, because a breakdown and a drop are
#: not the same sound, and they are published on the same clock.
FINGERPRINT = ("onset_strength", "rms_dbfs", "bass_frac_below150hz",
               "spectral_centroid_hz", "spectral_flatness")


def fingerprint(y: np.ndarray) -> np.ndarray:
    """(C, T) alignment channels for a recording, in FINGERPRINT order."""
    f = frame_features(y)
    f["onset_strength"] = onset_envelope(y)
    n = min(len(f[c]) for c in FINGERPRINT)
    return np.stack([np.nan_to_num(f[c][:n], nan=0.0) for c in FINGERPRINT])


def ref_fingerprint(mix: int) -> np.ndarray:
    """The same channels for a mix, taken from the record's published series."""
    r = load_feature_series(mix)
    n = min(len(r[c]) for c in FINGERPRINT)
    return np.stack([np.nan_to_num(r[c][:n], nan=0.0) for c in FINGERPRINT])


# ------------------------------------------------------------------ alignment

@dataclass
class Placement:
    """Where one recording sits inside one mix, and how well that is known."""
    title: str
    mix: int
    track_order: int
    nominal_start_s: float
    nominal_end_s: float
    nominal_bpm: float
    source_file: str
    ratio: float              # played tempo / native tempo; >1 is pitched up
    offset_s: float           # mix time at which the recording's t=0 falls
    corr: float               # peak envelope correlation at (ratio, offset)
    null_corr: float          # best correlation this envelope reaches elsewhere
    margin: float             # corr - null_corr
    native_bpm: float         # measured tempo of the supplied recording
    played_bpm: float         # measured tempo of the slot, from the record
    ratio_hint: float         # played_bpm / native_bpm, the bracket centre
    fitted_bpm: float         # played tempo past the log's integer resolution
    used_s: float             # how much of the recording the mix actually uses


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def _ncc(ref: np.ndarray, cand: np.ndarray,
         min_overlap_s: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation at every lag, computed over the overlap alone.

    Returns (r, lags) where a lag is where the candidate's first frame falls in
    the reference, and may be negative. Normalising over the overlap rather than
    over the whole of either series is not a refinement, it is what makes the
    numbers mean anything here: the recordings are longer than the slots they
    were mixed into -- Ladykiller runs 407 s and occupies 232 s of mix 1 -- so
    the candidate always hangs off both ends, and a global norm would then rate
    each ratio against a different denominator and make the ratio search a
    comparison of unlike things. Overlaps shorter than `min_overlap_s` are
    dropped, since a few seconds of anything correlates with a few seconds of
    anything else.
    """
    from scipy.signal import fftconvolve
    x = np.where(np.isfinite(ref), ref, 0.0).astype(np.float64)
    y = np.where(np.isfinite(cand), cand, 0.0).astype(np.float64)
    nr, nc = len(x), len(y)
    sxy = fftconvolve(x, y[::-1], mode="full")          # index k <-> lag k-(nc-1)
    lags = np.arange(nr + nc - 1) - (nc - 1)
    cx, cx2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
    cy, cy2 = np.r_[0.0, np.cumsum(y)], np.r_[0.0, np.cumsum(y * y)]
    a0 = np.clip(lags, 0, nr)
    a1 = np.clip(lags + nc, 0, nr)
    b0 = np.clip(-lags, 0, nc)
    b1 = np.clip(nr - lags, 0, nc)
    n = (a1 - a0).astype(np.float64)
    sx, sxx = cx[a1] - cx[a0], cx2[a1] - cx2[a0]
    sy, syy = cy[b1] - cy[b0], cy2[b1] - cy2[b0]
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0.0)
                  * np.maximum(n * syy - sy ** 2, 0.0))
    r = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    r[n < min_overlap_s * FPS] = -1.0
    return r, lags


def _resample_env(env: np.ndarray, ratio: float) -> np.ndarray:
    """Play `env` at `ratio` times speed; length shrinks as the ratio grows.

    Accepts a single channel or a (C, T) stack.
    """
    a = np.asarray(env, dtype=np.float64)
    n_out = max(int(round(a.shape[-1] / ratio)), 2)
    idx = np.arange(n_out) * ratio
    src = np.arange(a.shape[-1])
    if a.ndim == 1:
        return np.interp(idx, src, a)
    return np.stack([np.interp(idx, src, row) for row in a])


def _ncc_multi(ref: np.ndarray, cand: np.ndarray,
               min_overlap_s: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-lag correlation averaged over the fingerprint channels.

    Averaging the per-channel correlations, rather than correlating one
    concatenated vector, keeps each channel on its own scale -- centroid runs in
    thousands of hertz and bass fraction in tenths, and a concatenation would be
    a centroid match wearing the others as noise.
    """
    ref, cand = np.atleast_2d(ref), np.atleast_2d(cand)
    acc, lags = None, None
    for i in range(ref.shape[0]):
        c, lags = _ncc(ref[i], cand[i], min_overlap_s)
        acc = c if acc is None else acc + c
    return acc / ref.shape[0], lags


def _structure_env(env: np.ndarray, win_s: float = 2.0) -> np.ndarray:
    """The arrangement contour: onset energy smoothed past the bar.

    A box filter wider than a bar erases the pulse and leaves what the
    arrangement does -- the kick entering, the breakdown emptying out, the drop.
    That contour is what makes a position in a track unique.
    """
    from scipy.ndimage import uniform_filter1d
    n = max(int(win_s * FPS), 3)
    return uniform_filter1d(np.asarray(env, dtype=np.float64), n,
                            axis=-1, mode="nearest")


def measure_bpm(env: np.ndarray) -> float:
    """Tempo of an onset envelope, via the project's comb-filter grid."""
    from lightsem.audio import beatgrid
    return float(beatgrid.fit_constant_grid(env, FPS)["bpm"])


def fit_placement(ref_env: np.ndarray, cand_env: np.ndarray,
                  ratio_hint: float | None = None,
                  ratio_lo: float = 0.90, ratio_hi: float = 1.10,
                  step: float = 0.00025, refine_win_s: float = 0.8,
                  min_overlap_s: float = 45.0,
                  cover: tuple[int, int] | None = None,
                  cover_tol_s: float = 6.0) -> tuple[float, int, float]:
    """Best (ratio, lag in frames, correlation) of a recording against a mix.

    Three quantities, three sources, because no one series carries all of them.

    The ratio is swept across the whole range a beat-matched mix can occupy
    rather than bracketed around an estimate, and that is a correction of two
    earlier attempts. Bracketing by the correlation of smoothed envelopes does
    not work, because across a two-percent sweep that correlation moves by
    0.001 and so brackets at random. Bracketing by the quotient of two measured
    tempi does not work either, and fails worse for looking reasonable: tempo
    estimation is metrically ambiguous, the two sides can settle on different
    metrical levels, and the quotient then comes back as 0.73 for a track whose
    true ratio is 1.0135 -- on a controlled test with known ground truth that
    put two of four real recordings 80 s and 378 s out. A bracket cannot
    recover from a hint that wrong.

    Sweeping instead costs a few hundred more correlations and needs no
    estimate, because the constraint is physical: a DJ beat-matching two decks
    stays inside the pitch fader's range, so the ratio is near one by
    construction. `ratio_hint` is now carried only as a diagnostic.

    Inside that bracket the ratio is resolved by the raw onset envelope, which
    is razor sharp in ratio -- 0.91 at the truth against 0.07 two percent away
    -- and useless for position, since every bar of four-on-the-floor looks
    like every other and its peak lands a whole number of bars out.

    So position comes from the arrangement contour, which is unique within a
    track, and the two are searched jointly rather than in sequence because the
    lag moves with the ratio: an error of 0.02 puts the contour's own peak 1.6 s
    out, already further than the snap window. For each candidate ratio the
    contour says which bar and the transients say where in it, and the ratio
    that wins is the one whose transients then fit best. The snap window is kept
    under half a bar so it cannot alias back out.
    """
    ref_s = _structure_env(ref_env)
    win = int(refine_win_s * FPS)
    best = (1.0, 0, -2.0)
    tol = int(cover_tol_s * FPS)
    n_grid = max(int(round((ratio_hi - ratio_lo) / step)) + 1, 3)
    for r in np.linspace(ratio_lo, ratio_hi, n_grid):
        cand_r = _resample_env(cand_env, r)
        cs, ls = _ncc_multi(ref_s, _structure_env(cand_r), min_overlap_s)
        if cover is not None:                               # must fill the slot
            ok = (ls >= cover[1] - cand_r.shape[-1] - tol) & (ls <= cover[0] + tol)
            if not ok.any():
                continue
            cs = np.where(ok, cs, -2.0)
        lag_s = int(ls[int(np.argmax(cs))])                 # which bar
        c, lags = _ncc_multi(ref_env, cand_r, min_overlap_s)
        sel = np.flatnonzero(np.abs(lags - lag_s) <= win)   # where in it
        if cover is not None:
            sel = sel[(lags[sel] >= cover[1] - cand_r.shape[-1] - tol)
                      & (lags[sel] <= cover[0] + tol)]
        if not len(sel):
            continue
        i = int(sel[int(np.argmax(c[sel]))])
        if c[i] > best[2]:
            best = (float(r), int(lags[i]), float(c[i]))
    return best


def null_correlation(ref_env: np.ndarray, cand_env: np.ndarray, ratio: float,
                     lag: int, guard_s: float = 30.0,
                     min_overlap_s: float = 45.0) -> float:
    """The best this recording scores anywhere the answer is not.

    A high correlation means nothing on its own here: a beat-matched EDM mix is
    four-on-the-floor throughout, so a recording will correlate with some
    unrelated stretch of it at almost any lag. The peak is therefore priced
    against the best peak outside a guard band around it, and it is the margin,
    not the peak, that says the placement is real.

    Measured on the same channels and the same normalisation the peak came
    from. Scoring the null on the smoothed contour instead makes the margin a
    difference of two incommensurable quantities, and it then reads negative on
    placements that are in fact correct.
    """
    c, lags = _ncc_multi(ref_env, _resample_env(cand_env, ratio), min_overlap_s)
    mask = np.abs(lags - lag) > int(guard_s * FPS)
    return float(c[mask].max()) if mask.any() else float("nan")


def place_track(row: dict, src: Path, ref_env: np.ndarray,
                pad_s: float = 25.0) -> Placement:
    """Fit one recording into its slot in one mix.

    The played tempo is measured on the nominal slot alone, not on the padded
    window the correlation runs over: the pad deliberately spills into the
    neighbouring tracks so the fit can move, and those neighbours are at other
    tempi.
    """
    cand = fingerprint(read_audio(src))
    lo = max(0, int((row["start_s"] - pad_s) * FPS))
    hi = min(ref_env.shape[-1], int((row["end_s"] + pad_s) * FPS))
    seg = ref_env[:, lo:hi]
    core = ref_env[0, int(row["start_s"] * FPS): int(row["end_s"] * FPS)]
    native_bpm, played_bpm = measure_bpm(cand[0]), measure_bpm(core)
    hint = played_bpm / native_bpm
    overlap_s = min(0.6 * (hi - lo) / FPS, 45.0)
    cover = (int(row["start_s"] * FPS) - lo, int(row["end_s"] * FPS) - lo)
    ratio, lag, corr = fit_placement(seg, cand, min_overlap_s=overlap_s,
                                     cover=cover)
    null = null_correlation(seg, cand, ratio, lag, min_overlap_s=overlap_s)
    return Placement(
        title=row["title"], mix=row["dj_mix"], track_order=row["track_order"],
        nominal_start_s=row["start_s"], nominal_end_s=row["end_s"],
        nominal_bpm=row["bpm"], source_file=str(src),
        ratio=ratio, offset_s=(lo + lag) / FPS, corr=corr, null_corr=null,
        margin=corr - null, native_bpm=native_bpm, played_bpm=played_bpm,
        ratio_hint=hint, fitted_bpm=native_bpm * ratio,
        used_s=row["end_s"] - row["start_s"])


# --------------------------------------------------------------- mix assembly

def _resample_audio(y: np.ndarray, ratio: float) -> np.ndarray:
    from scipy.signal import resample_poly
    f = Fraction(ratio).limit_denominator(2000)
    return resample_poly(y, f.denominator, f.numerator)


def render_mix(mix: int, placements: list[Placement],
               fade_s: float = 4.0) -> np.ndarray:
    """Lay the placed recordings onto the mix clock and cross-fade the seams.

    Each recording is resampled by its fitted ratio, dropped at its fitted
    offset, and windowed to its nominal slot with an equal-power fade across the
    overlap. The seams will not match the DJ's EQ moves -- nothing recoverable
    from the published features could tell us those -- so the reconstruction is
    trustworthy inside a track and approximate for a few seconds either side of
    a transition. `verify` reports that boundary cost per track.
    """
    n = int(round(mix_duration(mix) * SR))
    out = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    fade = int(fade_s * SR)
    for p in sorted(placements, key=lambda q: q.track_order):
        y = _resample_audio(read_audio(Path(p.source_file)), p.ratio)
        a = int(round(p.nominal_start_s * SR))
        b = int(round(p.nominal_end_s * SR))
        i0 = int(round((p.nominal_start_s - p.offset_s) * SR))
        seg = np.zeros(b - a)
        # A weak fit can put the recording's t=0 after the slot's nominal start,
        # which leaves only `b - a - head` slots to fill; taking a full slot's
        # worth would then fail to broadcast and kill the whole build.
        head = min(max(-i0, 0), b - a)
        take = y[max(i0, 0): max(i0, 0) + (b - a) - head]
        seg[head: head + len(take)] = take
        w = np.ones(len(seg))
        k = min(fade, len(seg) // 2)
        if k > 0:
            ramp = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
            w[:k], w[-k:] = ramp, ramp[::-1]
        a, b = max(a, 0), min(b, n)
        out[a:b] += seg[: b - a] * w[: b - a]
        wsum[a:b] += w[: b - a]
    return out / np.maximum(wsum, 1e-9)


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf
    peak = np.abs(y).max()
    if peak > 1.0:
        y = y / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)


# ---------------------------------------------------------------- the verdict

def verify_mix(mix: int, y: np.ndarray, shift_s: float = 60.0) -> dict:
    """Score the reconstruction against the published series, with a null.

    Every column is correlated frame-by-frame with the record's own, and every
    correlation is stated beside the same correlation under a 60 s circular
    shift. The shifted number is what agreement looks like when the audio is
    merely the right kind of music rather than the right audio at the right
    time; without it a 0.8 on a bass fraction means nothing.
    """
    ref = load_feature_series(mix)
    got = frame_features(y)
    n = min(len(ref["t_s"]), len(got["rms_dbfs"]))
    shift = int(shift_s * FPS)
    cols = {}
    for c, v in got.items():
        a, b = _norm(ref[c][:n]), _norm(v[:n])
        cols[c] = {
            "r": float(a @ b),
            "r_shift_null": float(a @ np.roll(b, shift)),
        }
        cols[c]["margin"] = cols[c]["r"] - cols[c]["r_shift_null"]
    return {"mix": mix, "n_frames": int(n), "columns": cols,
            "gate_pass": bool(cols["onset_strength"]["r"] > 0.5
                              and cols["bass_frac_below150hz"]["r"] > 0.7
                              and cols["rms_dbfs"]["r"] > 0.7)}


def per_track_verify(mix: int, y: np.ndarray) -> list[dict]:
    """The same score inside each track's slot, so a bad seam is visible."""
    ref = load_feature_series(mix)
    got = frame_features(y)
    rows = [r for r in load_stimulus_tracks() if r["dj_mix"] == mix]
    out = []
    for r in rows:
        lo, hi = int(r["start_s"] * FPS), int(r["end_s"] * FPS)
        hi = min(hi, len(ref["t_s"]), len(got["rms_dbfs"]))
        d = {"title": r["title"], "track_order": r["track_order"],
             "start_s": r["start_s"], "end_s": r["end_s"]}
        for c in ("onset_strength", "rms_dbfs", "bass_frac_below150hz",
                  "spectral_centroid_hz"):
            d[c] = float(_norm(ref[c][lo:hi]) @ _norm(got[c][lo:hi]))
        out.append(d)
    return out


# ------------------------------------------------------------------- sourcing

def resolve_stimulus(timeout: float = 20.0) -> list[dict]:
    """Ask Deezer's public catalogue which master each named recording is.

    Metadata only, no account and no audio: it returns the ISRC, the exact
    title version and the duration, which is what tells you whether the file you
    bought is the extended mix the study played or the radio edit.
    """
    import urllib.parse, urllib.request
    out = []
    for title, meta in STIMULUS.items():
        url = ("https://api.deezer.com/search?limit=10&q="
               + urllib.parse.quote(meta["query"]))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.load(r).get("data", [])
        except Exception as e:                       # offline is not fatal here
            out.append({"title": title, **meta, "error": f"{type(e).__name__}: {e}"})
            continue
        want = meta.get("version", "").lower()
        hits = [h for h in data
                if h["artist"]["name"].lower().startswith(meta["artist"].split()[0].lower())]
        if want:
            pref = [h for h in hits if want.replace(" mix", "") in h["title"].lower()]
            hits = pref or hits
        out.append({
            "title": title, "artist": meta["artist"], "label": meta["label"],
            "year": meta["year"], "wanted_version": meta.get("version", ""),
            "candidates": [{"deezer_id": h["id"], "title": h["title"],
                            "artist": h["artist"]["name"],
                            "album": h["album"]["title"],
                            "isrc": h.get("isrc", ""),
                            "duration_s": h["duration"]} for h in hits[:5]],
        })
    return out


def ingest_dir(src_dir: Path = SOURCES) -> dict[str, Path]:
    """Match the audio files present to the four named recordings.

    Filename matching against the alias list, because a store's naming is not
    predictable and there is no point being clever about four files. Anything
    unmatched is reported rather than guessed at.
    """
    exts = {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg", ".opus"}
    found: dict[str, Path] = {}
    unmatched: list[Path] = []
    for p in sorted(src_dir.glob("*")):
        if p.suffix.lower() not in exts:
            continue
        stem = re.sub(r"[^a-z0-9]+", " ", p.stem.lower())
        hit = None
        for title, meta in STIMULUS.items():
            if any(a in stem for a in meta["aliases"]):
                hit = title
                break
        if hit and hit not in found:
            found[hit] = p
        elif hit is None:
            unmatched.append(p)
    ingest_dir.unmatched = unmatched            # type: ignore[attr-defined]
    return found


def placement_table(placements: list[Placement]) -> list[dict]:
    return [asdict(p) for p in placements]
