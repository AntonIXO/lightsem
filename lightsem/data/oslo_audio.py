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
    native_bpm: float         # the recording's own tempo implied by the fit
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
    """Play `env` at `ratio` times speed; length shrinks as the ratio grows."""
    n_out = max(int(round(len(env) / ratio)), 2)
    return np.interp(np.arange(n_out) * ratio, np.arange(len(env)), env)


def _structure_env(env: np.ndarray, win_s: float = 2.0) -> np.ndarray:
    """The arrangement contour: onset energy smoothed past the bar.

    A box filter wider than a bar erases the pulse and leaves what the
    arrangement does -- the kick entering, the breakdown emptying out, the drop.
    That contour is what makes a position in a track unique.
    """
    from scipy.ndimage import uniform_filter1d
    n = max(int(win_s * FPS), 3)
    return uniform_filter1d(np.asarray(env, dtype=np.float64), n, mode="nearest")


def fit_placement(ref_env: np.ndarray, cand_env: np.ndarray,
                  ratio_lo: float = 0.90, ratio_hi: float = 1.10,
                  coarse: int = 81, refine: int = 41,
                  refine_win_s: float = 0.8,
                  min_overlap_s: float = 45.0) -> tuple[float, int, float]:
    """Best (ratio, lag in frames, correlation) of a recording against a mix.

    Two stages, because one does not work. Correlating raw onset envelopes finds
    the tempo ratio exactly and then puts the track a whole number of bars out:
    every bar of four-on-the-floor looks like every other, so the peak it lands
    on is an alias, and a bar of error is ~2 s of misalignment against mocap.
    The coarse position therefore comes from the arrangement contour, which is
    unique within a track, and only then is the lag refined on the raw onset
    envelope, inside a window narrower than a bar so it cannot alias back out.

    The ratio is fitted rather than read off `stimulus_tracks.tsv` because those
    BPMs are integers: 124 against a true 123.6 is 0.3 %, a beat of drift per
    minute and tens of frames by the end of a track.
    """
    ref_s, cand_s = _structure_env(ref_env), _structure_env(cand_env)
    best = (1.0, 0, -2.0)
    for r in np.linspace(ratio_lo, ratio_hi, coarse):
        c, lags = _ncc(ref_s, _resample_env(cand_s, r), min_overlap_s)
        i = int(np.argmax(c))
        if c[i] > best[2]:
            best = (float(r), int(lags[i]), float(c[i]))

    step = (ratio_hi - ratio_lo) / (coarse - 1)
    win = int(refine_win_s * FPS)
    out = (best[0], best[1], -2.0)
    for r in np.linspace(best[0] - step, best[0] + step, refine):
        c, lags = _ncc(ref_env, _resample_env(cand_env, r), min_overlap_s)
        sel = np.flatnonzero(np.abs(lags - best[1]) <= win)
        if not len(sel):
            continue
        i = sel[int(np.argmax(c[sel]))]
        if c[i] > out[2]:
            out = (float(r), int(lags[i]), float(c[i]))
    return out if out[2] > -2.0 else best


def null_correlation(ref_env: np.ndarray, cand_env: np.ndarray, ratio: float,
                     lag: int, guard_s: float = 30.0,
                     min_overlap_s: float = 45.0) -> float:
    """The best this recording scores anywhere the answer is not.

    A high correlation means nothing on its own: an EDM onset envelope is a
    four-on-the-floor pulse and will correlate with any other one at some lag.
    So the peak is priced against the best peak outside a guard band around it,
    and it is the margin, not the peak, that says the placement is real.
    """
    c, lags = _ncc(_structure_env(ref_env),
                   _structure_env(_resample_env(cand_env, ratio)), min_overlap_s)
    mask = np.abs(lags - lag) > int(guard_s * FPS)
    return float(c[mask].max()) if mask.any() else float("nan")


def place_track(row: dict, src: Path, ref_env: np.ndarray,
                pad_s: float = 25.0) -> Placement:
    """Fit one recording into its slot in one mix."""
    cand = onset_envelope(read_audio(src))
    lo = max(0, int((row["start_s"] - pad_s) * FPS))
    hi = min(len(ref_env), int((row["end_s"] + pad_s) * FPS))
    seg = ref_env[lo:hi]
    overlap_s = min(0.6 * (hi - lo) / FPS, 45.0)
    ratio, lag, corr = fit_placement(seg, cand, min_overlap_s=overlap_s)
    null = null_correlation(seg, cand, ratio, lag, min_overlap_s=overlap_s)
    offset_s = (lo + lag) / FPS
    # `bpm` in stimulus_tracks.tsv is the tempo the DJ played, not the record's
    # own: Ladykiller is logged 124 in mix 1 and 127 in mix 2. So the fitted
    # ratio divides out to the recording's native tempo. Released house is
    # authored on an integer grid, so when that lands on an integer the integer
    # times the ratio gives the played tempo past the log's rounding -- and when
    # it does not, say nothing rather than quote the nominal back as a "fit".
    native = row["bpm"] / ratio
    k = round(native)
    fitted = k * ratio if abs(native - k) < 0.25 else float("nan")
    return Placement(
        title=row["title"], mix=row["dj_mix"], track_order=row["track_order"],
        nominal_start_s=row["start_s"], nominal_end_s=row["end_s"],
        nominal_bpm=row["bpm"], source_file=str(src),
        ratio=ratio, offset_s=offset_s, corr=corr, null_corr=null,
        margin=corr - null, native_bpm=native, fitted_bpm=fitted,
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
