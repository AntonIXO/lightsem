"""AIST++ / AIST Dance DB loading, NaN policy, and manifest generation.

Filename schema
---------------
``gLH_sBM_cAll_d16_mLH1_ch05.pkl``
  = genre / setting / camera / dancer / music-id / choreography-id

The camera field can be ``cAll`` as well as ``c01``, so the regex must accept
word characters, not just digits. Getting this wrong silently drops sequences.

NaN policy (measured on the real corpus, see docs/DATA_NOTES.md)
----------------------------------------------------------------
Each pickle holds TWO arrays and they differ in quality:

  * ``keypoints3d``        -- 19.3% of sampled files contain NaN frames
                              (mean bad-frame fraction 0.0071)
  * ``keypoints3d_optim``  -- ZERO NaN across the whole sample

This loader defaults to ``keypoints3d_optim``, so in the default path the
interpolation below NEVER FIRES. It is retained because it is genuinely
required for ``keypoints3d``, and as a cheap guard. Do not cite "NaN
handling" as if it were doing work in the default pipeline.

Policy when NaN is present:
  * a *bad frame* is any frame where any joint coordinate is NaN
  * if bad_frac > ``max_nan_frac`` (default 0.10) the sequence is REJECTED
  * otherwise each (joint, coord) series is linearly interpolated over the
    bad frames, with edge extension at the boundaries

Duration policy
---------------
Pose clips are SHORT (median ~8.7 s) while audio tracks run 30-51 s. Any
audio analysis must be clipped to the sequence's own duration. Ignoring this
once produced n=0 usable pairs after a ">=20 beats" filter — a silent, total
data loss. ``SequenceRecord.duration_s`` is the authority.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import KP_FPS

# Accepts cAll as well as c01/c02...
NAME_RE = re.compile(r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)")

DEFAULT_ROOT = Path("/root/lightsem/data")


@dataclass
class SequenceMeta:
    """One AIST++ keypoint sequence's identity and quality record."""

    name: str
    genre: str
    setting: str
    camera: str
    dancer: str
    music_id: str
    choreo_id: str
    n_frames: int
    duration_s: float
    nan_frac: float
    usable: bool
    reason: str
    wav_path: str | None = None


def parse_name(stem: str) -> dict[str, str] | None:
    """Parse an AIST++ filename stem into its component fields."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    genre, setting, camera, dancer, music_id, choreo_id = m.groups()
    return {
        "genre": genre,
        "setting": setting,
        "camera": camera,
        "dancer": dancer,
        "music_id": music_id,
        "choreo_id": choreo_id,
    }


def interpolate_nans(kp: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs per joint per coordinate.

    Parameters
    ----------
    kp : (T, J, 3) array possibly containing NaN.

    Returns
    -------
    (T, J, 3) array with NaNs filled. Edges are extended with the nearest
    finite value (``np.interp`` default behaviour).
    """
    out = kp.copy()
    T = out.shape[0]
    t = np.arange(T)
    for j in range(out.shape[1]):
        for c in range(out.shape[2]):
            series = out[:, j, c]
            good = np.isfinite(series)
            if good.all():
                continue
            if good.sum() < 2:
                # Not enough anchors to interpolate; zero-fill and let the
                # nan_frac gate reject the sequence.
                out[:, j, c] = 0.0
                continue
            out[:, j, c] = np.interp(t, t[good], series[good])
    return out


def load_keypoints(path: str | Path, key: str = "keypoints3d_optim") -> np.ndarray:
    """Load a keypoints pickle, preferring the optimised array when present."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        if key in d:
            return np.asarray(d[key], dtype=np.float64)
        for fallback in ("keypoints3d", "keypoints3d_optim"):
            if fallback in d:
                return np.asarray(d[fallback], dtype=np.float64)
        raise KeyError(f"no keypoint array in {path}: keys={list(d)}")
    return np.asarray(d, dtype=np.float64)


def root_relative(kp: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Subtract the root joint so global translation cannot dominate."""
    return kp - kp[:, root_joint : root_joint + 1, :]


def kinetic_energy(kp: np.ndarray, fps: float = KP_FPS) -> np.ndarray:
    """KE(t) = sum_j ||dP_j/dt||^2 on root-relative keypoints.

    Retained for provenance/reproduction of the probe phase only. Per the
    recorded rule-out this is NOT a viable dense DMX supervision signal.
    """
    rel = root_relative(kp)
    vel = np.diff(rel, axis=0) * fps
    return (vel**2).sum(axis=(1, 2))


class AistDataset:
    """Index over the on-disk AIST++ keypoints and paired AIST audio."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        max_nan_frac: float = 0.10,
        fps: float = KP_FPS,
    ) -> None:
        self.root = Path(root)
        self.kp_dir = self.root / "kp3d" / "keypoints3d"
        self.wav_dir = self.root / "wav" / "wav"
        self.max_nan_frac = max_nan_frac
        self.fps = fps

    def keypoint_files(self) -> list[Path]:
        return sorted(self.kp_dir.glob("*.pkl"))

    def wav_for(self, music_id: str) -> Path | None:
        p = self.wav_dir / f"{music_id}.wav"
        return p if p.exists() else None

    def inspect(self, path: str | Path) -> SequenceMeta:
        """Parse + quality-check one sequence WITHOUT keeping the array."""
        path = Path(path)
        fields = parse_name(path.stem)
        if fields is None:
            return SequenceMeta(
                name=path.stem, genre="", setting="", camera="", dancer="",
                music_id="", choreo_id="", n_frames=0, duration_s=0.0,
                nan_frac=1.0, usable=False, reason="unparseable_filename",
            )
        try:
            kp = load_keypoints(path)
        except Exception as e:  # noqa: BLE001 - record, don't crash the scan
            return SequenceMeta(
                name=path.stem, **fields, n_frames=0, duration_s=0.0,
                nan_frac=1.0, usable=False, reason=f"load_error:{type(e).__name__}",
            )

        n_frames = int(kp.shape[0])
        bad = ~np.isfinite(kp).all(axis=(1, 2))
        nan_frac = float(bad.mean()) if n_frames else 1.0
        duration_s = n_frames / self.fps

        wav = self.wav_for(fields["music_id"])
        usable, reason = True, "ok"
        if n_frames < 2:
            usable, reason = False, "too_short"
        elif nan_frac > self.max_nan_frac:
            usable, reason = False, f"nan_frac>{self.max_nan_frac}"
        elif wav is None:
            usable, reason = False, "missing_audio"

        return SequenceMeta(
            name=path.stem, **fields, n_frames=n_frames, duration_s=duration_s,
            nan_frac=nan_frac, usable=usable, reason=reason,
            wav_path=str(wav) if wav else None,
        )

    def load(self, path: str | Path) -> tuple[np.ndarray, SequenceMeta]:
        """Load one sequence with the NaN policy applied."""
        meta = self.inspect(path)
        kp = load_keypoints(path)
        if not np.isfinite(kp).all():
            kp = interpolate_nans(kp)
        return kp, meta

    def build_manifest(self, limit: int | None = None) -> list[SequenceMeta]:
        files = self.keypoint_files()
        if limit is not None:
            files = files[:limit]
        return [self.inspect(p) for p in files]


def write_manifest(
    metas: list[SequenceMeta],
    out_dir: str | Path = "/root/lightsem/data/cache",
    stem: str = "manifest",
) -> dict[str, Path]:
    """Persist the manifest as parquet (+ JSON summary)."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(m) for m in metas])
    pq = out_dir / f"{stem}.parquet"
    df.to_parquet(pq, index=False)

    usable = df[df.usable]
    summary = {
        "n_total": int(len(df)),
        "n_usable": int(len(usable)),
        "n_rejected": int((~df.usable).sum()),
        "reject_reasons": df.loc[~df.usable, "reason"].value_counts().to_dict(),
        "n_music_ids": int(df.music_id.nunique()),
        "n_dancers": int(df.dancer.nunique()),
        "n_choreo": int(df.choreo_id.nunique()),
        "duration_s": {
            "min": float(usable.duration_s.min()) if len(usable) else None,
            "median": float(usable.duration_s.median()) if len(usable) else None,
            "max": float(usable.duration_s.max()) if len(usable) else None,
        },
        "nan_frac": {
            "mean": float(df.nan_frac.mean()),
            "max": float(df.nan_frac.max()),
        },
    }
    js = out_dir / f"{stem}_summary.json"
    js.write_text(json.dumps(summary, indent=2))
    return {"parquet": pq, "summary": js}
