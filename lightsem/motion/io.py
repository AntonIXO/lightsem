"""Persistence for descriptor and audio-bus caches (npz + json sidecar)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..audio.online import AudioBus
from .descriptors import MotionDescriptors

CACHE_ROOT = Path("/mnt/lightsem-data/cache")


def save_descriptors(d: MotionDescriptors, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"ch__{k}": v.astype(np.float32) for k, v in d.channels.items()}
    payload.update({f"ev__{k}": v.astype(np.float64) for k, v in d.events.items()})
    payload.update({f"es__{k}": v.astype(np.float32) for k, v in d.event_strength.items()})
    payload["times"] = d.times.astype(np.float32)
    np.savez_compressed(path, meta=json.dumps({"fps": d.fps, **d.meta}), **payload)
    return path


def load_descriptors(path: str | Path) -> MotionDescriptors:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    fps = meta.pop("fps")
    ch = {k[4:]: z[k].astype(np.float64) for k in z.files if k.startswith("ch__")}
    ev = {k[4:]: z[k] for k in z.files if k.startswith("ev__")}
    es = {k[4:]: z[k] for k in z.files if k.startswith("es__")}
    return MotionDescriptors(fps=float(fps), times=z["times"].astype(np.float64),
                             channels=ch, events=ev, event_strength=es, meta=meta)


def save_bus(b: AudioBus, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"ch__{k}": v.astype(np.float32) for k, v in b.channels.items()}
    np.savez_compressed(
        path, times=b.times.astype(np.float32), mel=b.mel.astype(np.float16),
        beats=b.beats.astype(np.float64), downbeats=b.downbeats.astype(np.float64),
        meta=json.dumps({"fps": b.fps, "tempo": b.tempo, "duration_s": b.duration_s,
                         "flags": b.flags, **b.meta}),
        **payload)
    return path


def load_bus(path: str | Path) -> AudioBus:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    fps, tempo, dur = meta.pop("fps"), meta.pop("tempo"), meta.pop("duration_s")
    flags = meta.pop("flags", {})
    ch = {k[4:]: z[k].astype(np.float64) for k in z.files if k.startswith("ch__")}
    return AudioBus(fps=float(fps), times=z["times"].astype(np.float64), channels=ch,
                    mel=z["mel"].astype(np.float32), beats=z["beats"],
                    downbeats=z["downbeats"], tempo=float(tempo),
                    duration_s=float(dur), flags=flags, meta=meta)


def slice_bus(b: AudioBus, duration_s: float, offset_s: float = 0.0) -> AudioBus:
    """Window a cached track bus onto one dance clip's span."""
    i0 = int(round(offset_s * b.fps))
    i1 = min(int(round((offset_s + duration_s) * b.fps)), len(b.times))
    i0 = max(min(i0, i1 - 1), 0)
    ch = {k: v[i0:i1] for k, v in b.channels.items()}
    beats = b.beats[(b.beats >= offset_s) & (b.beats < offset_s + duration_s)] - offset_s
    downs = b.downbeats[(b.downbeats >= offset_s) & (b.downbeats < offset_s + duration_s)] - offset_s
    return AudioBus(fps=b.fps, times=b.times[i0:i1] - b.times[i0], channels=ch,
                    mel=b.mel[:, i0:i1], beats=beats, downbeats=downs, tempo=b.tempo,
                    duration_s=(i1 - i0) / b.fps, flags=dict(b.flags),
                    meta={**b.meta, "offset_s": offset_s})
