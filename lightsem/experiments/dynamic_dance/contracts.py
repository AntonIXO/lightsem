"""Validated, timestamp-explicit artifacts for the dynamic dance campaign.

Music observations are measurements/estimates, never calibrated activity claims.
Serialization uses arrays plus authenticated JSON; pickle is never required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA_VERSION = 1
DIAGNOSTIC_PREFIX = "diagnostic__"
SOURCE_NAMES = ("drums", "bass", "other", "vocals")
SOURCE_FIELDS = ("rms", "onset", "flux", "sustain")
MIXTURE_FIELDS = ("rms", "onset", "centroid", "flux")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _finite(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != len(shape) or any(n is not None and value.shape[i] != n for i, n in enumerate(shape)):
        raise ValueError(f"{name}: expected {shape}, got {value.shape}")
    if value.dtype.kind not in "fibu" or not np.isfinite(value).all():
        raise ValueError(f"{name}: finite numeric values required")
    return value


def _time(name: str, value: np.ndarray, *, empty: bool = False) -> np.ndarray:
    value = _finite(name, value, (None,))
    if (not empty and len(value) < 1) or np.any(np.diff(value) <= 0) or np.any(value < 0):
        raise ValueError(f"{name}: nonnegative strictly increasing timestamps required")
    return value


def _retiming_clock(time: np.ndarray, source_time: np.ndarray, metadata: dict) -> None:
    """Apply clock constraints only to explicitly classified pure retiming.

    D4's recorded fragment jumps are not playback-rate measurements.
    """
    if not metadata.get("source_order_preserved", False):
        return
    clock = np.asarray(source_time)
    if clock.ndim == 2 and clock.shape[1] != 5:
        raise ValueError("Pure regional retiming requires all five source clocks")
    clock = clock[:, None] if clock.ndim == 1 else clock
    if len(time) < 2:
        raise ValueError("Pure retiming requires at least two timestamps")
    source_start = float(metadata.get("source_start_s", 0.))
    source_end = float(metadata.get("source_end_s", time[-1] - time[0]))
    if not np.isfinite([source_start, source_end]).all() or source_start != 0 or source_end <= source_start:
        raise ValueError("Pure retiming source interval must begin at zero and have positive duration")
    precision = np.finfo(clock.dtype).eps if clock.dtype.kind == "f" else 0.
    tolerance = max(1e-7, 4 * precision * max(1., abs(source_end)))
    if np.any(np.abs(clock[0] - source_start) > tolerance) or np.any(np.abs(clock[-1] - source_end) > tolerance):
        raise ValueError("Pure retiming must preserve exact source endpoints")
    steps = np.diff(clock, axis=0)
    if np.any(steps < -tolerance) or np.any(steps > 1.5 * np.diff(time)[:, None] + tolerance):
        raise ValueError("Pure retiming source clocks must be monotone with playback rate in [0,1.5]")


def _json_safe(value: Any) -> None:
    json.dumps(value, allow_nan=False)


def _diagnostic_arrays(value: Any) -> dict[str, np.ndarray]:
    """Validate optional numeric audit arrays and reserve their NPZ namespace."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("diagnostics must be a mapping of names to arrays")
    result = {}
    for name, array in value.items():
        if not isinstance(name, str) or not name or name.startswith(DIAGNOSTIC_PREFIX):
            raise ValueError("diagnostic names must be nonempty and omit the reserved prefix")
        arr = np.asarray(array)
        if arr.dtype.kind not in "fibu" or not np.isfinite(arr).all():
            raise ValueError(f"diagnostic array {name!r} must contain finite numeric values")
        result[name] = arr
    return result


def _split_diagnostics(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    diagnostics = {key[len(DIAGNOSTIC_PREFIX):]: value for key, value in arrays.items()
                   if key.startswith(DIAGNOSTIC_PREFIX)}
    primary = {key: value for key, value in arrays.items() if not key.startswith(DIAGNOSTIC_PREFIX)}
    return primary, diagnostics


def _save(path: str | Path, kind: str, arrays: dict[str, np.ndarray], details: dict) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Artifact path must end in .npz")
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        raise FileExistsError(f"Refuse to overwrite artifact: {path}")
    _json_safe(details)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        metadata = {"schema_version": SCHEMA_VERSION, "kind": kind, "npz_sha256": sha256(path), **details}
        meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        meta_path.with_suffix(".json.sha256.json").write_text(json.dumps({"sha256": sha256(meta_path)}) + "\n")
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: str | Path, kind: str) -> tuple[dict, dict]:
    path = Path(path)
    meta_path = path.with_suffix(".json")
    digest_path = meta_path.with_suffix(".json.sha256.json")
    if sha256(meta_path) != json.loads(digest_path.read_text())["sha256"]:
        raise ValueError("Artifact metadata changed")
    details = json.loads(meta_path.read_text())
    if details["schema_version"] != SCHEMA_VERSION or details["kind"] != kind:
        raise ValueError("Unsupported artifact schema/kind")
    if sha256(path) != details["npz_sha256"]:
        raise ValueError("Artifact arrays changed")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, details


@dataclass
class MusicTimeline:
    time: np.ndarray
    muq: np.ndarray
    muq_time: np.ndarray
    sources: np.ndarray
    mixture: np.ndarray
    beats: np.ndarray
    downbeats: np.ndarray
    sections: list[dict]
    recurrence: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self) -> "MusicTimeline":
        t = _time("time", self.time)
        u = _time("muq_time", self.muq_time)
        _finite("muq", self.muq, (len(u), 1024))
        _finite("sources", self.sources, (len(t), 4, 4))
        _finite("mixture", self.mixture, (len(t), 4))
        if np.any(self.sources < 0) or np.any(self.mixture < 0):
            raise ValueError("Physical music observations must be nonnegative")
        _time("beats", self.beats, empty=True)
        _time("downbeats", self.downbeats, empty=True)
        _finite("recurrence", self.recurrence, (len(self.sections), len(self.sections)))
        if not np.allclose(self.recurrence, np.asarray(self.recurrence).T, atol=1e-6):
            raise ValueError("Recurrence must be symmetric")
        previous_end = 0.0
        for section in self.sections:
            start, end = float(section["start"]), float(section["end"])
            if not np.isfinite([start, end]).all() or start < previous_end - 1e-6 or end <= start:
                raise ValueError("Sections must be ordered non-overlapping positive intervals")
            if not isinstance(section["label"], str):
                raise ValueError("Section label must be a string")
            confidence = section.get("confidence")
            if confidence is not None and (not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Section confidence must be null or in [0,1]")
            previous_end = end
        if "duration_s" in self.metadata:
            duration = float(self.metadata["duration_s"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid waveform duration_s")
            observation_hop = 1 / float(self.metadata.get("recipe", {}).get("observation_hz", 30))
            native_hop = 1 / float(self.metadata.get("recipe", {}).get("muq_native_hz", 25))
            tolerance = max(observation_hop, native_hop) + 1e-6
            for name, timestamps in (("time", t), ("muq_time", u), ("beats", self.beats), ("downbeats", self.downbeats)):
                if len(timestamps) and timestamps[-1] > duration + tolerance:
                    raise ValueError(f"{name} exceeds analyzed waveform duration")
            if t[0] > observation_hop + 1e-6 or t[-1] < duration - observation_hop - 1e-6:
                raise ValueError("Physical observations do not cover the complete analyzed waveform")
            if u[0] > native_hop + 1e-6 or u[-1] < duration - native_hop - 1e-6:
                raise ValueError("MuQ does not cover the complete analyzed waveform")
            for section in self.sections:
                if section["end"] > duration + tolerance:
                    raise ValueError("Section exceeds analyzed waveform duration")
        _json_safe({"sections": self.sections, "metadata": self.metadata})
        return self

    def save(self, path: str | Path) -> None:
        self.validate()
        names = ("time", "muq", "muq_time", "sources", "mixture", "beats", "downbeats", "recurrence")
        _save(path, "MusicTimeline", {k: np.asarray(getattr(self, k)) for k in names},
              {"sections": self.sections, "metadata": self.metadata,
               "source_names": SOURCE_NAMES, "source_fields": SOURCE_FIELDS, "mixture_fields": MIXTURE_FIELDS})

    @classmethod
    def load(cls, path: str | Path) -> "MusicTimeline":
        arrays, details = _load(path, "MusicTimeline")
        if tuple(details["source_names"]) != SOURCE_NAMES or tuple(details["source_fields"]) != SOURCE_FIELDS:
            raise ValueError("Source schema changed")
        return cls(**arrays, sections=details["sections"], metadata=details["metadata"]).validate()


@dataclass
class DancePlan:
    time: np.ndarray
    source_time: np.ndarray
    effort: np.ndarray
    anchors: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> "DancePlan":
        t = _time("time", self.time)
        source = np.asarray(self.source_time)
        if source.ndim not in (1, 2):
            raise ValueError("source_time must be [T] or [T,R]")
        _finite("source_time", source, (len(t),) if source.ndim == 1 else (len(t), None))
        _finite("effort", self.effort, (len(t), None))
        if np.any(source < 0) or np.any(self.effort < 0):
            raise ValueError("Source times and effort must be nonnegative")
        _retiming_clock(t, source, self.metadata)
        _diagnostic_arrays(self.diagnostics)
        _json_safe({"anchors": self.anchors, "events": self.events, "metadata": self.metadata})
        return self

    def save(self, path: str | Path) -> None:
        self.validate()
        arrays = {k: np.asarray(getattr(self, k)) for k in ("time", "source_time", "effort")}
        arrays.update({DIAGNOSTIC_PREFIX + key: value for key, value in _diagnostic_arrays(self.diagnostics).items()})
        _save(path, "DancePlan", arrays,
              {"anchors": self.anchors, "events": self.events, "metadata": self.metadata})

    @classmethod
    def load(cls, path: str | Path) -> "DancePlan":
        arrays, details = _load(path, "DancePlan")
        arrays, diagnostics = _split_diagnostics(arrays)
        return cls(**arrays, **{k: details[k] for k in ("anchors", "events", "metadata")},
                   diagnostics=diagnostics).validate()


@dataclass
class DanceResult:
    time: np.ndarray
    world: np.ndarray
    contacts: np.ndarray
    source_time: np.ndarray | None = None
    native: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> "DanceResult":
        t = _time("time", self.time)
        _finite("world", self.world, (len(t), 22, 3))
        _finite("contacts", self.contacts, (len(t), 4))
        if np.any(np.asarray(self.contacts) < 0) or np.any(np.asarray(self.contacts) > 1):
            raise ValueError("Contact observations must lie in [0,1]")
        if self.source_time is not None:
            s = np.asarray(self.source_time)
            if s.ndim not in (1, 2):
                raise ValueError("source_time must be [T] or [T,R]")
            _finite("source_time", s, (len(t),) if s.ndim == 1 else (len(t), None))
            if np.any(s < 0):
                raise ValueError("Negative source timestamps")
            _retiming_clock(t, s, self.metadata)
        if self.source_time is None and self.metadata.get("source_order_preserved", False):
            raise ValueError("Pure retiming requires its actual source clock")
        if self.native is not None:
            _finite("native", self.native, (len(t), 139))
        _diagnostic_arrays(self.diagnostics)
        _json_safe(self.metadata)
        return self

    def save(self, path: str | Path) -> None:
        self.validate()
        names = ["time", "world", "contacts"] + [k for k in ("source_time", "native") if getattr(self, k) is not None]
        arrays = {k: np.asarray(getattr(self, k)) for k in names}
        arrays.update({DIAGNOSTIC_PREFIX + key: value for key, value in _diagnostic_arrays(self.diagnostics).items()})
        _save(path, "DanceResult", arrays, {"metadata": self.metadata})

    @classmethod
    def load(cls, path: str | Path) -> "DanceResult":
        arrays, details = _load(path, "DanceResult")
        arrays, diagnostics = _split_diagnostics(arrays)
        return cls(**arrays, metadata=details["metadata"], diagnostics=diagnostics).validate()
