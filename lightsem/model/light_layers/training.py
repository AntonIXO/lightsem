"""Bounded, opt-in training for the preregistered light-layer campaign.

No historical results, cached training tensors, or TEST targets are read. Source
NPZ files stay immutable. The manifest names every input and its SHA256. The
``extractor_v1`` adapter reads separate sam2/box outputs, preserves actual source
PTS for novelty, and uses the requested global10Hz grid for audio supervision.
Unannotated cuts are allowed only in explicitly unqualified smoke or the
separately preregistered ``--observer-proxy`` scope. Observer scope retains the
original31/40=.775 coverage failure and makes no clean-lighting truth claim.
Full fits require a separate preregistration binding the exact implementation,
manifest, split, configuration, frozen normalization, and palette vocabulary.

Manifest schema (version 1)::

    {"schema_version": 1, "campaign_id": "...", "campaign_root": "/...",
     "split_manifest": {"path": "/...json", "sha256": "..."},
     "records": [{"recording_id": "...", "canonical_id": "...",
       "partition": "train", "audio": {"path": "...npz", "sha256": "..."},
       "rhythm": {"path": "...npz", "sha256": "..."},
       "visual": {"path": "...npz", "sha256": "..."}, "visual_prefix": "sam2",
       "legacy_visual": {"path": "...npz", "sha256": "..."},
       "coaches": [{"start_s": 12.0, "motion": {"path": "...", "sha256": "..."},
         "validity": {"path": "...npz", "sha256": "..."}}]}]}

Split source is the immutable Stage11 split_manifest.json with a ``recordings``
mapping. Only train/val (or dev) records may be listed. Visual NPZ times_s are
absolute original-video timestamps at 10 Hz. Keys are ``{prefix}_rich``,
``{prefix}_hs_histogram``, ``{prefix}_color_valid``,
``{prefix}_retained_fraction``, ``{prefix}_chromatic_fraction``,
``{prefix}_value_mean``, ``{prefix}_value_valid``, and ``cuts`` (bool).
``visual_keys`` can explicitly map these semantic names to extractor keys.
Pose validity NPZ has absolute ``times_s`` and boolean ``valid`` (T,4), in
POSE_NAMES order. Missing gap/cut/tracking validity fails closed; filled motion
samples are not automatically valid. Audio is full-record 40 Hz Stage11 bus;
rhythm NPZ is full-record 50 Hz BeatThis logits. No disk tensor cache is made.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .model import LightLayersConfig, build_light_layers, count_trainable_parameters
from .targets import chromatic_novelty, chromatic_event_targets

# Ordered exactly as the immutable source bus; asserted against project schema
# on load, rather than importing a legacy data builder with its own split.
AUDIO_SCALARS = (
    "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid",
    "flux_hf", "energy_lf", "energy_lmid", "energy_mid", "energy_hmid", "energy_hf",
    "onset", "onset_ema", "flux_ema", "flux_lf_ema", "flux_hf_ema", "centroid",
    "flatness", "rolloff", "percussive_ratio", "pulse_clarity", "beat_phase_sin",
    "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s")
POSE_NAMES = ("vert_bounce_amp", "hand_spread", "qom", "head_speed")
RHYTHM_COLUMNS = tuple(64 + AUDIO_SCALARS.index(n) for n in (
    "pulse_clarity", "beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase",
    "beat_period_s")) + (92, 93)
CENTER10 = slice(280, 360)
CENTER40 = slice(1120, 1440)
GIB = 2**30
RECIPES = {
    "S_none_P0": ("shared", 0, 0), "S_V6_P0": ("shared", 6, 0),
    "S_V6_P1": ("shared", 6, 4), "S_VR_P0": ("shared", 96, 0),
    "S_VR_P1": ("shared", 96, 4), "I_VR_P0": ("independent", 96, 0),
    "S_VR_Pshuf": ("shared", 96, 4), "S_VR_P0_Ashuf": ("shared", 96, 0),
}


OBSERVER_STUDY_KIND = "historical_observer_proxy_v1"
OBSERVER_PROTOCOL_ID = "observer-proxy-20260922-v1"
OBSERVER_UNKNOWN_CUTS_POLICY = "included_in_observer_target"
OBSERVER_RECIPES = ("S_none_P0", "S_V6_P0", "S_VR_P0", "I_VR_P0", "S_VR_P0_Ashuf")
OBSERVER_SCOPE_BINDINGS = ("study_kind", "unknown_cuts_policy", "observer_protocol_id",
                           "observer_protocol_path", "observer_protocol_sha256",
                           "identity_exclusions_id", "identity_exclusions_path", "identity_exclusions_sha256",
                           "scope_sha256")


class ContractError(ValueError):
    """Blocked execution, not a scientific negative result."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def _finite(value, name):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ContractError(f"Nonfinite {name}")
    return value


def _times(value, rate, name, *, start=None):
    value = _finite(np.asarray(value, dtype=np.float64), name)
    if value.ndim != 1 or len(value) < 2 or not np.allclose(
            np.diff(value), 1 / rate, atol=3e-5, rtol=0):
        raise ContractError(f"{name} must be uniform {rate}Hz timestamps")
    if start is not None and abs(value[0] - start) > 3e-5:
        raise ContractError(f"{name} must start at {start}, not {value[0]}")
    return value


def sample_linear(times, values, query, *, max_gap_s=.2):
    """Interpolate only inside observed support; return explicit validity."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values)
    query = np.asarray(query, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ContractError("Interpolation needs strictly increasing source times")
    columns = values[:, None] if values.ndim == 1 else values
    if columns.ndim != 2 or len(columns) != len(times):
        raise ContractError("Interpolation values shape mismatch")
    upper = np.clip(np.searchsorted(times, query, side="right"), 1, len(times)-1)
    lower = upper - 1
    exact = np.abs(query - times[upper]) < 3e-5
    exact_lower = np.abs(query - times[lower]) < 3e-5
    fraction = np.clip((query-times[lower]) / (times[upper]-times[lower]), 0, 1)
    valid = (query >= times[0]-3e-5) & (query <= times[-1]+3e-5)
    valid &= (times[upper]-times[lower] <= max_gap_s+3e-5) | exact | exact_lower
    result = columns[lower] * (1-fraction[:, None]) + columns[upper] * fraction[:, None]
    result[~valid] = 0
    return (result[:, 0] if values.ndim == 1 else result).astype(np.float32), valid


def _nearest_valid(times, valid, query, tolerance=.051):
    times, query = np.asarray(times), np.asarray(query)
    right = np.clip(np.searchsorted(times, query), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    pick = np.where(np.abs(times[right]-query) < np.abs(times[left]-query), right, left)
    supported = np.abs(times[pick]-query) <= tolerance
    result = np.asarray(valid)[pick].copy()
    result &= supported.reshape((-1,) + (1,)*(result.ndim-1))
    return result


def intensity_novelty(times, values, valid, cuts):
    """Positive .3s pre/post V rise after recording-relative mean-gain removal.

    Normalization uses the complete *available extracted recording window*, not
    each sampled crop; scope is saved in artifacts. Multiplying recording gain
    leaves this proxy unchanged. No chromatic feature enters it. Scores are
    bounded by rise/(1+rise) so the frozen event primitive can be shared.
    """
    times, values = np.asarray(times), _finite(values, "value_mean")
    valid, cuts = np.asarray(valid, bool), np.asarray(cuts, bool)
    gain = float(np.mean(values[valid])) if valid.any() else 0.
    normalized = values / max(gain, 1e-6)
    scores, eligible = np.zeros(len(times)), np.zeros(len(times), bool)
    for i, center in enumerate(times):
        left, right = center-.3, center+.3
        if left < times[0]-1e-8 or right > times[-1]+1e-8:
            continue
        indices = np.arange(max(0, np.searchsorted(times, left, side="right")-1),
                            min(len(times)-1, np.searchsorted(times, right, side="left")))
        starts, ends = times[indices], times[indices+1]
        pre = np.maximum(0., np.minimum(ends, center)-np.maximum(starts, left))
        post = np.maximum(0., np.minimum(ends, right)-np.maximum(starts, center))
        active = pre+post > 1e-9
        boundaries = (times > left+1e-8) & (times < right-1e-8)
        if (active.any() and valid[indices[active]].all() and not cuts[boundaries].any()
                and np.all(ends[active]-starts[active] <= .2+1e-8)
                and abs(pre.sum()-.3) < 1e-8 and abs(post.sum()-.3) < 1e-8):
            rise = max(0., float(post @ normalized[indices]/post.sum()
                                - pre @ normalized[indices]/pre.sum()))
            scores[i], eligible[i] = rise/(1+rise), True
    return SimpleNamespace(times_s=times, scores=scores, eligible=eligible), gain


def event_labels(events, times):
    """Native 10Hz point events; tolerance belongs to later .2s proxy metrics."""
    out = np.zeros(len(times), np.float32)
    if len(events.event_indices):
        out[events.event_indices] = 1
    return out


def positive_weight(labels, mask):
    labels, mask = np.asarray(labels), np.asarray(mask, bool)
    if not mask.any():
        raise ContractError("Zero eligible event labels")
    positive = float(labels[mask].sum())
    # All-negative songs remain valid. A globally all-negative head is finite,
    # explicitly reported, and uses neutral weight rather than infinity.
    return float(min(50., (mask.sum()-positive)/positive)) if positive else 1.


@dataclass
class VisualRecord:
    recording_id: str
    canonical_id: str
    partition: str
    times: np.ndarray
    rich: np.ndarray
    hs: np.ndarray
    color_valid: np.ndarray
    retained: np.ndarray
    chromatic: np.ndarray
    values: np.ndarray
    value_valid: np.ndarray
    cuts: np.ndarray
    legacy: np.ndarray | None = None
    legacy_valid: np.ndarray | None = None
    pose: np.ndarray | None = None
    pose_valid: np.ndarray | None = None
    rich_valid: np.ndarray | None = None
    source_times: np.ndarray | None = None
    stored_novelty: object | None = None
    frame_valid: np.ndarray | None = None
    mask_activity_known: np.ndarray | None = None

    def validate(self):
        _times(self.times, 10, "visual times_s")
        if not np.allclose(self.times*10, np.rint(self.times*10), atol=3e-4, rtol=0):
            raise ContractError("Requested visual grid must align with full-record 10Hz pooling phase")
        n = len(self.times)
        for name, dimension in (("rich", 96), ("hs", 72)):
            value = _finite(getattr(self, name), name)
            if value.shape != (n, dimension):
                raise ContractError(f"Expected {name} shape {(n, dimension)}")
        for name in ("color_valid", "value_valid", "cuts"):
            value = getattr(self, name)
            if np.asarray(value).dtype != bool or np.shape(value) != (n,):
                raise ContractError(f"{name} must be boolean (T,)")
        for name in ("retained", "chromatic", "values"):
            value = _finite(getattr(self, name), name)
            if value.shape != (n,) or np.any((value < 0) | (value > 1)):
                raise ContractError(f"{name} must be fractions (T,)")
        if self.frame_valid is not None:
            if self.frame_valid.dtype != bool or self.frame_valid.shape != (n,):
                raise ContractError("frame_valid must be boolean (T,)")
            missing = ~self.frame_valid
            if (self.color_valid[missing].any() or self.value_valid[missing].any()
                    or np.any(self.hs[missing] != 0) or np.any(self.rich[missing] != 0)
                    or np.any(self.values[missing] != 0)):
                raise ContractError("Missing visual frames must be invalid zero placeholders, not imputed")
            if self.stored_novelty is not None and np.any(self.stored_novelty.eligible & missing):
                raise ContractError("Missing visual frame cannot be an eligible stored color event")
        if self.mask_activity_known is not None and (
                self.mask_activity_known.dtype != bool or self.mask_activity_known.shape != (n,)):
            raise ContractError("mask_activity_known must be boolean (T,)")
        if len(self.times) < 640:
            raise ContractError("A visual record needs >=640 frames (64 seconds)")

    def novelty(self):
        if self.stored_novelty is not None:
            return self.stored_novelty
        times = self.times if self.source_times is None else self.source_times
        return chromatic_novelty(times, self.hs, self.color_valid,
                                retained_fraction=self.retained,
                                chromatic_fraction=self.chromatic, cuts=self.cuts)


def _standardizer(rows):
    total = count = squares = None
    for values, valid in rows:
        values = np.asarray(values, np.float64)
        valid = np.asarray(valid, bool)
        if valid.ndim == 1:
            valid = np.broadcast_to(valid[:, None], values.shape)
        if valid.shape != values.shape:
            raise ContractError("Standardization validity shape mismatch")
        if not np.isfinite(values[valid]).all():
            raise ContractError("Nonfinite valid auxiliary coordinate")
        safe = np.where(valid, values, 0)
        if total is None:
            total, squares, count = safe.sum(0), (safe*safe).sum(0), valid.sum(0)
        else:
            total += safe.sum(0); squares += (safe*safe).sum(0); count += valid.sum(0)
    if count is None or np.any(count == 0):
        raise ContractError("Zero eligible TRAIN normalization coordinate")
    mean = total/count
    std = np.sqrt(np.maximum(squares/count-mean*mean, 0))
    return {"mean": mean.tolist(), "std": np.maximum(std, 1e-3).tolist(),
            "counts": count.tolist()}


def _rich_mask(record):
    frame_valid = (record.value_valid if record.frame_valid is None else record.frame_valid)
    if record.rich_valid is not None:
        return record.rich_valid & frame_valid[:, None] & ~record.cuts[:, None]
    # No per-quadrant guessed observation: its supplied chromatic fraction and
    # retained coverage determine validity of hue/saturation coordinates.
    mask = np.broadcast_to(record.color_valid[:, None], record.rich.shape).copy()
    for quadrant in range(4):
        first = 72+quadrant*6
        mask[:, first:first+4] &= (record.rich[:, first+4:first+5] > 0)
        mask[:, first+4:first+6] = record.value_valid[:, None]
    return mask & frame_valid[:, None] & ~record.cuts[:, None]


def fit_transforms(records, *, audio_rows=(), palette_size=32, max_samples=512, seed=0):
    """Fit TRAIN only; mixed partitions are rejected, never silently filtered."""
    records = list(records)
    if not records or any(r.partition != "train" for r in records):
        raise ContractError("fit_transforms accepts nonempty TRAIN-only records")
    if len({r.recording_id for r in records}) != len(records):
        raise ContractError("Fit records must be unique; coaches are not samples")
    records = sorted(records, key=lambda r: r.recording_id)
    novelty, intensity, histograms = [], [], []
    for record in records:
        record.validate()
        novelty.append(record.novelty())
        intensity.append(intensity_novelty(
            record.times if record.source_times is None else record.source_times,
            record.values, record.value_valid, record.cuts)[0])
        indices = np.flatnonzero(record.color_valid & ~record.cuts)
        if len(indices):
            choose = indices[np.linspace(0, len(indices)-1, min(len(indices), max_samples), dtype=int)]
            histograms.append(record.hs[choose])
    scores = [n.scores[n.eligible] for n in novelty if n.eligible.any()]
    accents = [n.scores[n.eligible] for n in intensity if n.eligible.any()]
    if not scores or not accents or not histograms:
        raise ContractError("Zero eligible TRAIN chromatic/intensity/palette target")
    color_threshold = float(np.quantile(np.concatenate(scores), .95))
    accent_threshold = float(np.quantile(np.concatenate(accents), .95))
    samples = np.concatenate(histograms)
    if len(samples) < palette_size:
        raise ContractError("Fewer valid TRAIN palette samples than vocabulary entries")
    from sklearn.cluster import MiniBatchKMeans
    vocabulary = MiniBatchKMeans(n_clusters=palette_size, random_state=seed, n_init=3,
                                batch_size=1024, max_iter=100, reassignment_ratio=0.)
    vocabulary.fit(samples)
    prototypes = vocabulary.cluster_centers_.astype(np.float32)
    prototypes /= np.maximum(prototypes.sum(1, keepdims=True), 1e-12)
    labels = {"color": [], "accent": []}; masks = {"color": [], "accent": []}
    for key, ns, threshold in (("color", novelty, color_threshold),
                               ("accent", intensity, accent_threshold)):
        for n in ns:
            labels[key].append(event_labels(chromatic_event_targets(n, threshold=threshold), n.times_s))
            masks[key].append(n.eligible)
    aux = {"rich": _standardizer((r.rich, _rich_mask(r)) for r in records)}
    for key in ("legacy", "pose"):
        present = [r for r in records if getattr(r, key) is not None]
        if present:
            if len(present) != len(records):
                raise ContractError(f"Partial TRAIN {key} availability")
            aux[key] = _standardizer((getattr(r, key), getattr(r, key+"_valid")) for r in records)
    # Stream buses one at a time: never pin the entire dataset through a list.
    def tagged_audio():
        for partition, values in audio_rows:
            if partition != "train":
                raise ContractError("Audio normalization accepts TRAIN only")
            yield values, np.ones(len(values), bool)
    iterator = iter(tagged_audio())
    first = next(iterator, None)
    if first is None:
        audio_norm = None
    else:
        from itertools import chain
        audio_norm = _standardizer(chain([first], iterator))
    result = {"schema_version": 1, "fit_partition": "train",
        "fit_recording_ids": [r.recording_id for r in records],
        "fit_canonical_ids": sorted({r.canonical_id for r in records}),
        "fit_validity": [{"recording_id":r.recording_id, "frames":len(r.times),
            "frame_valid":int(np.sum(r.frame_valid)) if r.frame_valid is not None else None,
            "color_valid":int(r.color_valid.sum()), "value_valid":int(r.value_valid.sum()),
            "mask_activity_known":int(np.sum(r.mask_activity_known)) if r.mask_activity_known is not None else None}
            for r in records],
        "palette_prototypes": prototypes.tolist(), "palette_size": palette_size,
        "palette_max_samples_per_recording": max_samples, "fit_seed": seed,
        "color_threshold": color_threshold, "accent_threshold": accent_threshold,
        "threshold_quantile": .95, "novelty_window_s": .3, "event_nms_s": .5,
        "intensity_gain_scope": "complete_available_extracted_recording_window",
        "pos_weight": {k: positive_weight(np.concatenate(labels[k]), np.concatenate(masks[k]))
                       for k in labels},
        "event_counts": {k: int(np.concatenate(labels[k]).sum()) for k in labels},
        "eligible_counts": {k: int(np.concatenate(masks[k]).sum()) for k in masks},
        "aux": aux, "audio": audio_norm}
    result["norm_sha256"] = json_hash({"audio": audio_norm, "aux": aux})
    result["vocab_sha256"] = json_hash(result["palette_prototypes"])
    return result


def _normalize(values, normalizer):
    if normalizer is None:
        raise ContractError("Missing frozen TRAIN normalizer")
    return ((values-np.asarray(normalizer["mean"])) /
            np.asarray(normalizer["std"])).astype(np.float32)


def make_targets(record, transforms):
    color = record.novelty()
    accent, gain = intensity_novelty(
        record.times if record.source_times is None else record.source_times,
        record.values, record.value_valid, record.cuts)
    color_events = chromatic_event_targets(color, threshold=transforms["color_threshold"])
    accent_events = chromatic_event_targets(accent, threshold=transforms["accent_threshold"])
    prototypes = np.asarray(transforms["palette_prototypes"], np.float32)
    # Squared Euclidean assignment is exactly the frozen KMeans categorical mark.
    distances = ((record.hs[:, None]-prototypes[None])**2).sum(2)
    target = {"color": event_labels(color_events, record.times), "color_mask": color.eligible,
              "accent10": event_labels(accent_events, record.times), "accent_mask10": accent.eligible,
              "palette": distances.argmin(1).astype(np.int64),
              "palette_mask": record.color_valid & ~record.cuts,
              "intensity_recording_gain": gain}
    for key, values, mask in (("rich", record.rich, _rich_mask(record)),
                               ("legacy", record.legacy, record.legacy_valid),
                               ("pose", record.pose, record.pose_valid)):
        if values is not None:
            target[key] = _normalize(values, transforms["aux"][key])
            target[key+"_mask"] = mask
    return target


class SourceStore:
    """Hash-checked source readers with a strict bounded in-memory audio LRU."""
    def __init__(self, manifest, *, max_ram_bytes=512*1024**2):
        self.manifest = manifest
        self.max_ram_bytes = int(max_ram_bytes)
        self.audio_cache = OrderedDict()
        self.cache_bytes = 0
        self.verified = set()
        self.rows = {r["recording_id"]: r for r in manifest["records"]}
        self.validate_manifest()

    def checked(self, ref):
        if not isinstance(ref, dict) or set(("path", "sha256"))-ref.keys():
            raise ContractError("Every input requires path and sha256")
        path = Path(ref["path"]).resolve()
        key = str(path), ref["sha256"]
        if key not in self.verified:
            if not path.is_file() or sha256(path) != ref["sha256"]:
                raise ContractError(f"Missing or changed source: {path}")
            self.verified.add(key)
        return path

    def validate_manifest(self):
        if self.manifest.get("schema_version") != 1 or not self.rows:
            raise ContractError("Expected nonempty manifest schema_version 1")
        if len(self.rows) != len(self.manifest["records"]):
            raise ContractError("Duplicate recording rows in manifest")
        split = json.loads(self.checked(self.manifest["split_manifest"]).read_text())
        if not isinstance(split.get("recordings"), dict):
            raise ContractError("Expected immutable Stage11 recordings split mapping")
        group_partitions = {}
        for rid, row in self.rows.items():
            if row.get("partition") not in ("train", "val", "dev"):
                raise ContractError("TEST/result access is locked; manifest only accepts TRAIN/DEV")
            identity = split["recordings"].get(rid)
            if identity is None or identity["canonical_id"] != row["canonical_id"]:
                raise ContractError(f"Canonical identity drift: {rid}")
            canonical_partition = lambda x: "dev" if x == "val" else x
            partition = canonical_partition(row["partition"])
            if canonical_partition(identity["partition"]) != partition:
                raise ContractError(f"Historical split drift: {rid}")
            previous = group_partitions.setdefault(row["canonical_id"], partition)
            if previous != partition:
                raise ContractError("Canonical family appears across partitions")
        self.split_sha256 = self.manifest["split_manifest"]["sha256"]

    def audio(self, rid):
        if rid in self.audio_cache:
            self.audio_cache.move_to_end(rid)
            return self.audio_cache[rid]
        row = self.rows[rid]
        with np.load(self.checked(row["audio"]), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if abs(meta["fps"]-40) > 1e-6:
                raise ContractError("Source audio bus must be 40Hz")
            times = _times(z["times"], 40, "full-record audio", start=0)
            mel = z["mel"].astype(np.float32)
            if mel.shape != (64, len(times)):
                raise ContractError("Source audio mel must be (64,T)")
            values = np.concatenate((mel.T, np.stack([z["ch__"+k] for k in AUDIO_SCALARS], 1)), 1)
        with np.load(self.checked(row["rhythm"]), allow_pickle=False) as z:
            rt = _times(z["times_s"], 50, "BeatThis full-record times", start=0)
            logits = np.stack((z["beat_logit"], z["downbeat_logit"]), 1)
        probabilities = 1/(1+np.exp(-np.clip(_finite(logits, "rhythm logits"), -60, 60)))
        # BeatThis and bus endpoint definitions differ by at most one audio
        # frame; nearest endpoint hold is allowed ONLY within .025 seconds.
        if times[-1] > rt[-1]+.025+1e-6:
            raise ContractError("BeatThis predictions do not cover full source bus")
        rhythm, supported = sample_linear(rt, probabilities, np.minimum(times, rt[-1]))
        if not supported.all():
            raise ContractError("Unsupported rhythm interpolation")
        result = _finite(np.concatenate((values, rhythm), 1).astype(np.float32), "audio94")
        if result.nbytes > self.max_ram_bytes:
            raise ContractError("One source bus exceeds bounded audio RAM cap")
        while self.cache_bytes+result.nbytes > self.max_ram_bytes and self.audio_cache:
            _, removed = self.audio_cache.popitem(last=False)
            self.cache_bytes -= removed.nbytes
        self.audio_cache[rid] = result
        self.cache_bytes += result.nbytes
        return result

    def visual(self, rid, *, include_legacy=False, include_pose=False):
        row = self.rows[rid]
        prefix = row.get("visual_prefix", "sam2")
        keymap = row.get("visual_keys", {})
        names = {"rich": "rich", "hs": "hs_histogram", "color_valid": "color_valid",
                 "retained": "retained_fraction", "chromatic": "chromatic_fraction",
                 "values": "value_mean", "value_valid": "value_valid"}
        source_times = stored = frame_valid = activity_known = None
        with np.load(self.checked(row["visual"]), allow_pickle=False) as z:
            if row.get("visual_schema") == "extractor_v1":
                receipt = json.loads(self.checked(row["visual_complete"]).read_text())
                if receipt.get("status") != "complete":
                    raise ContractError("Visual extractor record is not complete")
                times = z["requested_times_s"].astype(np.float64)
                source_times = z["source_times_s"].astype(np.float64)
                frame_valid = z["frame_valid"] if "frame_valid" in z else None
                activity_known = z["mask_activity_known"] if "mask_activity_known" in z else None
                if (source_times.shape != times.shape or np.any(np.diff(source_times) <= 0)
                        or not np.isfinite(source_times).all()):
                    raise ContractError("Actual source PTS must increase strictly")
                arrays = {"rich":z["rich96"].astype(np.float32),
                          "hs":z["hs_histogram"].astype(np.float32),
                          "color_valid":z["color_valid"], "retained":z["retained_fraction"].astype(np.float32),
                          "chromatic":z["chromatic_fraction"].astype(np.float32),
                          "values":z["value_summary"][:,0].astype(np.float32),
                          "value_valid":z["value_valid"]}
                # Re-normalize quantized distribution roundoff, not invalid rows.
                mass = arrays["hs"].sum(1)
                valid = arrays["color_valid"]
                if np.any(np.abs(mass[valid]-1) > .002) or np.any(arrays["hs"][~valid] != 0):
                    raise ContractError("Invalid quantized HS distribution")
                arrays["hs"][valid] /= mass[valid, None]
                arrays["rich"][:, :72] = arrays["hs"]
                if "cuts" in row:
                    with np.load(self.checked(row["cuts"]), allow_pickle=False) as cz:
                        if not np.array_equal(cz["times_s"], times):
                            raise ContractError("Cut mask does not match requested visual grid")
                        cuts = cz["cuts"]
                elif row.get("cuts_status") == "unannotated":
                    cuts = np.zeros(len(times), bool)
                    stored = SimpleNamespace(times_s=z["novelty_times_s"].astype(np.float64),
                                             scores=z["novelty_scores"].astype(np.float64),
                                             eligible=z["novelty_eligible"])
                else:
                    raise ContractError("Extractor lacks cuts: supply explicit cut mask or unannotated smoke status")
            else:
                times = z[keymap.get("times", "times_s")].astype(np.float64)
                try:
                    arrays = {name: z[keymap.get(name, prefix+"_"+suffix)]
                              for name, suffix in names.items()}
                    cuts = z[keymap.get("cuts", "cuts")]
                except KeyError as error:
                    raise ContractError(f"Visual key missing; provide explicit visual_keys: {error}") from error
        record = VisualRecord(rid, row["canonical_id"], row["partition"], times,
                              cuts=cuts, source_times=source_times, stored_novelty=stored,
                              frame_valid=frame_valid, mask_activity_known=activity_known, **arrays)
        record.validate()
        if include_legacy:
            with np.load(self.checked(row["legacy_visual"]), allow_pickle=False) as z:
                lt = z["times"].astype(np.float64) + float(z["offset_s"])
                legacy, supported = sample_linear(lt, z["y"], times)
            observed_frame = record.value_valid if record.frame_valid is None else record.frame_valid
            record.legacy, record.legacy_valid = legacy, supported & ~cuts & observed_frame
        if include_pose:
            if not row.get("coaches"):
                raise ContractError(f"Pose arm lacks coaches: {rid}")
            all_values, all_valid = [], []
            for coach in row["coaches"]:
                with np.load(self.checked(coach["motion"]), allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                    offset = float(coach["start_s"])
                    if abs(float(meta["start_s"])-offset) > 1e-5:
                        raise ContractError("Coach active-span offset drift")
                    pt = z["times"].astype(np.float64)+offset
                    raw = np.stack([z["ch__"+name] for name in POSE_NAMES], 1)
                    pose, supported = sample_linear(pt, raw, times)
                if "validity" not in coach:
                    raise ContractError("Pose requires explicit gap/cut/tracking validity; finite is not observed")
                with np.load(self.checked(coach["validity"]), allow_pickle=False) as z:
                    valid = z["valid"]
                    vt = z["times_s"]
                    if valid.dtype != bool or valid.shape != (len(vt), 4):
                        raise ContractError("Pose validity must be boolean (T,4) in POSE_NAMES order")
                    valid = _nearest_valid(vt, valid, times)
                valid &= supported[:, None] & ~cuts[:, None] & np.isfinite(pose)
                all_values.append(np.where(valid, pose, 0)); all_valid.append(valid)
            count = np.sum(all_valid, axis=0)
            record.pose = (np.sum(all_values, axis=0)/np.maximum(count, 1)).astype(np.float32)
            record.pose_valid = count > 0
        return record


def deterministic_donors(records, tempos=None):
    """Fixed different-family, same-partition donor; nearest log tempo if given."""
    result = {}
    for row in sorted(records, key=lambda r: r.recording_id):
        choices = [r for r in records if r.canonical_id != row.canonical_id
                   and r.partition == row.partition and len(r.times) >= 640]
        if not choices:
            raise ContractError("No different-canonical donor in this partition")
        def key(candidate):
            distance = (abs(math.log(tempos[candidate.recording_id]/tempos[row.recording_id]))
                        if tempos is not None else 0.)
            return distance, json_hash([row.recording_id, candidate.recording_id])
        result[row.recording_id] = min(choices, key=key).recording_id
    return result


class BatchSampler:
    """Equal canonical-song probability, then recording, then crop start."""
    def __init__(self, records, seed=0):
        self.rng = np.random.default_rng(seed)
        self.groups = defaultdict(list)
        for record in sorted(records, key=lambda r: r.recording_id):
            self.groups[record.canonical_id].append((record.recording_id, len(record.times)-640+1))
        self.canonicals = sorted(self.groups)
        if not self.canonicals or any(n < 1 for group in self.groups.values() for _, n in group):
            raise ContractError("Sampling requires nonempty >=64s records")

    def sample(self, batch):
        result = []
        for _ in range(batch):
            group = self.groups[self.canonicals[int(self.rng.integers(len(self.canonicals)))]]
            rid, starts = group[int(self.rng.integers(len(group)))]
            result.append((rid, int(self.rng.integers(starts))))
        return result


class TrainingData:
    def __init__(self, store, records, transforms, recipe):
        self.store, self.records, self.transforms, self.recipe = store, {r.recording_id:r for r in records}, transforms, recipe
        self.targets = {rid: make_targets(r, transforms) for rid, r in self.records.items()}
        self.donors = {}
        self.donor_tempos = {}
        if recipe in ("S_VR_Pshuf", "S_VR_P0_Ashuf"):
            tempos = {}
            for rid in self.records:
                period = store.audio(rid)[:, 64+AUDIO_SCALARS.index("beat_period_s")]
                if not np.all(period > 0):
                    raise ContractError("Donor tempo matching needs positive beat periods")
                tempos[rid] = 60/float(np.median(period))
            self.donor_tempos = tempos
            self.donors = deterministic_donors(records, tempos)

    def donor_statistics(self):
        from collections import Counter
        reuse = Counter(self.donors.values())
        counts = Counter(reuse.values())
        return {"policy":"conditional_tempo_nearest_donor_not_bijective_permutation",
            "tempo_source":"60/median(full_source_audio_bus_beat_period_s)",
            "recipient_count":len(self.donors), "unique_donor_count":len(reuse),
            "maximum_recipient_reuse":max(reuse.values(),default=0),
            "donor_reuse_distribution":{str(uses):count for uses,count in sorted(counts.items())},
            "donor_map_sha256":json_hash(self.donors), "recording_tempo_bpm":self.donor_tempos,
            "claim_limit":"explicit recipient-rhythm-conditioned donor control; not a bijective permutation or pure nonrhythm causal intervention"}

    def batch(self, samples, device="cpu"):
        batches = defaultdict(list)
        visual_dim = RECIPES[self.recipe][1]
        for rid, start in samples:
            record, target = self.records[rid], self.targets[rid]
            ts = record.times[start:start+640]
            if len(ts) != 640:
                raise ContractError("Crop does not contain 640 source visual frames")
            # Use absolute original-video time, never active-span-relative time.
            query40 = ts[0]+np.arange(2560)/40
            audio = self.store.audio(rid)
            x, supported = sample_linear(np.arange(len(audio))/40, audio, query40, max_gap_s=.026)
            if not supported.all():
                raise ContractError("Full source audio does not cover absolute crop")
            if self.recipe == "S_VR_P0_Ashuf":
                donor = self.donors[rid]
                dr = self.records[donor]
                ds = min(start, len(dr.times)-640)
                da = self.store.audio(donor)
                dx, dv = sample_linear(np.arange(len(da))/40, da,
                                       dr.times[ds]+np.arange(2560)/40, max_gap_s=.026)
                if not dv.all():
                    raise ContractError("Audio donor has insufficient context")
                keep = x[:, RHYTHM_COLUMNS].copy()
                x = dx
                x[:, RHYTHM_COLUMNS] = keep
            batches["audio40"].append(_normalize(x, self.transforms["audio"]))
            section = slice(start, start+640)
            for key in ("color", "color_mask", "palette", "palette_mask"):
                batches[key].append(target[key][section])
            # A native 10Hz bin supervises its four aligned acoustic subframes;
            # this is explicitly NOT independent fine beat timing ground truth.
            batches["accent"].append(np.repeat(target["accent10"][section], 4))
            batches["accent_mask"].append(np.repeat(target["accent_mask10"][section], 4))
            if visual_dim:
                key = "rich" if visual_dim == 96 else "legacy"
                if key not in target:
                    raise ContractError(f"Missing {key} auxiliary for {self.recipe}")
                batches["visual_aux"].append(target[key][section])
                batches["visual_aux_mask"].append(target[key+"_mask"][section])
            if RECIPES[self.recipe][2]:
                pt, ps = target, start
                if self.recipe == "S_VR_Pshuf":
                    pt = self.targets[self.donors[rid]]
                    ps = min(start, len(pt["pose"])-640)
                batches["pose_aux"].append(pt["pose"][ps:ps+640])
                batches["pose_aux_mask"].append(pt["pose_mask"][ps:ps+640])
        return {key: torch.from_numpy(np.stack(value)).to(device) for key, value in batches.items()}


def _cue_indices(probabilities, threshold=.5):
    rows = []
    for probability in probabilities:
        novelty = SimpleNamespace(times_s=np.arange(len(probability))/10,
                                  scores=probability, eligible=np.ones(len(probability), bool))
        rows.append(chromatic_event_targets(novelty, threshold=threshold).event_indices.tolist())
    return rows


@torch.no_grad()
def generated_history(model, context10, color_logits, *, initial_ids=None, state_ablation=False,
                      cue_threshold=.5, initialize=True):
    """Autoregress from own cue probabilities and own mark argmax, never gold.

    Previous ids at t are state BEFORE t. First state is null. Between own
    cues the generated mark is held. Initial mark at sequence start initializes
    display state; it is not counted as a predicted color cue. All K+1 possible
    states at cue locations are evaluated in one bounded tensor, then the exact
    discrete recurrence is resolved on CPU. This avoids T tiny GPU launches.
    """
    batch, frames, width = context10.shape
    k = model.cfg.palette_size
    state = (torch.full((batch,), k, dtype=torch.long, device=context10.device)
             if initial_ids is None else initial_ids.to(context10.device).long().clone())
    if state.shape != (batch,) or bool(((state < 0) | (state > k)).any()):
        raise ContractError("Initial palette state must be valid (B,) generated ids")
    cues = _cue_indices(torch.sigmoid(color_logits.float()).cpu().numpy(), cue_threshold)
    pairs = [(b, t) for b in range(batch) for t in sorted(set(cues[b]+([0] if initialize else [])))]
    previous = torch.empty((batch, frames), dtype=torch.long)
    generated = torch.empty_like(previous)
    event_mask = torch.zeros_like(previous, dtype=torch.bool)
    for b, indices in enumerate(cues):
        event_mask[b, indices] = True
    tables = {}
    if pairs:
        contexts = torch.stack([context10[b, t] for b, t in pairs])
        expanded = contexts[:, None].expand(-1, k+1, -1)
        ids = torch.arange(k+1, device=context10.device).expand(len(pairs), -1)
        if state_ablation:
            ids = torch.full_like(ids, k)
        choices = model.palette_logits(expanded, ids).argmax(-1).cpu().numpy()
        tables = {pair: choices[i] for i, pair in enumerate(pairs)}
    for b in range(batch):
        current = int(state[b])
        for t in range(frames):
            previous[b, t] = k if state_ablation else current
            if (b, t) in tables:
                current = int(tables[b, t][current])
            generated[b, t] = current
        state[b] = current
    return {"previous_ids": previous.to(context10.device),
            "generated_palette_ids": generated.to(context10.device),
            "color_cues": event_mask.to(context10.device), "final_ids": state}


def _masked_mean(loss, mask):
    mask = mask.bool()
    if mask.shape != loss.shape:
        mask = torch.broadcast_to(mask, loss.shape)
    # Empty slices yield differentiable zero, never NaN. Dataset-wide zero
    # eligibility is rejected at fitting/validation, not by dropping quiet crops.
    return torch.where(mask, loss, torch.zeros_like(loss)).sum()/mask.sum().clamp_min(1)


def _aux_loss(output, target, mask):
    if mask.ndim == 2:
        mask = mask[..., None].expand_as(target)
    error = (output.float()-target.float()).square()
    # Mean per coordinate, then over observed coordinates: 96D never dominates 6D.
    count = mask.sum((0, 1))
    per_coordinate = torch.where(mask, error, torch.zeros_like(error)).sum((0, 1))/count.clamp_min(1)
    return _masked_mean(per_coordinate, count > 0)


def compute_loss(model, batch, transforms, *, state_dropout=.5, dropout_generator=None):
    """Only center 8s is supervised; generated state uses the 64s crop context."""
    outputs = model(batch["audio40"])
    rollout = generated_history(model, outputs["context10"].detach(),
                                outputs["chromatic_change_logit"].detach())
    ids = rollout["previous_ids"]
    if not 0 <= state_dropout <= 1:
        raise ContractError("state_dropout must be in [0,1]")
    drop = torch.rand((len(ids), 1), generator=dropout_generator, device="cpu") < state_dropout
    ids = ids.masked_fill(drop.to(ids.device), model.cfg.null_palette_id)
    palette = model.palette_logits(outputs["context10"], ids)
    losses = {}
    for key, output_key, section in (("accent", "accent_logit", CENTER40),
                                     ("color", "chromatic_change_logit", CENTER10)):
        logits, target = outputs[output_key][:, section].float(), batch[key][:, section].float()
        weight = torch.tensor(transforms["pos_weight"][key], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
        losses[key] = _masked_mean(loss, batch[key+"_mask"][:, section])
    ce = F.cross_entropy(palette[:, CENTER10].float().transpose(1, 2),
                         batch["palette"][:, CENTER10], reduction="none")
    losses["palette"] = _masked_mean(ce, batch["palette_mask"][:, CENTER10])
    losses["main"] = losses["accent"]+losses["color"]+losses["palette"]
    total = losses["main"]
    for key in ("visual_aux", "pose_aux"):
        if key in outputs:
            losses[key] = _aux_loss(outputs[key][:, CENTER10], batch[key][:, CENTER10],
                                   batch[key+"_mask"][:, CENTER10])
            total = total + .1*losses[key]
    losses["total"] = total
    return losses, {**outputs, "palette_logits": palette,
                    "generated_previous_ids": ids, **rollout}


@torch.no_grad()
def predict(model, audio40, *, transforms=None, state_ablation=False, initial_ids=None):
    """Bounded crop-context predictions with full chronological generated state.

    One full-record (T40,94) array, raw if transforms are passed, otherwise
    standardized. Each interior tile emits its center8s from a 64s crop. At
    true source boundaries the crop is CLIPPED, never zero padded; truncated
    support is marked. No chunk/full equivalence is assumed. All tile contexts
    are decoded together in chronological order so cue NMS and palette state
    continue across tile seams. No gold palette/history input is accepted.
    """
    values = np.asarray(audio40, np.float32)
    if values.ndim != 2 or values.shape[1] != model.cfg.in_dim or len(values) % 4:
        raise ContractError("predict needs (T40,94) with T40 divisible by four")
    _finite(values, "prediction audio")
    if not len(values):
        raise ContractError("Empty prediction audio")
    if transforms is not None:
        values = _normalize(values, transforms["audio"])
    if model.cfg.context_radius10 > 280:
        raise ContractError("Crop halo does not cover configured model radius")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    accents, colors, contexts = [], [], []
    frames = len(values)//4
    try:
        for start in range(0, frames, 80):
            count = min(80, frames-start)
            low, high = max(0, start-280), min(frames, start+360)
            center = start-low
            outputs = model(torch.from_numpy(values[None, low*4:high*4]).to(device))
            contexts.append(outputs["context10"][:,center:center+count].cpu())
            colors.append(outputs["chromatic_change_logit"][:,center:center+count].cpu())
            accents.append(outputs["accent_logit"][:,center*4:(center+count)*4].float().cpu())
        context, color = torch.cat(contexts,1).to(device), torch.cat(colors,1).to(device)
        rollout = generated_history(model, context, color, initial_ids=initial_ids,
                                    state_ablation=state_ablation, initialize=initial_ids is None)
        palette = model.palette_logits(context, rollout["previous_ids"])
        output = {
            "accent_probability":torch.sigmoid(torch.cat(accents,1))[0].numpy(),
            "color_probability":torch.sigmoid(color.float())[0].cpu().numpy(),
            "palette_probability":torch.softmax(palette.float(),-1)[0].cpu().numpy(),
            "generated_palette_ids":rollout["generated_palette_ids"][0].cpu().numpy(),
            "color_cues":rollout["color_cues"][0].cpu().numpy(),
            "final_palette_ids":rollout["final_ids"].cpu().numpy(),
            "times40_s":np.arange(len(values))/40, "times10_s":np.arange(frames)/10,
            "context_truncated":((np.arange(frames) < model.cfg.context_radius10) |
                (np.arange(frames) >= frames-model.cfg.context_radius10)),
            "schema_version":1, "context_method":"64s_crop_8s_center_true_edges_clipped",
            "chunk_full_equivalence_verified":False, "state_ablation":bool(state_ablation)}
    finally:
        model.train(was_training)
    return output


def build_training_manifest(video_manifest_path, visual_root, split_path, *, mode="sam2",
                            exclusions=None):
    """Adapt completed extractor outputs; no reads of historical result files.

    ``exclusions`` is an explicit {original_recording_id: reason} mapping. It
    permits a reduced engineering smoke, never replacement records. Source
    extraction selection and coverage remain in the generated manifest.
    """
    if mode not in ("box", "sam2"):
        raise ContractError("Mask mode must be box or sam2")
    video_manifest_path, visual_root = Path(video_manifest_path), Path(visual_root)
    video = json.loads(video_manifest_path.read_text())
    exclusions = exclusions or {}
    original = [row["recording_id"] for row in video["inputs"]]
    if len(original) != len(set(original)) or set(exclusions)-set(original):
        raise ContractError("Invalid original selection/exclusion ledger")
    if any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()):
        raise ContractError("Every exclusion needs an explicit reason")
    def reference(path):
        path = Path(path).resolve()
        return {"path":str(path), "sha256":sha256(path)}
    rows = []
    for row in video["inputs"]:
        rid = row["recording_id"]
        if rid in exclusions:
            continue
        directory = visual_root/rid
        receipt = json.loads((directory/"complete.json").read_text())
        if receipt.get("status") != "complete" or (directory/"invalid.json").exists():
            raise ContractError(f"Visual extraction not complete/valid: {rid}")
        rows.append({"recording_id":rid, "canonical_id":row["canonical_id"],
            "enrollment_input_sha256":json_hash(row),
            "source_video":{"path":row["source_path"], "sha256":row["source_sha256"]},
            "source_pose":{"path":row["pose_path"], "sha256":row["pose_sha256"]},
            "partition":row["partition"], "audio":reference(row["audio_bus_path"]),
            "rhythm":reference(row["rhythm_path"]), "visual":reference(directory/f"{mode}.npz"),
            "visual_complete":reference(directory/"complete.json"), "visual_schema":"extractor_v1",
            "cuts_status":"unannotated", "legacy_visual":reference(row["legacy_visual_path"]),
            "coaches":[{"start_s":coach["start_s"], "motion":reference(coach["motion_path"])}
                       for coach in row.get("coaches", [])]})
    return {"schema_version":1, "campaign_id":video["campaign_id"],
        "campaign_root":str(video_manifest_path.parent), "split_manifest":reference(split_path),
        "video_input_manifest":reference(video_manifest_path), "mask_mode":mode,
        "fit_auxiliaries":["rich", "legacy"], "records":rows,
        "smoke_selection":{"original_recording_ids":original, "exclusions":exclusions,
                           "replacement_records":False, "scope":"functional_only_not_quality"}}


def validate_smoke_selection(manifest):
    rows = manifest["records"]
    if not 1 <= len(rows) <= 12 or any(row["partition"] != "train" for row in rows):
        raise ContractError("Smoke needs 1..12 TRAIN-only records; no DEV/TEST")
    if len(rows) < 12:
        ledger = manifest.get("smoke_selection", {})
        originals = ledger.get("original_recording_ids", [])
        exclusions = ledger.get("exclusions", {})
        included = {r["recording_id"] for r in rows}
        if (len(originals) != 12 or len(set(originals)) != 12 or ledger.get("replacement_records") is not False
                or included & set(exclusions) or included | set(exclusions) != set(originals)
                or any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values())):
            raise ContractError("Reduced smoke requires original12 IDs plus explicit no-replacement exclusions")


@dataclass(frozen=True)
class TrainConfig:
    recipe: str = "S_VR_P0"
    seed: int = 0
    steps: int = 6000
    batch: int = 4
    lr: float = 3e-4
    weight_decay: float = .01
    warmup: int = 300
    gradient_clip: float = 1.
    check_interval: int = 300
    state_dropout: float = .5
    cue_threshold: float = .5
    main_weights: tuple = (1., 1., 1.)
    auxiliary_weight: float = .1
    crop_frames40: int = 2560
    center10: tuple = (280, 360)
    precision: str = "bf16"
    compile: bool = False
    max_minutes: float = 60.

    def validate(self, smoke):
        if self.recipe not in RECIPES:
            raise ContractError("Unknown/unqualified recipe; R1 recurrence is blocked")
        if self.seed not in (0, 1, 2) or self.steps < 1 or self.batch < 1:
            raise ContractError("Seeds must be 0/1/2 and steps/batch positive")
        if not smoke and (self.steps != 6000 or self.batch != 4):
            raise ContractError("Non-smoke fits require exactly 6000 updates and batch4")
        if smoke and self.steps > 300:
            raise ContractError("Smoke is bounded to <=300 functional updates")
        fixed = TrainConfig()
        for name in ("lr", "weight_decay", "warmup", "gradient_clip", "check_interval",
                     "state_dropout", "cue_threshold", "main_weights", "auxiliary_weight",
                     "crop_frames40", "center10", "precision", "compile", "max_minutes"):
            if getattr(self, name) != getattr(fixed, name):
                raise ContractError(f"Frozen implementation constant cannot change silently: {name}")


def validate_run_scope(manifest, config, *, smoke=False, observer_proxy=False):
    """Fail closed before target reads; this does not waive any original gate.

    Observer mode is a separate registered instrument. Unknown cuts belong to
    its observed authored-video labels; they are not certified clean lighting.
    The default path retains the original missing-cut prohibition.
    """
    config.validate(smoke)
    marked = manifest.get("study_kind") == OBSERVER_STUDY_KIND
    if marked and not observer_proxy:
        raise ContractError("Observer manifest requires explicit --observer-proxy; original scope unchanged")
    if observer_proxy:
        if smoke or not marked:
            raise ContractError("--observer-proxy requires non-smoke historical_observer_proxy_v1 manifest")
        if manifest.get("unknown_cuts_policy") != OBSERVER_UNKNOWN_CUTS_POLICY:
            raise ContractError("Observer unknown_cuts_policy must include unknown cuts in observed targets")
        if config.recipe not in OBSERVER_RECIPES or RECIPES[config.recipe][2] != 0:
            raise ContractError("Observer scope allows only the five frozen P0 recipes; pose/R1 stay blocked")
        if set(manifest.get("fit_auxiliaries", [])) != {"rich", "legacy"}:
            raise ContractError("Observer common transforms require rich+legacy only, never pose")
        if manifest.get("mask_mode") != "sam2":
            raise ContractError("Observer common target must be the frozen SAM2 observer, not a changed mask arm")
        reference = manifest.get("observer_protocol", {})
        if not reference.get("path") or not reference.get("sha256"):
            raise ContractError("Observer scope requires protocol path and SHA256")
        path = Path(reference["path"]).resolve()
        if not path.is_file() or sha256(path) != reference["sha256"]:
            raise ContractError("Observer protocol hash mismatch")
        protocol = json.loads(path.read_text())
        if protocol.get("id") != OBSERVER_PROTOCOL_ID:
            raise ContractError("Observer protocol id mismatch")
        if protocol.get("campaign_id") != manifest.get("campaign_id"):
            raise ContractError("Observer campaign id mismatch")
        if protocol.get("separate_from_original_quality_claims") is not True:
            raise ContractError("Observer scope must remain separate from original quality claims")
        if protocol.get("capture", {}).get("cut_policy") != (
                "Unknown cuts INCLUDED IN OBSERVER target, never called cut-clean"):
            raise ContractError("Observer protocol cut policy payload mismatch")
        student = protocol.get("student", {})
        expected = {"recipes":list(OBSERVER_RECIPES), "seeds":[0,1,2], "max_fits":15,
                    "steps":6000, "batch":4, "optimizer":"AdamW", "lr":.0003,
                    "weight_decay":.01, "warmup":300, "clip":1,
                    "precision":"BF16 mixed/FP32 losses", "compile":False,
                    "state_dropout":.5, "main_loss_weights":[1,1,1], "auxiliary_weight":.1}
        if any(student.get(key) != value for key,value in expected.items()):
            raise ContractError("Observer protocol student recipe/config payload mismatch")
        context = student.get("context", "")
        if not all(text in context for text in (
                "64s student feature window", "FULL SOURCE RECORD",
                "NOT a64s raw-audio or causal-context claim", "Legacy6 fullrecord-relative")):
            raise ContractError("Observer feature provenance payload missing/full-source context changed")
        selection = protocol.get("selection", {})
        if (selection.get("train_groups") != 100 or selection.get("historical_dev_groups") != 31
                or selection.get("original_dev_groups") != 40
                or selection.get("original_dev_coverage") != .775
                or selection.get("original80percentgate") != "FAILED; not waived/relabelled"):
            raise ContractError("Observer selection cannot alter the original31/40=.775 failed gate")
        capture_ref = manifest.get("video_input_manifest", {})
        if (capture_ref.get("sha256") != selection.get("sha256") or not capture_ref.get("path")
                or Path(capture_ref["path"]).resolve() != Path(selection["manifest"]).resolve()
                or sha256(capture_ref["path"]) != capture_ref["sha256"]):
            raise ContractError("Observer fixed capture selection hash/path mismatch")
        enrollment = json.loads(Path(capture_ref["path"]).read_text())
        inventory = {row["recording_id"]:row for row in enrollment["inputs"]}
        if len(inventory) != 131 or len(inventory) != len(enrollment["inputs"]):
            raise ContractError("Observer original inventory must remain131 unique recording IDs")
        rows = manifest["records"]
        identity_ref = manifest.get("identity_exclusions", {})
        if (not identity_ref.get("path") or not identity_ref.get("sha256")
                or not Path(identity_ref["path"]).is_file()
                or sha256(identity_ref["path"]) != identity_ref["sha256"]):
            raise ContractError("Observer identity-exclusion ledger path/hash missing or changed")
        identity = json.loads(Path(identity_ref["path"]).read_text())
        if (identity.get("id") != "observer-proxy-identity-exclusions-v1"
                or identity.get("study_kind") != OBSERVER_STUDY_KIND
                or identity.get("campaign_id") != manifest["campaign_id"]
                or identity.get("capture_manifest_sha256") != capture_ref["sha256"]
                or identity.get("observer_protocol_sha256") != reference["sha256"]
                or Path(identity.get("capture_manifest_path", "")).resolve() != Path(capture_ref["path"]).resolve()
                or Path(identity.get("observer_protocol_path", "")).resolve() != path
                or identity.get("fit_counts_expected") != {"train":91,"val":31}
                or identity.get("immutable_enrollment") != {"train":100,"val":31}
                or identity.get("capture_continues_unchanged") is not True):
            raise ContractError("Observer identity-exclusion ledger scope/count/source binding mismatch")
        identity_entries = identity.get("exclusions", {})
        if len(identity_entries) != 9:
            raise ContractError("Observer v1 identity quarantine must retain exactly9 TRAIN exclusions")
        exclusions = []
        for rid, entry in identity_entries.items():
            if (rid not in inventory or entry.get("partition") != "train"
                    or entry.get("canonical_id") != inventory[rid]["canonical_id"]
                    or "unresolved_audio_similarity" not in entry.get("source_reasons", [])):
                raise ContractError("Identity ledger may only quarantine its enrolled ambiguousTRAIN families")
            exclusions.append({"recording_id":rid,"canonical_id":entry["canonical_id"],"partition":"train",
                "enrollment_input_sha256":json_hash(inventory[rid]),"pre_fit":True,
                "not_based_on_target_or_model_scores":True,"basis":"source_identity",
                "reason":entry.get("reason", ""),"receipt":identity_ref})
        additional = manifest.get("capture_exclusions", [])
        if not isinstance(additional, list):
            raise ContractError("capture_exclusions must be an explicit pre-fit receipt list")
        exclusions.extend(additional)
        usable_ids = {row["recording_id"] for row in rows}
        excluded_ids = {row["recording_id"] for row in exclusions}
        if (len(rows) != len(usable_ids) or len(exclusions) != len(excluded_ids)
                or usable_ids & excluded_ids or usable_ids | excluded_ids != set(inventory)):
            raise ContractError("Observer enrollment changed:131ID inventory must equal disjoint usable+excluded IDs; no replacements")
        for row in rows+exclusions:
            original = inventory[row["recording_id"]]
            if (row.get("canonical_id") != original["canonical_id"]
                    or row.get("partition") != original["partition"]
                    or row.get("enrollment_input_sha256") != json_hash(original)):
                raise ContractError("Observer enrollment identity/partition/source binding mismatch")
        for row in exclusions:
            ref = row.get("receipt", {})
            if (row.get("pre_fit") is not True or row.get("not_based_on_target_or_model_scores") is not True
                    or row.get("basis") not in ("source_identity", "capture_integrity", "source_availability")
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                    or not ref.get("path") or not ref.get("sha256")
                    or not Path(ref["path"]).is_file() or sha256(ref["path"]) != ref["sha256"]):
                raise ContractError("Capture exclusions require pre-fit reason/source-only basis and exact receipt provenance")
        for row in rows:
            original = inventory[row["recording_id"]]
            for field, path_key, hash_key in (
                    ("audio", "audio_bus_path", "audio_bus_sha256"),
                    ("rhythm", "rhythm_path", None), ("legacy_visual", "legacy_visual_path", None),
                    ("source_video", "source_path", "source_sha256"),
                    ("source_pose", "pose_path", "pose_sha256")):
                ref = row.get(field, {})
                if (not ref.get("path") or not original.get(path_key)
                        or Path(ref["path"]).resolve() != Path(original[path_key]).resolve()
                        or (hash_key is not None and original.get(hash_key) is not None
                            and ref.get("sha256") != original[hash_key])):
                    raise ContractError(f"Observer original source binding changed: {field}")
            complete = row.get("visual_complete", {})
            if (not complete.get("path") or not complete.get("sha256")
                    or not Path(complete["path"]).is_file()
                    or sha256(complete["path"]) != complete["sha256"]):
                raise ContractError("Missing completed feature receipt: block or explicitly exclude; never zero-impute a record")
            receipt = json.loads(Path(complete["path"]).read_text())
            if receipt.get("status") != "complete" or json_hash(receipt.get("input")) != json_hash(original):
                raise ContractError("Completed visual receipt does not match exact original enrollment input")
        train_groups = {r["canonical_id"] for r in rows if r["partition"] == "train"}
        dev_groups = {r["canonical_id"] for r in rows if r["partition"] in ("val", "dev")}
        if len(train_groups) < 80 or len(dev_groups) < 30:
            raise ContractError("Observer usable cohort needs >=80 TRAIN and >=30 historicalDEV groups")
        if any(row.get("cuts_status") != "unannotated" or "cuts" in row for row in rows):
            raise ContractError("Observer v1 includes unknown cuts; a changed cut mask needs a new protocol")
        scope = {"study_kind":OBSERVER_STUDY_KIND,
            "unknown_cuts_policy":OBSERVER_UNKNOWN_CUTS_POLICY,
            "observer_protocol_id":protocol["id"], "observer_protocol_path":str(path),
            "observer_protocol_sha256":reference["sha256"],
            "identity_exclusions_id":identity["id"],
            "identity_exclusions_path":str(Path(identity_ref["path"]).resolve()),
            "identity_exclusions_sha256":identity_ref["sha256"],
            "identity_quarantined_recording_ids":sorted(identity_entries),
            "identity_policy":identity.get("policy", "source-metadata ambiguity quarantine before fit"),
            "primary_scope":protocol["primary_scope"],
            "original_quality_gates_unchanged":True,
            "original_dev_coverage":.775, "original_80percent_gate":"FAILED; not waived/relabelled",
            "original_inventory_recordings":131,
            "enrollment_inventory_sha256":json_hash({rid:json_hash(row) for rid,row in sorted(inventory.items())}),
            "usable_train_groups":len(train_groups), "usable_dev_groups":len(dev_groups),
            "usable_dev_fraction_of_original40":len(dev_groups)/40,
            "excluded_recording_ids":sorted(excluded_ids), "capture_exclusions":exclusions,
            "missing_record_imputation":False,
            "student_feature_window_seconds":64, "supervised_center_seconds":8,
            "audio_feature_provenance":"audio bus/global beat grid/BeatThis computed offline on FULL SOURCE RECORD",
            "audio_normalizer_fit_support":"all full-source94D bus frames of usable TRAIN recordings only; no DEV fitting",
            "audio_normalizer_fit_train_groups":len(train_groups),
            "visual_transform_fit_support":"selected TRAIN64s windows for event thresholds, palette vocabulary, posweights and global auxiliary normalizers",
            "main_supervision_support":"center8s only; normalization support is not supervision support",
            "raw_audio_64_second_context_claim":False, "causal_claim":False,
            "legacy_visual_provenance":"immutable Stage11 z[y]: value columns3/4 mean/std normalized over complete active recording before crop; then TRAIN-global per-coordinate auxiliary normalization",
            "pure_feature_dimension_causal_claim":False,
            "unknown_cuts_mask_error_UI_track_nuisance_part_of_observer":True,
            "pose_and_R1":"UNRUN; qualification missing", "max_scope_fits":15}
    else:
        if manifest.get("study_kind") not in (None, "original_light_layers_v1"):
            raise ContractError("Unknown study_kind cannot inherit original training scope")
        if not smoke and any(r.get("cuts_status") == "unannotated" for r in manifest["records"]):
            raise ContractError("Unannotated cuts blocked in original scope; not an implicit observer waiver")
        scope = {"study_kind":"functional_smoke" if smoke else "original_light_layers_v1",
                 "unknown_cuts_policy":"unqualified_smoke_only" if smoke else "qualified_frozen_cut_policy_required",
                 "original_quality_gates_unchanged":True}
    return {**scope, "scope_sha256":json_hash(scope)}


def implementation_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), Path(__file__).with_name("model.py"),
             Path(__file__).with_name("targets.py"), root/"lightsem"/"model"/"light_trunk.py",
             root/"scripts"/"light_layers"/"run_light_layers_train.py"]
    return {str(path): sha256(path) for path in paths}


def provenance(manifest_path, store, config, transforms=None, *, scope=None):
    source = implementation_hashes()
    if transforms is not None:
        actual_norm = json_hash({"audio":transforms["audio"], "aux":transforms["aux"]})
        actual_vocab = json_hash(transforms["palette_prototypes"])
        if transforms.get("norm_sha256") != actual_norm or transforms.get("vocab_sha256") != actual_vocab:
            raise ContractError("Frozen transform payload does not match stored norm/vocab hashes")
    return {"manifest_sha256": sha256(manifest_path), "source_sha256": json_hash(source),
            "source_files": source, "split_sha256": store.split_sha256,
            "config_sha256": json_hash(asdict(config)),
            "norm_sha256": None if transforms is None else transforms["norm_sha256"],
            "vocab_sha256": None if transforms is None else transforms["vocab_sha256"],
            **({key:scope[key] for key in OBSERVER_SCOPE_BINDINGS if key in scope}
               if scope is not None else {})}


def verify_preregistration(path, bindings, recipe, *, require_fits=True):
    if path is None:
        raise ContractError("Non-smoke needs an exact preregistration JSON, not merely a design citation")
    registration = json.loads(Path(path).read_text())
    if not registration.get("registered_before_fit") or not registration.get("vault_experiment_id"):
        raise ContractError("Preregistration needs registered_before_fit and vault_experiment_id")
    allowed = registration.get("recipes", {})
    if recipe not in allowed:
        raise ContractError("Recipe not bound in preregistration")
    expected = {**registration.get("bindings", {}), **allowed[recipe]}
    is_observer = bindings.get("study_kind") == OBSERVER_STUDY_KIND
    if is_observer:
        if set(allowed) != set(OBSERVER_RECIPES):
            raise ContractError("Observer preregistration must bind exactly the five P0 recipes")
        for name in OBSERVER_RECIPES:
            configs = allowed[name].get("config_sha256")
            expected_configs = {json_hash(asdict(TrainConfig(recipe=name, seed=seed))) for seed in (0,1,2)}
            if not isinstance(configs, list) or len(configs) != 3 or set(configs) != expected_configs:
                raise ContractError("Observer preregistration needs exact6000/batch4 configs for seeds0/1/2 per recipe")
        if (registration.get("study_kind") != OBSERVER_STUDY_KIND
                or registration.get("unknown_cuts_policy") != OBSERVER_UNKNOWN_CUTS_POLICY):
            raise ContractError("Observer preregistration scope/cut policy mismatch")
        for key in OBSERVER_SCOPE_BINDINGS:
            if not bindings.get(key) or registration.get("bindings", {}).get(key) != bindings[key]:
                raise ContractError(f"Observer preregistration scope binding mismatch: {key}")
    elif registration.get("study_kind") == OBSERVER_STUDY_KIND:
        raise ContractError("Observer preregistration cannot authorize original quality scope")
    for key in ("manifest_sha256", "source_sha256", "split_sha256", "config_sha256",
                "norm_sha256", "vocab_sha256", "transforms_sha256"):
        if not require_fits and key in ("norm_sha256", "vocab_sha256", "transforms_sha256"):
            continue
        allowed = expected.get(key)
        matches = bindings.get(key) in allowed if isinstance(allowed, list) else allowed == bindings.get(key)
        if not bindings.get(key) or not matches:
            raise ContractError(f"Preregistration binding absent or changed: {key}")
    return sha256(path)


def campaign_space(root, reserve_bytes=0):
    """All campaign files count, including weights, rhythm and video artifacts."""
    root = Path(root)
    if not root.is_dir():
        raise ContractError("Campaign root must already exist on the authorized mount")
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    free = shutil.disk_usage(root).free
    if used+reserve_bytes > int(.95*GIB) or free-reserve_bytes < GIB:
        raise ContractError(f"Storage gate: campaign={used}, reserve={reserve_bytes}, free={free}; "
                            "cap .95GiB, free floor1GiB; no old artifacts removed")
    return {"campaign_bytes": used, "free_bytes": free}


@contextmanager
def gpu_lock(campaign_root):
    # Shared fixed lock across THIS campaign's fits. Parent must also coordinate
    # extraction runners; they may have their own locks. Nonblocking exclusive.
    with (Path(campaign_root)/"gpu.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError("Another campaign GPU job holds gpu.lock") from error
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seed_all(seed, *, cuda=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_digest(sampler, dropout_generator, *, cuda=False):
    result = {"python_sha256": json_hash(random.getstate()),
              "numpy_batch_state": sampler.rng.bit_generator.state,
              "torch_cpu_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
              "state_dropout_sha256": hashlib.sha256(dropout_generator.get_state().numpy().tobytes()).hexdigest()}
    if cuda:
        result["torch_cuda_sha256"] = [hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                                         for state in torch.cuda.get_rng_state_all()]
    return result


def load_checkpoint(path, *, device="cpu", expected_bindings=None):
    """Load model-only weights without narrowing an independent model twice."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    for key, expected in (expected_bindings or {}).items():
        if metadata.get(key) != expected:
            raise ContractError(f"Checkpoint binding mismatch: {key}")
    reference = LightLayersConfig(**metadata["reference_model_config"])
    model = build_light_layers(reference, architecture=metadata["architecture"])
    if asdict(model.cfg) != metadata["model_config"]:
        raise ContractError("Checkpoint model config does not match deterministic architecture")
    match = asdict(model.parameter_match) if hasattr(model, "parameter_match") else None
    if "parameter_match" in metadata and metadata["parameter_match"] != match:
        raise ContractError("Checkpoint parameter match receipt changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), metadata


def check_campaign_fit_budget(root, config, *, scope=None):
    fits = []
    observer_fits = 0
    kind = (scope or {}).get("study_kind", "original_light_layers_v1")
    for path in Path(root).rglob("run.json"):
        initial = json.loads(path.read_text())
        if initial.get("smoke", True) or initial.get("config", {}).get("recipe") not in RECIPES:
            continue
        old_kind = initial.get("study_scope", {}).get("study_kind", "original_light_layers_v1")
        observer_fits += old_kind == OBSERVER_STUDY_KIND
        if old_kind == kind and (initial["config"]["recipe"], initial["config"]["seed"]) == (config.recipe, config.seed):
            raise ContractError("A non-smoke fit for this study/recipe/seed already exists; no silent retries")
        final_path = path.with_name("complete.json")
        if not final_path.exists():
            final_path = path.with_name("blocked.json")
        final = json.loads(final_path.read_text()) if final_path.exists() else initial
        # An incomplete orphan reserves its full ceiling rather than pretending
        # it spent zero accelerator time. Wall time is a conservative GPU bound.
        fits.append(float(final.get("elapsed_s", 3600.)))
    if kind == OBSERVER_STUDY_KIND and observer_fits >= 15:
        raise ContractError("Observer scope15-fit budget exhausted; original quality arm budget is not a waiver")
    if len(fits) >= 27 or sum(fits)+config.max_minutes*60 > 36*3600:
        raise ContractError("Campaign27-fit/36-GPU-hour conservative budget exhausted")


def _weights(path, model, metadata, root):
    estimated = sum(p.numel()*p.element_size() for p in model.state_dict().values())+2**20
    campaign_space(root, estimated)
    path = Path(path)
    temporary = path.with_name(path.name+".partial")
    if temporary.exists():
        raise ContractError("Incomplete weight write already exists; no overwrite/resume")
    metadata = {**metadata, "parameter_match":
                asdict(model.parameter_match) if hasattr(model, "parameter_match") else None}
    torch.save({"model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "metadata": metadata, "resume_supported": False}, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def development_loss(model, data, dev_records, transforms, device):
    """Fixed chronological nonoverlapping center tiles, macro-song main loss.

    This bounded checkpoint-selection instrument is NOT a full-song metric.
    Palette rollout initializes in each 64s context; no reference states are
    used. Final full-history metrics must use the separate predict API.
    """
    was_training = model.training
    # Aux outputs unnecessary here. compute_loss tolerates their absence.
    model.eval()
    songs = defaultdict(list)
    try:
        for record in sorted(dev_records, key=lambda r:r.recording_id):
            starts = list(range(0, len(record.times)-639, 80))
            values = []
            for start in starts:
                batch = data.batch([(record.recording_id, start)], device)
                losses, _ = compute_loss(model, batch, transforms, state_dropout=0.)
                values.append(float(losses["main"]))
            songs[record.canonical_id].append(float(np.mean(values)))
    finally:
        model.train(was_training)
    if not songs:
        raise ContractError("No common DEV checkpoint-selection records")
    score = float(np.mean([np.mean(recordings) for recordings in songs.values()]))
    if not np.isfinite(score):
        raise ContractError("Nonfinite development checkpoint loss")
    return score


def train_one(store, records, transforms, output, config, bindings, *, smoke=False,
              device="cuda", gpu_authorized=False, observer_proxy=False):
    config.validate(smoke)
    scope = validate_run_scope(store.manifest, config, smoke=smoke, observer_proxy=observer_proxy)
    if observer_proxy:
        if transforms.get("study_scope") != scope:
            raise ContractError("Observer frozen transforms scope mismatch")
        if not bindings.get("preregistration_sha256") or any(
                bindings.get(key) != scope[key] for key in OBSERVER_SCOPE_BINDINGS):
            raise ContractError("Observer fit needs exact preregistered scope bindings")
        if any(len(record.times) != 640 for record in records):
            raise ContractError("Observer v1 uses exactly64s student feature windows, not variable contexts")
    train_records = [r for r in records if r.partition == "train"]
    dev_records = [r for r in records if r.partition in ("val", "dev")]
    if smoke:
        validate_smoke_selection(store.manifest)
        if dev_records:
            raise ContractError("Smoke cannot read DEV/TEST")
    elif (len({r.canonical_id for r in train_records}) < 80 or
          len({r.canonical_id for r in dev_records}) < 30):
        raise ContractError("Scientific fits require >=80 TRAIN and >=30 DEV canonical families")
    if device.startswith("cuda") and not gpu_authorized:
        raise ContractError("Parent must explicitly authorize the sole GPU job (--gpu-authorized)")
    if not device.startswith("cuda") and not smoke:
        raise ContractError("Full preregistered fits require BF16 CUDA; no silent CPU fallback")
    output = Path(output)
    if output.exists():
        raise ContractError("Run directory already exists: no overwrite/resume of old artifacts")
    campaign_root = Path(store.manifest["campaign_root"]).resolve()
    if not output.resolve().is_relative_to(campaign_root):
        raise ContractError("Run outputs must be inside the accounted campaign root")
    if not smoke:
        check_campaign_fit_budget(campaign_root, config, scope=scope)
    architecture, visual_dim, pose_dim = RECIPES[config.recipe]
    model_config = LightLayersConfig(visual_aux_dim=visual_dim, pose_aux_dim=pose_dim)
    if model_config.context_radius10 > 280:
        raise ContractError("Model context does not fit frozen center loss")
    cuda = device.startswith("cuda")
    # Seed before initialization; sampler/dropout use separate independent RNGs.
    seed_all(config.seed, cuda=False)
    model = build_light_layers(model_config, architecture=architecture)
    parameters = count_trainable_parameters(model)
    estimated_new_bytes = parameters*4*3 + config.steps*2048 + 8*1024**2
    campaign_space(campaign_root, estimated_new_bytes)
    data = TrainingData(store, records, transforms, config.recipe)
    sampler = BatchSampler(train_records, config.seed)
    dropout_generator = torch.Generator(device="cpu").manual_seed(config.seed+10_000)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version":1, "status":"running", "smoke":smoke,
              "scope":("functional_only_no_scientific_success" if smoke else
                       "selected_historical_observer_proxy_only" if observer_proxy else "historically_exposed_development_only"),
              "study_scope":scope,
              "resume_supported":False, "optimizer_checkpoints":False, "bindings":bindings,
              "config":asdict(config), "model_config":asdict(model.cfg),
              "reference_model_config":asdict(model_config), "architecture":architecture,
              "active_trainable_parameters":parameters, "donors":data.donors,
              "donor_statistics":data.donor_statistics(),
              "estimated_peak_new_bytes":estimated_new_bytes, "binding_content_sha256":json_hash(bindings),
              "seed":config.seed, "rng_initial":rng_digest(sampler, dropout_generator, cuda=False),
              "software":{"torch":torch.__version__, "numpy":np.__version__},
              "checkpoint_scalar":"macrocanonical_song_mean_accent_BCE_plus_color_BCE_plus_palette_CE",
              "checkpoint_context":"64s generated-history crops, center8s; no gold history",
              "selection_interval":300, "native_target_hz":10,
              "proxy_event_tolerance_s":.2, "fine_beat_label_claim":False,
              "best_development_main_loss":None, "best_step":None,
              "timing":"audio full-record clock; pose active-span offset added exactly once",
              "model_selection_on_test":False, "frozen_transforms":transforms}
    if hasattr(model, "parameter_match"):
        report["parameter_match"] = asdict(model.parameter_match)
    write_json(output/"run.json", report)
    started = time.monotonic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best = math.inf
    try:
        with gpu_lock(campaign_root):
            if cuda:
                import subprocess
                active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        check=True, capture_output=True, text=True)
                other = [pid.strip() for pid in active.stdout.splitlines()
                         if pid.strip() and pid.strip() != str(os.getpid())]
                if other:
                    raise ContractError("Another GPU compute process is active: "+",".join(other))
                if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                    raise ContractError("CUDA BF16 unavailable")
                torch.cuda.manual_seed_all(config.seed)
                report["rng_initial_cuda"] = rng_digest(sampler, dropout_generator, cuda=True)
            model.to(device)
            model.train()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            for step in range(1, config.steps+1):
                if time.monotonic()-started > config.max_minutes*60:
                    raise ContractError("Per-fit wall budget exceeded; execution blocked, not refuted")
                samples = sampler.sample(config.batch)
                batch = data.batch(samples, device)
                optimizer.zero_grad(set_to_none=True)
                lr = config.lr*min(1., step/config.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16,
                                    enabled=cuda):
                    losses, outputs = compute_loss(model, batch, transforms,
                        state_dropout=config.state_dropout, dropout_generator=dropout_generator)
                if not all(torch.isfinite(loss).item() for loss in losses.values()):
                    raise ContractError("Nonfinite loss")
                losses["total"].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                               error_if_nonfinite=True)
                optimizer.step()
                row = {"step":step, "seed":config.seed, "smoke":smoke,
                       "bindings_sha256":report["binding_content_sha256"], "bindings_file":"run.json",
                       "loss":{key:float(value.detach()) for key,value in losses.items()},
                       "lr":lr, "gradient_norm":float(gradient_norm), "samples":samples,
                       "elapsed_s":time.monotonic()-started}
                if cuda:
                    row["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
                if not smoke and (step % config.check_interval == 0 or step == config.steps):
                    score = development_loss(model, data, dev_records, transforms, device)
                    row["development_main_loss"] = score
                    if score < best:
                        best = score
                        report["best_development_main_loss"], report["best_step"] = score, step
                        _weights(output/"best.pt", model,
                                 {**bindings, "config":asdict(config), "model_config":asdict(model.cfg),
                                  "reference_model_config":asdict(model_config), "architecture":architecture,
                                  "study_scope":scope,
                                  "step":step, "selection_loss":score}, campaign_root)
                # Exact full bindings occur once in run.json. Every row pins
                # their content hash, rather than duplicating file maps 6000x.
                serialized = json.dumps(row, allow_nan=False)
                if len(serialized.encode()) > 2048:
                    raise ContractError("Per-update log exceeds preflight byte reservation")
                with (output/"updates.jsonl").open("a") as stream:
                    stream.write(serialized+"\n")
                print(serialized, flush=True)
                if step % 30 == 0:
                    campaign_space(campaign_root)
                del outputs, losses, batch
            _weights(output/"final.pt", model,
                     {**bindings, "config":asdict(config), "model_config":asdict(model.cfg),
                      "reference_model_config":asdict(model_config), "architecture":architecture,
                      "study_scope":scope,
                      "step":config.steps, "smoke":smoke}, campaign_root)
        report.update(status="complete", elapsed_s=time.monotonic()-started,
                      rng_final=rng_digest(sampler, dropout_generator, cuda=cuda),
                      final_weights_sha256=sha256(output/"final.pt"))
        if (output/"best.pt").exists():
            report["best_weights_sha256"] = sha256(output/"best.pt")
        write_json(output/"complete.json", report)
        return report
    except Exception as error:
        report.update(status="execution_blocked", error=f"{type(error).__name__}: {error}",
                      elapsed_s=time.monotonic()-started)
        write_json(output/"blocked.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("prepare", "train"), default="train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transforms", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--recipe", default="S_VR_P0", choices=tuple(RECIPES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--observer-proxy", action="store_true",
                        help="Separate registered historical observer proxy; never waives original quality gates")
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audio-ram-mib", type=int, default=256)
    parser.add_argument("--target-ram-mib", type=int, default=512)
    args = parser.parse_args(argv)
    config = TrainConfig(recipe=args.recipe, seed=args.seed, steps=args.steps, batch=args.batch)
    config.validate(args.smoke)
    manifest = json.loads(args.manifest.read_text())
    scope = validate_run_scope(manifest, config, smoke=args.smoke, observer_proxy=args.observer_proxy)
    store = SourceStore(manifest, max_ram_bytes=args.audio_ram_mib*1024**2)
    if args.smoke:
        validate_smoke_selection(manifest)
    if args.output.exists():
        raise ContractError("Output already exists; no overwrite/resume")
    if not args.output.resolve().is_relative_to(Path(manifest["campaign_root"]).resolve()):
        raise ContractError("All prepared/run artifacts must stay inside the accounted campaign root")
    # Explicit manifest-wide fit contract, independent of selected arm. Missing
    # pose qualification blocks only pose arms; core targets never change.
    auxiliaries = manifest.get("fit_auxiliaries", ["rich", "legacy"])
    if not set(auxiliaries).issubset({"rich", "legacy", "pose"}) or "rich" not in auxiliaries:
        raise ContractError("fit_auxiliaries must explicitly include rich and optional legacy/pose")
    include_legacy, include_pose = "legacy" in auxiliaries, "pose" in auxiliaries
    if RECIPES[args.recipe][1] == 6 and not include_legacy:
        raise ContractError("V6 recipe requires frozen common legacy auxiliary fit")
    if RECIPES[args.recipe][2] and not include_pose:
        raise ContractError("Pose recipe requires frozen qualified pose validity and normalization")
    records = []; used = 0
    for rid in sorted(store.rows):
        record = store.visual(rid, include_legacy=include_legacy, include_pose=include_pose)
        used += sum(v.nbytes for v in vars(record).values() if isinstance(v, np.ndarray))
        # Target arrays later approximately double this storage.
        if used*2 > args.target_ram_mib*1024**2:
            raise ContractError("Bounded target RAM cap exceeded; stream a smaller frozen cohort")
        records.append(record)
    if args.phase == "prepare":
        bindings = provenance(args.manifest, store, config, scope=scope)
        if not args.smoke:
            verify_preregistration(args.preregistration, bindings, args.recipe, require_fits=False)
        train = [r for r in records if r.partition == "train"]
        transforms = fit_transforms(train,
            audio_rows=(("train", store.audio(r.recording_id)) for r in train))
        bindings = provenance(args.manifest, store, config, transforms, scope=scope)
        transforms["bindings"] = bindings
        transforms["smoke"] = args.smoke
        transforms["study_scope"] = scope
        transforms["fit_support"] = {
            "audio":"all full-source94D bus frames of each usable TRAIN recording; no DEV",
            "audio_fit_recording_count":len(train),
            "audio_fit_frame_count":int(transforms["audio"]["counts"][0]),
            "visual":"selected TRAIN visual windows (observer64s), not only supervisedcenter8s",
            "cached_legacy_v6":"immutable z[y] recording-local value normalization fitted over full active recording before crop",
            "main_supervision":"center8s only"}
        campaign_space(manifest["campaign_root"], 1024**2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, transforms)
        print(json.dumps({"status":"prepared", "smoke":args.smoke, "path":str(args.output),
                          "sha256":sha256(args.output), "bindings":bindings}), flush=True)
        return
    if args.transforms is None:
        raise ContractError("Train requires --transforms from a separate --phase prepare")
    transforms = json.loads(args.transforms.read_text())
    if transforms.get("study_scope") != scope:
        raise ContractError("Prepared transforms study scope differs from explicit run scope")
    bindings = provenance(args.manifest, store, config, transforms, scope=scope)
    bindings["transforms_sha256"] = sha256(args.transforms)
    for key in ("manifest_sha256", "source_sha256", "split_sha256"):
        if transforms.get("bindings", {}).get(key) != bindings[key]:
            raise ContractError(f"Frozen transforms drift: {key}")
    if not args.smoke:
        if transforms.get("smoke"):
            raise ContractError("Smoke transforms cannot be promoted to scientific fits")
        bindings["preregistration_sha256"] = verify_preregistration(
            args.preregistration, bindings, args.recipe)
    train_one(store, records, transforms, args.output, config, bindings,
              smoke=args.smoke, device=args.device, gpu_authorized=args.gpu_authorized,
              observer_proxy=args.observer_proxy)
