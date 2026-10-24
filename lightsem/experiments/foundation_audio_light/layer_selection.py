"""CPU layer-selection probes for authenticated MuQ feature families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...eval.music_motion import feature_family_predictive_gain
from .muq import MuQContract, validate_base92, validate_muq_features


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    values: np.ndarray
    time: np.ndarray
    provenance: Mapping[str, Any] | None = None


def effective_rank(values: Any) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("values must be T,D")
    x = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    p = energy[energy > 1e-12] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def geometry(values: Any, *, constant_tolerance: float = 1e-8) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("values must be finite T,D")
    sd = x.std(0)
    return {
        "finite": True,
        "frames": int(x.shape[0]),
        "dimension": int(x.shape[1]),
        "effective_rank": effective_rank(x),
        "constant_channel_fraction": float(np.mean(sd <= constant_tolerance)),
        "channel_std_p50": float(np.percentile(sd, 50)),
        "channel_std_p95": float(np.percentile(sd, 95)),
        "temporal_delta_rms": float(np.sqrt(np.mean(np.diff(x, axis=0) ** 2))) if len(x) > 1 else 0.0,
        "scope": "frozen feature geometry; not a conditioning verdict",
    }


def validate_family(family: FeatureFamily, *, contract: MuQContract = MuQContract()) -> FeatureFamily:
    values = np.asarray(family.values)
    time = np.asarray(family.time, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(time) or len(time) < 2:
        raise ValueError(f"{family.name}: values/time shape mismatch")
    if not np.isfinite(values).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"{family.name}: nonfinite or non-monotone data")
    lower = family.name.lower()
    if lower == "base92":
        values = validate_base92(values, contract=contract)
    elif lower == "muq_plus_base92":
        expected = contract.dimension + contract.base92_dimension
        if values.shape[1] != expected:
            raise ValueError(f"{family.name} must have dimension {expected}")
    elif lower.startswith("muq"):
        values = validate_muq_features(values, contract=contract, name=family.name)
    return FeatureFamily(family.name, values.astype(np.float32, copy=False), time, family.provenance)


def compare_feature_families(
    families: list[FeatureFamily],
    *,
    target_time: Any,
    target: Any,
    groups: Any | None = None,
    contract: MuQContract = MuQContract(),
    **probe_kwargs: Any,
) -> dict[str, Any]:
    """Return geometry and grouped predictive probes for every family."""
    target_time = np.asarray(target_time, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target_time.ndim != 1 or target.ndim != 1 or len(target_time) != len(target):
        raise ValueError("target_time and target must be aligned 1-D arrays")
    result: dict[str, Any] = {"families": {}, "selection": {"metric": "incremental_gain"}}
    gains = {}
    for family in families:
        family = validate_family(family, contract=contract)
        probe = feature_family_predictive_gain(
            family.time, family.values, target_time, target, groups=groups, **probe_kwargs
        )
        result["families"][family.name] = {
            "geometry": geometry(family.values),
            "probe": probe,
            "provenance": dict(family.provenance or {}),
        }
        gains[family.name] = probe.get("incremental_gain")
    valid = [(name, value) for name, value in gains.items() if value is not None and np.isfinite(value)]
    result["selection"]["ranked_by_gain"] = [name for name, _ in sorted(valid, key=lambda row: row[1], reverse=True)]
    result["selection"]["selected_reference"] = "MuQ_layer9"
    result["selection"]["policy"] = (
        "Layer ranking is diagnostic. Keep the pinned MuQ layer9 baseline unless a candidate "
        "wins the grouped probe and all downstream conditioning/safety gates."
    )
    return result


def load_probe_npz(path: str) -> tuple[list[FeatureFamily], np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load a compact CPU probe file.

    Required arrays are ``target_time`` and ``target``. Feature arrays are
    ``base92`` and/or ``muq_layer_<n>`` with a shared ``time`` array. Optional
    ``groups`` enables song/group-disjoint folds.
    """
    with np.load(path, allow_pickle=False) as data:
        required = {"time", "target_time", "target"}
        if not required.issubset(data.files):
            raise ValueError("probe NPZ requires time, target_time and target")
        time = np.asarray(data["time"], dtype=np.float64)
        target_time = np.asarray(data["target_time"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        groups = np.asarray(data["groups"]) if "groups" in data.files else None
        families = []
        if "base92" in data.files:
            families.append(FeatureFamily("Base92", np.asarray(data["base92"]), time))
        for key in sorted(x for x in data.files if x.startswith("muq_layer_")):
            layer = key.removeprefix("muq_layer_")
            families.append(FeatureFamily(f"MuQ_layer{layer}", np.asarray(data[key]), time))
        if "muq_plus_base92" in data.files:
            families.append(FeatureFamily("MuQ_plus_Base92", np.asarray(data["muq_plus_base92"]), time))
        meta = {}
        if "meta" in data.files:
            meta = json.loads(str(data["meta"].item()))
    return families, target_time, target, groups, meta

