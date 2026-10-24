#!/usr/bin/env python3
"""Audit raw audio-only foundation-v3 aim streams before mapping or rendering.

The v3 head emits a native 2-D angular aim ``(T, fixtures, 2)`` rather than a
3-D body.  This audit therefore keeps two representations explicit:

* direct angle finite differences are the binding safety check;
* the same angles are converted to unit 3-D ray directions only so the shared
  training-health ``motion_metrics`` can report a comparable T,J,3 diagnostic.

No smoothing, clamping, retiming, interpolation, mapper, DMX, or video path is
used here.  The input prediction files and sealed training receipt are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lightsem.eval.training_health import motion_metrics


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return {"count": 0, "min": None, "max": None, "p50": None, "p95": None, "p99": None}
    q = np.percentile(finite, [50, 95, 99])
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p50": float(q[0]),
        "p95": float(q[1]),
        "p99": float(q[2]),
    }


def effective_rank(value: np.ndarray) -> float:
    x = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1])
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    return float((total * total) / max(float((energy * energy).sum()), 1e-12)) if total else 0.0


def angles_to_directions(angle: np.ndarray) -> np.ndarray:
    pan = angle[..., 0] * np.pi
    tilt = angle[..., 1] * np.pi / 2.0
    direction = np.stack(
        (
            np.sin(pan) * np.cos(tilt),
            -np.cos(pan) * np.cos(tilt),
            np.sin(tilt),
        ),
        axis=-1,
    )
    return direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)


def duplicate_groups(streams: np.ndarray) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for fixture in range(streams.shape[1]):
        digest = hashlib.sha256(np.ascontiguousarray(streams[:, fixture]).tobytes()).hexdigest()
        groups.setdefault(digest, []).append(fixture)
    return [members for members in groups.values() if len(members) > 1]


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"time", "aim", "rgb", "brightness", "event", "audio_latent"}
        missing = sorted(required.difference(bundle.files))
        if missing:
            raise ValueError(f"{path} missing required arrays: {missing}")
        arrays = {key: np.asarray(bundle[key]) for key in required}
    aim = arrays["aim"]
    if aim.ndim == 4 and aim.shape[0] == 1:
        aim = aim[0]
    if aim.ndim != 3 or aim.shape[-1] != 2:
        raise ValueError(f"{path} aim must be (T,fixtures,2) or (1,T,fixtures,2), got {aim.shape}")
    for key in ("rgb", "brightness", "event", "audio_latent"):
        value = arrays[key]
        if value.ndim >= 1 and value.shape[0] == 1:
            arrays[key] = value[0]
    arrays["aim"] = aim
    return arrays


def audit_one(path: Path, fps: float, limits: dict[str, float], output_dir: Path) -> dict[str, Any]:
    arrays = load_prediction(path)
    aim = arrays["aim"].astype(np.float64, copy=False)
    if aim.shape[0] < 4:
        raise ValueError(f"{path} requires at least four frames for jerk")
    velocity = np.diff(aim, axis=0) * fps
    acceleration = np.diff(aim, n=2, axis=0) * (fps**2)
    jerk = np.diff(aim, n=3, axis=0) * (fps**3)
    speed_norm = np.linalg.norm(velocity, axis=-1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    directions = angles_to_directions(aim)
    label = path.stem.removeprefix("PREDICTIONS_")
    direction_path = output_dir / f"DIRECTION_{label}.npz"
    np.savez_compressed(direction_path, time=np.asarray(arrays["time"]), body=directions.astype(np.float32),
                        angle=aim.astype(np.float32))
    direction_health = motion_metrics(directions, fps=fps)
    latent = arrays["audio_latent"].astype(np.float64, copy=False)
    latent_flat = latent.reshape(-1, latent.shape[-1])
    rgb = arrays["rgb"].astype(np.float64, copy=False)
    brightness = arrays["brightness"].astype(np.float64, copy=False)
    event = arrays["event"].astype(np.float64, copy=False)
    duplicate = duplicate_groups(aim)
    raw = {
        "finite": bool(all(np.isfinite(arrays[key]).all() for key in ("aim", "rgb", "brightness", "event", "audio_latent"))),
        "shape": list(aim.shape),
        "fps": float(fps),
        "angle_abs": finite_stats(np.abs(aim)),
        "speed_norm": finite_stats(speed_norm),
        "acceleration_norm": finite_stats(acceleration_norm),
        "jerk_norm": finite_stats(jerk_norm),
        "speed_component_abs": finite_stats(np.abs(velocity)),
        "acceleration_component_abs": finite_stats(np.abs(acceleration)),
        "jerk_component_abs": finite_stats(np.abs(jerk)),
        "latent_effective_rank": effective_rank(latent_flat),
        "latent_constant_channel_fraction": float(np.mean(latent_flat.std(axis=0) <= 1e-8)),
        "latent_temporal_delta_rms": float(np.sqrt(np.mean(np.diff(latent_flat, axis=0) ** 2))),
        "rgb_std": float(rgb.std()),
        "brightness_std": float(brightness.std()),
        "event_logit_std": float(event.std()),
        "exact_duplicate_fixture_groups": duplicate,
        "direction_representation": {
            "path": str(direction_path.resolve()),
            "sha256": sha(direction_path),
            "shape": list(directions.shape),
            "unit_norm_max_error": float(np.max(np.abs(np.linalg.norm(directions, axis=-1) - 1.0))),
            "motion_health": direction_health,
        },
    }
    raw["raw_angle_safety_pass"] = bool(
        raw["finite"]
        and raw["speed_norm"]["p99"] <= limits["max_velocity"]
        and raw["acceleration_norm"]["p99"] <= limits["max_acceleration"]
        and raw["jerk_norm"]["p99"] <= limits["max_jerk"]
    )
    raw["raw_angle_failures"] = [
        name for name, observed, limit in (
            ("speed_p99", raw["speed_norm"]["p99"], limits["max_velocity"]),
            ("acceleration_p99", raw["acceleration_norm"]["p99"], limits["max_acceleration"]),
            ("jerk_p99", raw["jerk_norm"]["p99"], limits["max_jerk"]),
        ) if observed is None or observed > limit
    ]
    raw["output_variation"] = {
        "rgb_nonconstant": bool(raw["rgb_std"] > 0.01),
        "brightness_nonconstant": bool(raw["brightness_std"] > 0.01),
        "event_nonconstant": bool(raw["event_logit_std"] > 0.01),
        "exact_duplicate_fixture_groups": duplicate,
    }
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, help="LABEL=prediction.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-velocity", type=float, default=0.18)
    parser.add_argument("--max-acceleration", type=float, default=0.5)
    parser.add_argument("--max-jerk", type=float, default=1.5)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing raw audit root: {args.out}")
    pairs: dict[str, Path] = {}
    for item in args.prediction:
        if "=" not in item:
            raise SystemExit(f"--prediction expects LABEL=PATH, got {item!r}")
        label, value = item.split("=", 1)
        if label in pairs:
            raise SystemExit(f"duplicate prediction label: {label}")
        pairs[label] = Path(value)
    args.out.mkdir(parents=True, exist_ok=False)
    training = json.loads(args.training.read_text())
    limits = {"max_velocity": args.max_velocity, "max_acceleration": args.max_acceleration, "max_jerk": args.max_jerk}
    songs = {label: audit_one(path, args.fps, limits, args.out) for label, path in pairs.items()}
    failures = {label: row["raw_angle_failures"] for label, row in songs.items() if not row["raw_angle_safety_pass"]}
    parameterized = {
        row.get("id"): row for row in training.get("validation_audits", []) if isinstance(row, dict) and row.get("id")
    }
    report = {
        "schema": "lightsem.foundation_v3_raw_audit.v1",
        "protocol_id": "foundation-v3-raw-angle-audit-20261020-v1",
        "status": "failed_raw_safety" if failures else "diagnostic_candidate",
        "scope": "raw 2-D angular aim before mapper/interpolation/DMX/video; no posthoc correction",
        "source_training": str(args.training.resolve()),
        "source_training_sha256": sha(args.training),
        "source_predictions": {label: {"path": str(path.resolve()), "sha256": sha(path)} for label, path in pairs.items()},
        "fps": float(args.fps),
        "limits": limits,
        "parameterized_safety_source": "TRAINING.json validation_audits parameterized_* fields; component-wise, not finite-difference vector norms",
        "parameterized_vs_raw": {
            label: {
                "parameterized_raw_safety_pass": parameterized.get(label, {}).get("raw_safety_pass"),
                "parameterized_speed_component_p99": parameterized.get(label, {}).get("parameterized_speed_component_p99"),
                "parameterized_acceleration_component_p99": parameterized.get(label, {}).get("parameterized_acceleration_component_p99"),
                "parameterized_jerk_component_p99": parameterized.get(label, {}).get("parameterized_jerk_component_p99"),
                "raw_angle_safety_pass": row["raw_angle_safety_pass"],
                "raw_angle_failures": row["raw_angle_failures"],
            }
            for label, row in songs.items()
        },
        "songs": songs,
        "raw_safety_failures": failures,
        "promotion": "forbidden; raw finite-difference safety precedes mapper, decoded DMX, and video",
        "missing_evidence": [
            "native decoded DMX stream",
            "independent own-song recomputation",
            "complete feature/audio provenance for this validation-only checkpoint",
            "human review",
        ],
    }
    output = args.out / "RAW_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"out": str(output), "status": report["status"], "raw_safety_failures": failures}, sort_keys=True))


if __name__ == "__main__":
    main()
