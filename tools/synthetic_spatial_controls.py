#!/usr/bin/env python3
"""Generate a provenance-labeled synthetic spatial-control benchmark.

These are analytic test signals, not measurements: no video, DMX capture,
fixture manufacturer data, audio, model, network, or random state is used.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class FixtureLimits:
    pan_min_deg: float = -270.0
    pan_max_deg: float = 270.0
    tilt_min_deg: float = -135.0
    tilt_max_deg: float = 135.0
    max_speed_deg_s: float = 90.0
    max_acceleration_deg_s2: float = 360.0

def requested_trajectory(duration_s: float = 8.0, fps: float = 40.0):
    if duration_s <= 0 or fps <= 0:
        raise ValueError("duration_s and fps must be positive")
    time = np.arange(int(round(duration_s * fps)), dtype=float) / fps
    phase = 2 * np.pi * time
    fast = 2 * np.pi * 3.5 * time
    angles = np.empty((len(time), 3, 2), dtype=float)
    angles[:, 0] = np.column_stack((105 * np.sin(phase), 42 * np.cos(phase)))
    angles[:, 1] = np.column_stack((190 * np.sin(fast), 75 * np.cos(fast)))
    angles[:, 2] = np.column_stack((70 * np.sin(0.5 * phase + 0.4), 35 * np.sin(0.7 * phase)))
    return time, angles

def project_trajectory(requested: np.ndarray, fps: float, limits: FixtureLimits):
    requested = np.asarray(requested, dtype=float)
    if requested.ndim != 3 or requested.shape[1:] != (3, 2):
        raise ValueError("requested must have shape (T,3,2)")
    if fps <= 0:
        raise ValueError("fps must be positive")
    out = np.empty_like(requested)
    lo = np.array([limits.pan_min_deg, limits.tilt_min_deg])
    hi = np.array([limits.pan_max_deg, limits.tilt_max_deg])
    out[0] = np.clip(requested[0], lo, hi)
    velocity = np.zeros((3, 2), dtype=float)
    max_dv = limits.max_acceleration_deg_s2 / fps
    for index in range(1, len(requested)):
        desired_velocity = np.clip((requested[index] - out[index - 1]) * fps,
                                   -limits.max_speed_deg_s, limits.max_speed_deg_s)
        # Brake before a travel limit: v^2 <= 2*a*distance keeps the next
        # discrete samples inside the range without an acceleration spike at
        # the boundary.
        distance_to_hi = np.maximum(hi - out[index - 1], 0.0)
        distance_to_lo = np.maximum(out[index - 1] - lo, 0.0)
        velocity = np.clip(velocity,
                           -np.sqrt(2 * limits.max_acceleration_deg_s2 * distance_to_lo),
                           np.sqrt(2 * limits.max_acceleration_deg_s2 * distance_to_hi))
        desired_velocity = np.where(desired_velocity > 0,
            np.minimum(desired_velocity, np.sqrt(2 * limits.max_acceleration_deg_s2 * distance_to_hi)),
            np.maximum(desired_velocity, -np.sqrt(2 * limits.max_acceleration_deg_s2 * distance_to_lo)))
        velocity += np.clip(desired_velocity - velocity, -max_dv, max_dv)
        out[index] = np.clip(out[index - 1] + np.clip(velocity, -limits.max_speed_deg_s, limits.max_speed_deg_s) / fps, lo, hi)
        velocity = (out[index] - out[index - 1]) * fps
    return out

def audit(angles: np.ndarray, fps: float, limits: FixtureLimits):
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 3 or angles.shape[1:] != (3, 2) or len(angles) < 3:
        raise ValueError("angles must have shape (T>=3,3,2)")
    finite = bool(np.isfinite(angles).all())
    speed = np.diff(angles, axis=0) * fps
    acceleration = np.diff(angles, n=2, axis=0) * fps * fps
    lo = np.array([limits.pan_min_deg, limits.tilt_min_deg])
    hi = np.array([limits.pan_max_deg, limits.tilt_max_deg])
    in_range = bool(((angles >= lo) & (angles <= hi)).all())
    speed_ok = bool((np.abs(speed) <= limits.max_speed_deg_s + 1e-9).all())
    acceleration_ok = bool((np.abs(acceleration) <= limits.max_acceleration_deg_s2 + 1e-9).all())
    return {
        "finite": finite, "in_range": in_range, "speed_ok": speed_ok,
        "acceleration_ok": acceleration_ok,
        "physically_admissible": bool(finite and in_range and speed_ok and acceleration_ok),
        "max_component_speed_deg_s": float(np.abs(speed).max()),
        "max_component_acceleration_deg_s2": float(np.abs(acceleration).max()),
        "max_vector_speed_deg_s": float(np.linalg.norm(speed, axis=-1).max()),
        "max_vector_acceleration_deg_s2": float(np.linalg.norm(acceleration, axis=-1).max()),
        "frames": int(len(angles)), "fps": float(fps),
    }

def generate(duration_s: float = 8.0, fps: float = 40.0, limits: FixtureLimits | None = None):
    limits = limits or FixtureLimits()
    time, requested = requested_trajectory(duration_s, fps)
    feasible = project_trajectory(requested, fps, limits)
    return {
        "time_s": time, "requested_fast_deg": requested,
        "feasible_projected_deg": feasible,
        "requested_audit": audit(requested, fps, limits),
        "feasible_audit": audit(feasible, fps, limits),
        "limits": asdict(limits),
    }

def write_dataset(out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = generate()
    npz_path = out / "spatial_controls_v1.npz"
    manifest_path = out / "MANIFEST.json"
    np.savez_compressed(npz_path, time_s=result["time_s"],
                        requested_fast_deg=result["requested_fast_deg"],
                        feasible_projected_deg=result["feasible_projected_deg"])
    manifest = {
        "schema": "lightsem.synthetic_spatial_controls.v1",
        "provenance": {
            "kind": "synthetic_analytic_control",
            "source": "This repository script only",
            "contains_original_dmx": False,
            "contains_video_extracted_trajectory": False,
            "contains_fixture_capture": False,
            "purpose": "Regression and mechanical-feasibility audit; not a training corpus.",
        },
        "generation": {
            "command": "python tools/synthetic_spatial_controls.py --out-dir <DIR>",
            "deterministic": True, "audio_used": False, "model_used": False,
            "random_seed": None,
        },
        "fixtures": {"count": 3, "axes": ["pan", "tilt"], "units": "degrees"},
        "limits": result["limits"],
        "streams": {
            "requested_fast_deg": result["requested_audit"],
            "feasible_projected_deg": result["feasible_audit"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return npz_path, manifest_path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic/spatial_controls_v1"))
    args = parser.parse_args()
    npz_path, manifest_path = write_dataset(args.out_dir)
    print(json.dumps({"npz": str(npz_path), "manifest": str(manifest_path)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
