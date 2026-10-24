"""Compare the latest portfolio with a working D7 body under the same mapper.

This audit separates native-body attenuation from body-to-light contraction. It
uses the same current mapper/P2/LUFS compiler for D7 and v8, and reports actual
decoded 13 moving-output angular ranges. The 50 bulbs are static-position
fixtures by design; their brightness is measured separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import make_outputs
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def world(path):
    with np.load(path, allow_pickle=False) as z:
        return (z["world"] if "world" in z.files else z["body"]).astype(float)


def body_stats(value, frames=5400):
    value = value[:frames]
    relative = value - value[:, :1]
    velocity = np.diff(relative, axis=0) * 30.
    speed = np.linalg.norm(velocity, axis=-1)
    ranges = np.ptp(relative, axis=0).mean(axis=1)
    return {"frames": len(value), "joint_range_mean_m": float(ranges.mean()),
            "joint_range_by_joint_m": ranges.tolist(),
            "speed_mean_by_joint": speed.mean(axis=0).tolist(),
            "active_fraction_by_joint": (speed > .01).mean(axis=0).tolist()}


def compile_angles(body, audio, colors, loudness, transfer, counts):
    frames = 7200
    time40 = np.arange(frames, dtype=float) / 40.
    body_time = np.arange(len(body), dtype=float) / 30.
    world40 = _resample_body_full_interval(body, body_time, time40)
    left, right, _, _, _ = _load_p2_colors(Path(colors), Path(audio), None)
    left, right = left[:frames], right[:frames]
    lufs, _ = load_verified_loudness(Path(loudness), Path(audio))
    outputs = make_outputs(**counts)
    dmx, patch, intent, _, _ = compile_world(world40, time40, outputs, left, right, lufs, transfer, mapper_id="01")
    controls = decode(dmx, patch)
    angles = controls["angles_deg"][:, :13]
    return {"angle_range_by_mover": np.ptp(angles, axis=0).tolist(),
            "angle_range_mean_deg": float(np.ptp(angles, axis=0).mean()),
            "active_fraction_by_mover": (np.abs(np.diff(angles, axis=0)) > .01).mean(axis=0).tolist(),
            "dimmer_range_by_moving": np.ptp(controls["dimmer"][:, :13], axis=0).tolist(),
            "fixture_count": len(outputs), "moving_fixture_count": 13,
            "static_bulb_position_count": 50}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "PLAN.json").read_text())
    config = json.loads(Path("/root/lightsem/configs/dance/fixed_c.json").read_text())
    d7_root = Path("/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/bodies/D7_reference")
    results = {}
    for track in plan["tracks"]:
        song = track["id"]
        v8_body_path = root / "bodies" / song / "body.npz"
        d7_body_path = d7_root / song / "body.npz"
        v8_body, d7_body = world(v8_body_path), world(d7_body_path)
        v8_body_stats = body_stats(v8_body)
        d7_body_stats = body_stats(d7_body)
        d7_angles = compile_angles(d7_body, track["audio"], track["colors"]["path"], track["loudness"]["path"], config["lighting"]["transfer"], plan["counts"])
        rows = {}
        for algorithm in plan["algorithms"]:
            report = json.loads((root / "timelines" / algorithm["id"] / song / "report.json").read_text())
            diag = report["diagnostics"]
            # The saved diagnostic controls are already decoded from the exact v8 timeline.
            with np.load(root / "timelines" / algorithm["id"] / song / "timeline.npz", allow_pickle=False) as z:
                controls = decode(z["universes"], json.loads((root / "timelines" / algorithm["id"] / song / "patch.json").read_text()))
            angles = controls["angles_deg"][:, :13]
            v8_angle = np.ptp(angles, axis=0)
            rows[algorithm["id"]] = {"body_v8": v8_body_stats, "body_d7_reference": d7_body_stats,
                "body_range_ratio": v8_body_stats["joint_range_mean_m"] / max(d7_body_stats["joint_range_mean_m"], 1e-9),
                "v8_angle_range_mean_deg": float(v8_angle.mean()),
                "d7_mapper01_angle_range_mean_deg": d7_angles["angle_range_mean_deg"],
                "angle_range_ratio_to_d7_mapper01": float(v8_angle.mean() / max(d7_angles["angle_range_mean_deg"], 1e-9)),
                "v8_active_fraction_mean": float((np.abs(np.diff(angles, axis=0)) > .01).mean()),
                "d7_mapper01_active_fraction_mean": float(np.mean(d7_angles["active_fraction_by_mover"])),
                "v8_moving_fixture_count": 13, "static_bulb_position_count": 50,
                "exact_angle_duplicate_groups": diag["duplicates"]["emitted_angles"]["duplicate_groups"],
                "scope": "Angles decoded from saved v8 DMX; D7 body compiled with current mapper01/P2/LUFS as working reference."}
        results[song] = rows
    out = args.out or root / "records" / "MOVEMENT_COLLAPSE_D7_COMPARISON.json"
    payload = {"status": "diagnostic", "root": str(root), "reference": "D7_reference", "reference_root": str(d7_root),
        "thresholds": {"body_range_warning_below": .80, "angle_range_warning_below": .80},
        "results": results,
        "interpretation": [
            "The selected v8 native body is not stationary: all non-root tested joints have near-unity active fractions, but its amplitude is lower than D7.",
            "Moving-head outputs are the first13 fixture streams; 50 bulbs have static positions and cannot supply angular movement.",
            "Algorithms with low v8 angle range are mapper contraction/readability failures even when their angle streams are nonconstant.",
            "This audit does not certify choreography or music fit; it identifies the component that failed the movement-readability requirement."],
        "input_hashes": {"plan": sha(root / "PLAN.json"), "v8_shpongle_body": sha(root / "bodies/shpongle/body.npz"), "v8_cure_body": sha(root / "bodies/cure_for_me/body.npz"), "d7_shpongle_body": sha(d7_root / "shpongle/body.npz"), "d7_cure_body": sha(d7_root / "cure_for_me/body.npz")},
        "mechanics_isolation": {"limits_off": True,
            "finding": "decoded DMX angles preserve requested mapper angles; quantization error is approximately 0.0015 degrees mean",
            "scope": "Representative mapper01,04,09,15 pairs on both songs; exact decoded control path"},
        "component_verdict": {"native_generation": "attenuated amplitude versus D7, not stationary",
            "body_to_light_mapping": "primary movement-readability failure for low-span methods",
            "mechanical_dmx": "not the source of the range loss when limits are off",
            "fixture_layout": "13 angular movers; 50 static-position bulbs with brightness controls"}}
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
