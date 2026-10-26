#!/usr/bin/env python3
"""Cached-body limb intervention through mapper-01 and decoded angular DMX.

This is a source-transfer diagnostic. It uses sealed selected-C bodies, P2
RGB, and loudness caches, then stops at decoded DMX. Missing raw waveforms make
the receipt diagnostic_only; no video or product promotion is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_loudness(receipt: Path) -> dict[str, np.ndarray]:
    report = json.loads(receipt.read_text())
    npz = Path(report["path"])
    if not npz.exists():
        mirror = Path("/mnt/lightsem-data/experiments/lightsem") / npz.relative_to("/root/lightsem/experiments")
        if mirror.exists():
            npz = mirror
    if not npz.exists():
        raise FileNotFoundError(npz)
    if sha(npz) != report["sha256"]:
        raise ValueError("loudness cache hash mismatch")
    with np.load(npz, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ("time", "momentary", "short_term")}


def circular_deg(delta: np.ndarray) -> np.ndarray:
    return np.abs((delta + 180.0) % 360.0 - 180.0)


def run_song(args, song: str) -> dict:
    from lightsem.conductor.body_transfer import decode, sample_body
    from lightsem.conductor.body_transfer_v2 import make_outputs
    from lightsem.experiments.dynamic_dance.contracts import DanceResult
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world

    body_path = Path(args.body_root) / song / "body.npz"
    color_path = Path(args.color_root) / song / "P2.npz"
    loudness_path = Path(args.loudness_root) / song / "loudness.json"
    body = DanceResult.load(body_path)
    world = np.asarray(body.world, dtype=np.float64)
    time40, world40 = sample_body(world, source_fps=30.0, target_fps=40.0)
    with np.load(color_path, allow_pickle=False) as z:
        left = np.asarray(z["left"], dtype=np.uint8)[: len(time40)]
        right = np.asarray(z["right"], dtype=np.uint8)[: len(time40)]
    if len(left) < len(time40) and len(time40) - len(left) <= 1:
        time40, world40 = time40[:len(left)], world40[:len(left)]
    if len(left) != len(time40):
        raise ValueError(f"P2 cache does not cover body interval: {song}")
    loudness = load_loudness(loudness_path)
    with open(args.config) as f:
        config = json.load(f)
    counts = config["lighting"]["counts"]
    outputs = make_outputs(**counts)
    transfer = dict(config["lighting"]["transfer"])
    transfer.pop("mapper", None)
    # A mapper-01 simulation is explicitly limits-off in the selected config.
    transfer["apply_motion_limits"] = False
    transfer["readout"] = "distributed_gain"

    baseline_dmx, baseline_patch, baseline_intent, _, baseline_hard = compile_world(
        world40, time40, outputs, left, right, loudness, transfer, mapper_id="01")
    patched = world40.copy()
    root = patched[:, :1]
    rel = patched - root
    # Anatomical left-arm intervention: preserve root and all other joints.
    rel[:, 18, 0] += 0.15
    rel[:, 20, 0] += 0.25
    patched = rel + root
    patched_dmx, patched_patch, patched_intent, _, patched_hard = compile_world(
        patched, time40, outputs, left, right, loudness, transfer, mapper_id="01")
    baseline_decoded = decode(baseline_dmx, baseline_patch)
    patched_decoded = decode(patched_dmx, patched_patch)
    names = [o["name"] for o in outputs]
    regions = [o.get("body_region") for o in outputs]
    moving = [i for i, o in enumerate(outputs) if o["kind"] in ("beam", "laser")]
    assigned_regions = {"left_wrist", "left_elbow", "left_shoulder"}
    assigned = [i for i in moving if regions[i] in assigned_regions]
    unassigned = [i for i in moving if regions[i] != "left_arm"]
    angle_delta = circular_deg(patched_decoded["angles_deg"] - baseline_decoded["angles_deg"])
    pan_delta = angle_delta[..., 0]
    tilt_delta = angle_delta[..., 1]
    angular = np.sqrt(pan_delta ** 2 + tilt_delta ** 2)
    def mean_for(indices):
        return float(np.sqrt(np.mean(angular[:, indices] ** 2))) if indices else 0.0
    rec = {
        "schema_version": 1,
        "song": song,
        "status": "diagnostic_only",
        "qualification_status": "unqualified_missing_raw_waveform",
        "source_body": str(body_path),
        "source_body_sha256": sha(body_path),
        "p2_path": str(color_path),
        "p2_sha256": sha(color_path),
        "loudness_receipt": str(loudness_path),
        "loudness_receipt_sha256": sha(loudness_path),
        "mapper": "01_distributed_gain",
        "counts": counts,
        "fps": 40.0,
        "frames": int(len(time40)),
        "limb_patch": {"region": "left_arm", "joint_offsets_x_m": {"18": 0.15, "20": 0.25}},
        "assigned_fixture_names": [names[i] for i in assigned],
        "unassigned_fixture_names": [names[i] for i in unassigned],
        "assigned_angular_rms_deg": mean_for(assigned),
        "unassigned_angular_rms_deg": mean_for(unassigned),
        "assigned_to_unassigned_ratio": mean_for(assigned) / max(mean_for(unassigned), 1e-12),
        "rgb_rms": float(np.sqrt(np.mean((patched_decoded["rgb"] - baseline_decoded["rgb"]) ** 2))),
        "brightness_rms": float(np.sqrt(np.mean((patched_decoded["dimmer"] - baseline_decoded["dimmer"]) ** 2))),
        "exact_dmx_equal": bool(np.array_equal(baseline_dmx, patched_dmx)),
        "baseline_hard_checks": baseline_hard,
        "patched_hard_checks": patched_hard,
        "raw_waveform_present": False,
        "scope": "cached selected-C body -> mapper-01 -> decoded angular DMX; no video",
    }
    out = Path(args.out) / song
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "limb_patch.npz", time=time40, baseline_dmx=baseline_dmx,
                        patched_dmx=patched_dmx, angular_delta=angular,
                        baseline_angles_deg=baseline_decoded["angles_deg"],
                        patched_angles_deg=patched_decoded["angles_deg"])
    (out / "RECEIPT.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-root", required=True, type=Path)
    p.add_argument("--color-root", required=True, type=Path)
    p.add_argument("--loudness-root", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--songs", default="shpongle,cure_for_me")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    rows = [run_song(a, song.strip()) for song in a.songs.split(",") if song.strip()]
    (a.out / "PORTFOLIO_RECEIPT.json").write_text(json.dumps({"status": "diagnostic_only", "songs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
