#!/usr/bin/env python3
"""Run the SSL latent light adapter on saved full-song native baseline data.

This is the deterministic integration experiment behind the opt-in CLI flag.
It consumes already sealed body/DMX/timeline artifacts and writes candidate DMX
plus raw-safety receipts; it never substitutes a waveform or generates motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lightsem.experiments.body_light_ssl.adapter import SSLAdapter


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_song(adapter: SSLAdapter, song: str, baseline_root: Path, native_root: Path,
             out_root: Path, device: torch.device) -> dict:
    source = baseline_root / song
    with np.load(source / "body.npz", allow_pickle=False) as values:
        body = np.asarray(values["world"], dtype=np.float32)
    with np.load(source / "lights.npz", allow_pickle=False) as values:
        time40 = np.asarray(values["time"], dtype=np.float64)
        universes = np.asarray(values["dmx"], dtype=np.uint8)
        controls = {key: np.asarray(values[key]) for key in ("rgb", "dimmer", "angles_deg", "directions")}
    patch = json.loads((source / "patch.json").read_text())
    with np.load(native_root / "demo_features/features/timelines" / f"{song}.npz", allow_pickle=False) as values:
        music = SimpleNamespace(muq=values["muq"], muq_time=values["muq_time"])
    base92 = native_root / "demo_features/features/base92" / f"{song}.npz"
    candidate, report = adapter.apply(
        body_world=body, music=music, base92_path=base92, controls=controls,
        universes=universes, patch=patch, time40=time40,
        color_active40=None, device=device)
    out = out_root / song
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "lights_ssl.npz"
    np.savez_compressed(candidate_path, time=time40, dmx=candidate)
    report = {**report, "song": song, "baseline_root": str(source),
              "baseline_lights_sha256": sha(source / "lights.npz"),
              "candidate_lights_sha256": sha(candidate_path),
              "body_sha256": sha(source / "body.npz"),
              "patch_sha256": sha(source / "patch.json"),
              "audio_features": str(native_root / "demo_features/features/timelines" / f"{song}.npz")}
    write_json(out / "REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True,
                        help="Root containing song/body.npz, lights.npz and patch.json")
    parser.add_argument("--native-root", type=Path, required=True,
                        help="Native feature root containing demo_features")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"preserving existing experiment root: {args.out}")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    adapter = SSLAdapter.load(args.checkpoint, device)
    rows = {song: run_song(adapter, song, args.baseline_root, args.native_root, args.out, device)
            for song in ("shpongle", "cure_for_me")}
    write_json(args.out / "INTEGRATION.json", {
        "schema_version": 1, "status": "diagnostic_only",
        "scope": "full saved native baseline body/DMX with causal SSL latent light adapter",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
        "device": str(device), "songs": rows,
        "controls_fixed": ["P2 RGB", "restored LUFS dimmer"],
        "body_generation_changed": False, "promotion": "forbidden",
    })


if __name__ == "__main__":
    main()
