#!/usr/bin/env python3
"""Map and render a raw-safe Fourier candidate through the native renderer.

This command is intentionally diagnostic-only.  It refuses an incomplete or
non-finite candidate and keeps every song/condition in a fresh artifact root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import load_config
from lightsem.dance.lighting import map_lights
from lightsem.dance.render import render_video
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.dance.pipeline import write_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    candidate = args.candidate_root.expanduser().resolve()
    body_path = candidate / f"{args.song}.npz"
    if not body_path.is_file():
        raise FileNotFoundError(body_path)
    audio = args.audio.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"preserving existing render root: {out}")
    with np.load(body_path, allow_pickle=False) as values:
        time = values["time"].astype(np.float64)
        world = values["body"].astype(np.float32)
        native = values["native"].astype(np.float32)
    if (world.ndim != 3 or world.shape[1:] != (22, 3) or len(time) != len(world)
            or len(native) != len(world) or not np.isfinite(world).all()
            or not np.isfinite(native).all()):
        raise ValueError("candidate body is non-finite or has an invalid native shape")
    audio_hash = sha(audio)
    body = DanceResult(
        time=time,
        world=world,
        contacts=np.zeros((len(world), 4), dtype=np.float32),
        native=native,
        metadata={
            "audio": {"audio_sha256": audio_hash},
            "d4max_enabled": False,
            "d6_enabled": False,
            "newly_generated_motion": True,
            "diagnostic_only": True,
            "display_title": "Fixed C | Local Fourier diagnostic",
        },
    )
    out.mkdir(parents=True, exist_ok=False)
    body.save(out / "body.npz")
    # The native renderer charts the declared plan clock.  The stereo showcase
    # generator stores planner tensors under plan_* keys but omits the compact
    # plan.npz envelope expected by the renderer; preserve that exact clock and
    # mark effort as the neutral modifier used by this diagnostic.
    np.savez_compressed(
        out / "plan.npz",
        time=time,
        source_time=time,
        effort=np.ones((len(time), 1), dtype=np.float32),
    )
    generation = {
        "schema_version": 1,
        "status": "finite_body_ready",
        "diagnostic_only": True,
        "candidate": "local_fourier_dual_cross_attention_corrected",
        "song": args.song,
        "audio": str(audio),
        "audio_sha256": audio_hash,
        "candidate_source_body": str(body_path),
        "candidate_source_body_sha256": sha(body_path),
        "body_sha256": sha(out / "body.npz"),
        "body_fps": 30,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha(args.checkpoint) if args.checkpoint else None,
        "source_body_used": False,
        "d4max": False,
        "d6": False,
        "raw_safety_gate": "passed before mapping; see candidate MANIFEST.json",
        "musical_superiority_claim": False,
    }
    write_json(out / "GENERATION.json", generation)
    config, _ = load_config()
    lighting = map_lights(body, audio, out, config)
    if lighting[3].get("source_dependence_checks", {}).get("status") != "pass":
        raise RuntimeError("source-dependence gate failed; render refused")
    output = out / f"{args.song}_local_fourier_{args.mode}.mp4"
    render_video(body, out, audio, output, lighting, duration=args.duration)
    return {
        "status": "diagnostic_only",
        "song": args.song,
        "mode": args.mode,
        "video": str(output),
        "video_sha256": sha(output),
        "run": str(out),
        "source_dependence": lighting[3]["source_dependence_checks"]["status"],
        "duration_s": args.duration,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--song", choices=("shpongle", "cure_for_me"), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args(argv)
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
