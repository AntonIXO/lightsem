#!/usr/bin/env python3
"""Render short, review-only videos from saved stereo-adapter outputs.

The script stages only symlinks to saved body/DMX/audio/color/loudness inputs
and delegates presentation to the existing native 3D renderer.  It never
regenerates a body, changes mapper/P2 settings, or applies D4/D6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.rendering import render_native_bf16_v7b_3d as renderer


REPO_ROOT = Path("/root/lightsem")
BASELINE_ROOT = REPO_ROOT / "experiments/native_bf16_v7b"
STEREO_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b_stereo_v3")
SONGS = ("shpongle", "cure_for_me")
CONDITIONS = ("baseline", "R1", "R2", "R2_zero", "R2_reverse")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"refusing to overwrite staged input: {dst}")
    dst.symlink_to(src)


def stage_inputs(stage: Path, condition: str) -> None:
    """Create a stable, tiny input root for the legacy presentation adapter."""
    if stage.exists():
        raise FileExistsError(f"staging root already exists: {stage}")
    for relative in ("demo_audio", "demo_colors_v2", "demo_loudness", "demo_features/features"):
        (stage / relative).mkdir(parents=True, exist_ok=True)
    # The renderer's source_data() expects these two fixed subtrees.
    (stage / "demo_bodies/joint_seed0").mkdir(parents=True, exist_ok=True)
    (stage / "demo_emission/joint_seed0").mkdir(parents=True, exist_ok=True)
    (stage / "learning/runs/joint/seed_0").mkdir(parents=True, exist_ok=True)
    link(stage / "learning/runs/joint/seed_0/final.pt",
         Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/learning/runs/joint/seed_0/final.pt"))
    for song in SONGS:
        if condition == "baseline":
            body = BASELINE_ROOT / "demo_bodies/joint_seed0" / f"{song}.npz"
            emission = BASELINE_ROOT / "demo_emission/joint_seed0" / f"{song}.npz"
        else:
            body = STEREO_ROOT / "showcase_bodies_v2" / condition / f"{song}.npz"
            emission = STEREO_ROOT / "showcase_emission_v2" / f"{condition}_{song}.npz"
        link(stage / "demo_bodies/joint_seed0" / f"{song}.npz", body)
        link(stage / "demo_emission/joint_seed0" / f"{song}.npz", emission)
        link(stage / "demo_colors_v2" / song / "P2.npz", BASELINE_ROOT / "demo_colors_v2" / song / "P2.npz")
        link(stage / "demo_colors_v2" / song / "REPORT.json", BASELINE_ROOT / "demo_colors_v2" / song / "REPORT.json")
        link(stage / "demo_loudness" / song / "loudness.json", BASELINE_ROOT / "demo_loudness" / song / "loudness.json")
        link(stage / "demo_loudness" / song / "loudness.json.sha256.json", BASELINE_ROOT / "demo_loudness" / song / "loudness.json.sha256.json")
        link(stage / "demo_features/features" / "timelines" / f"{song}.npz", BASELINE_ROOT / "demo_features/features/timelines" / f"{song}.npz")
        link(stage / "demo_features/features" / "base92" / f"{song}.npz", BASELINE_ROOT / "demo_features/features/base92" / f"{song}.npz")
    link(stage / "demo_audio/Cure_For_Me.wav", BASELINE_ROOT / "demo_audio/Cure_For_Me.wav")


def run_condition(condition: str, out_root: Path, duration: float, encoder: str, preset: str) -> dict:
    # Bind staging to the output artifact root so an interrupted render can be
    # preserved and a diagnosed rerun cannot accidentally reuse its links.
    stage = out_root.parent / f"{out_root.name}_inputs_{condition}"
    stage_inputs(stage, condition)
    # The presentation module has a fixed historical ROOT; replace it only in
    # this isolated process so every input remains hash-linked and read-only.
    renderer.ROOT = stage
    renderer.DURATION = float(duration)
    renderer.SONGS = {
        "shpongle": Path("/root/shpongle.flac"),
        "cure_for_me": stage / "demo_audio/Cure_For_Me.wav",
    }
    renderer.prepare(out_root)
    rows = []
    for song in SONGS:
        renderer.smoke(out_root, song)
        renderer.full_render(out_root, song, video_encoder=encoder, video_preset=preset, full=False)
        renderer.audit(out_root, song)
        render_record = json.loads((out_root / song / "RENDER.json").read_text())
        audit_record = json.loads((out_root / song / "AUDIT.json").read_text())
        rows.append({
            "song": song,
            "video": render_record["path"],
            "video_sha256": render_record["sha256"],
            "audit": str(out_root / song / "AUDIT.json"),
            "audit_sha256": sha256(out_root / song / "AUDIT.json"),
            "software_decode_valid": audit_record["software_decode"],
        })
    return {"condition": condition, "status": "complete", "duration_s": duration, "rows": rows,
            "input_stage": str(stage), "diagnostic_only": True,
            "scope": "native 3D review; saved body and decoded mapper-01 DMX; no promotion"}


def main(argv: list[str] | None = None) -> int:
    global STEREO_ROOT, BASELINE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    parser.add_argument("--stereo-root", type=Path, default=STEREO_ROOT,
                        help="artifact root containing showcase_bodies_v2 and showcase_emission_v2")
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    # New review generations use 120 seconds; historical roots are immutable.
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--encoder", choices=("h264_nvenc", "av1_nvenc", "libx264"), default="h264_nvenc")
    parser.add_argument("--preset", default="p4")
    args = parser.parse_args(argv)
    STEREO_ROOT = args.stereo_root
    BASELINE_ROOT = args.baseline_root
    conditions = tuple(args.condition or CONDITIONS)
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite showcase video root: {args.out_root}")
    args.out_root.mkdir(parents=True)
    rows = [run_condition(condition, args.out_root / condition, args.duration, args.encoder, args.preset)
            for condition in conditions]
    result = {"schema_version": 1, "status": "complete", "diagnostic_only": True,
              "conditions": list(conditions), "rows": rows,
              "source_index": str(STEREO_ROOT / "DEMO_INDEX_V2.json"),
              "scope": "short review clips only; raw body/DMX audits remain authoritative"}
    manifest = args.out_root / "SHOWCASE_VIDEO_MANIFEST.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest.with_name(manifest.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(manifest)}, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "conditions": list(conditions), "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
