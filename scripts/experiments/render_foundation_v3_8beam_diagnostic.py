#!/usr/bin/env python3
"""Render an 8-learned-beam v3 diagnostic only; never expands to arbitrary N."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.dance.pipeline import accelerator_lock
from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
from lightsem.sim.renderer import RenderConfig


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def directions(angle: np.ndarray) -> np.ndarray:
    pan = angle[..., 0] * np.pi
    tilt = angle[..., 1] * np.pi / 2.0
    xyz = np.stack((np.sin(pan) * np.cos(tilt), -np.cos(pan) * np.cos(tilt), np.sin(tilt)), -1)
    return xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8)


def stage_aware_directions(angle: np.ndarray, model_anchor: np.ndarray, patch: list[dict]) -> np.ndarray:
    """Map learned residual angles around each patch's physical target anchor."""
    positions = np.asarray([row["position"] for row in patch], dtype=np.float32)
    targets = np.asarray([row["anchor"] for row in patch], dtype=np.float32)
    base = targets - positions
    base_pan = np.arctan2(base[:, 0], -base[:, 2])
    base_tilt = np.arctan2(base[:, 1], np.linalg.norm(base[:, (0, 2)], axis=1))
    residual = angle - model_anchor[None, ...]
    pan = base_pan[None, :] + residual[..., 0] * np.pi
    tilt = base_tilt[None, :] + residual[..., 1] * (np.pi / 2.0)
    xyz = np.stack((np.sin(pan) * np.cos(tilt), -np.cos(pan) * np.cos(tilt), np.sin(tilt)), -1)
    return xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--mmca", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--label", default="Foundation-v3 jerk-safe")
    parser.add_argument("--protocol-id", default="foundation-v3-jerk-safe-8beam-diagnostic-20261020-v1")
    parser.add_argument("--stage-aware", action="store_true")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing demo root: {args.out}")
    with np.load(args.prediction, allow_pickle=False) as z:
        time = np.asarray(z["time"], np.float64)
        angle = np.asarray(z["aim"], np.float32)
        rgb = np.asarray(z["rgb"], np.float32)
        brightness = np.asarray(z["brightness"], np.float32)
        model_anchor = np.asarray(z["anchor"], np.float32)[0] if "anchor" in z.files else None
    with args.patch.open() as f:
        source_patch = json.load(f)
    learned_patch = [row for row in source_patch if row["kind"] == "beam"]
    if len(learned_patch) != 8:
        raise ValueError(f"expected exactly 8 beam fixtures, found {len(learned_patch)}")
    n = min(len(time), int(args.seconds * 10))
    time, angle, rgb, brightness = time[:n], angle[:n, :8], rgb[:n, :8], brightness[:n, :8]
    frames = int((n - 1) * 30 / 10) + 1
    width, height = 640, 600
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frames").mkdir()
    patch_path = args.out / "patch_8_beams.json"
    patch_path.write_text(json.dumps(learned_patch, indent=2, sort_keys=True) + "\n")
    if args.stage_aware:
        if model_anchor is None:
            raise ValueError("--stage-aware requires saved anchor array in prediction")
        dirs = stage_aware_directions(angle, model_anchor[:8], learned_patch)
    else:
        dirs = directions(angle)
    bright = brightness[..., 0]
    cfg = RenderConfig(width=width, height=height, fps=30, cam_orbit=0.0, haze=0.25,
                       intensity_gain=2.4, exposure_profile="fixed", fixed_exposure_scale=.4,
                       fixed_role_gains=(1., .3, .03, .2), beam_edge_profile="angular",
                       beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                       require_gpu=True, cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", "30", "-i", "pipe:0", "-i", str(args.audio),
               "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frames), "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-t", str(frames / 30), "-movflags", "+faststart", str(args.out / "demo.mp4")]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    neutral = np.full((22, 3), 1000.0, np.float32)
    with accelerator_lock():
        renderer = LitBodyStageRenderer(learned_patch, cfg, body_offset=(0., 0., 0.), body_scale=1., neutral_fill=0.)
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin was not created")
            for frame in range(frames):
                x = frame * 10.0 / 30.0
                left = min(n - 1, int(x)); right = min(n - 1, left + 1); alpha = x - left
                d = dirs[left] * (1 - alpha) + dirs[right] * alpha
                c = rgb[left] * (1 - alpha) + rgb[right] * alpha
                b = bright[left] * (1 - alpha) + bright[right] * alpha
                view = renderer.render_controls(d, c, b, neutral)
                canvas = Image.new("RGB", (width, height), (7, 10, 16))
                canvas.paste(Image.fromarray(view), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), f"{args.label} | {args.song}", font=font, fill="white")
                draw.text((10, 31), "8 learned beams only | raw-safe diagnostic | MMCA unqualified | no native DMX", font=small, fill="#c1d2e2")
                encoder.stdin.write(canvas.tobytes())
                if frame in (0, frames // 2, frames - 1):
                    canvas.save(args.out / "frames" / f"frame_{frame:06d}.png")
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        finally:
            renderer.close()
    montage = Image.new("RGB", (960, 300))
    for index, frame in enumerate((0, frames // 2, frames - 1)):
        montage.paste(Image.open(args.out / "frames" / f"frame_{frame:06d}.png").resize((320, 300)), (index * 320, 0))
    montage.save(args.out / "frames" / "montage.png")
    receipt = {
        "schema": "lightsem.foundation_v3_8beam_demo.v1",
        "protocol_id": args.protocol_id,
        "status": "diagnostic_only",
        "song": args.song,
        "seconds": frames / 30,
        "fps": 30,
        "frames": frames,
        "prediction": str(args.prediction.resolve()),
        "prediction_sha256": sha(args.prediction),
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha(args.audio),
        "patch": str(patch_path.resolve()),
        "patch_sha256": sha(patch_path),
        "source_patch": str(args.patch.resolve()),
        "source_patch_sha256": sha(args.patch),
        "raw_audit": str(args.raw_audit.resolve()),
        "raw_audit_sha256": sha(args.raw_audit),
        "mmca": str(args.mmca.resolve()),
        "mmca_sha256": sha(args.mmca),
        "learned_beams": 8,
        "presentation_lasers": 0,
        "presentation_bulbs": 0,
        "arbitrary_output_test": False,
        "native_dmx": False,
        "body_inference_input": False,
        "stage_aware_mapper": args.stage_aware,
        "conditioning_status": "unconditioned_or_shared_prior",
        "promotion": "forbidden_until_correct_song_conditioning_decoded_dmx_and_human_gates",
        "video": str((args.out / "demo.mp4").resolve()),
        "video_sha256": sha(args.out / "demo.mp4"),
        "frames_inspected": [str((args.out / "frames" / f"frame_{frame:06d}.png").resolve()) for frame in (0, frames // 2, frames - 1)],
        "montage": str((args.out / "frames" / "montage.png").resolve()),
    }
    (args.out / "DEMO.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
