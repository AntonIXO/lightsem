#!/usr/bin/env python3
"""Render a gated audio-only v3 direct-light diagnostic demo."""

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


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directions(angle: np.ndarray) -> np.ndarray:
    pan = angle[..., 0] * np.pi
    tilt = angle[..., 1] * np.pi / 2.0
    xyz = np.stack((np.sin(pan) * np.cos(tilt), -np.cos(pan) * np.cos(tilt), np.sin(tilt)), -1)
    return xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="Foundation-v3 audio-only direct light")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing demo root: {args.out}")
    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "diagnostic_candidate":
        raise SystemExit(f"render blocked by own-song audit status: {audit.get('status')}")
    with np.load(args.prediction, allow_pickle=False) as z:
        time = np.asarray(z["time"], np.float64)
        angle = np.asarray(z["aim"], np.float32)
        rgb = np.asarray(z["rgb"], np.float32)
        brightness = np.asarray(z["brightness"], np.float32)
        event = np.asarray(z["event"], np.float32)
    n = min(len(time), int(args.seconds * 10))
    time, angle, rgb, brightness, event = time[:n], angle[:n], rgb[:n], brightness[:n], event[:n]
    fps = 30
    frames = int((n - 1) * fps / 10) + 1
    width, height = 640, 600
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frames").mkdir()
    directions_10 = directions(angle)
    learned_groups = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 5, 7])
    directions_10 = directions_10[:, learned_groups]
    rgb_10 = rgb[:, learned_groups]
    bright_10 = brightness[:, learned_groups, 0]
    dir_50 = np.repeat(directions_10[:, :1], 50, axis=1)
    rgb_50 = np.repeat(rgb_10[:, :1], 50, axis=1)
    bright_50 = np.repeat(bright_10[:, :1], 50, axis=1)
    # The first 13 patch entries are 8 learned beams plus 5 presentation
    # laser entries; the final 50 patch entries are explicit bulb copies.
    all_dirs = np.concatenate([directions_10, dir_50], 1)
    all_rgb = np.concatenate([rgb_10, rgb_50], 1)
    all_bright = np.concatenate([bright_10, bright_50], 1)
    cfg = RenderConfig(width=width, height=height, fps=fps, cam_orbit=0.0, haze=0.25,
                       intensity_gain=2.4, exposure_profile="fixed", fixed_exposure_scale=.4,
                       fixed_role_gains=(1., .3, .03, .2), beam_edge_profile="angular",
                       beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                       require_gpu=True, cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-i", str(args.audio),
               "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frames), "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-t", str(frames / fps), "-movflags", "+faststart", str(args.out / "demo.mp4")]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    patch_data = json.loads(args.patch.read_text())
    with accelerator_lock():
        renderer = LitBodyStageRenderer(patch_data, cfg, body_offset=(0., 0., 0.), body_scale=1., neutral_fill=0.)
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(frames):
                x = frame * 10.0 / fps
                left = min(n - 1, int(x)); right = min(n - 1, left + 1); alpha = x - left
                dirs = all_dirs[left] * (1 - alpha) + all_dirs[right] * alpha
                colors = all_rgb[left] * (1 - alpha) + all_rgb[right] * alpha
                dimmer = all_bright[left] * (1 - alpha) + all_bright[right] * alpha
                view = renderer.render_controls(dirs, colors, dimmer, np.full((22, 3), 1000., np.float32))
                canvas = Image.new("RGB", (width, height), (7, 10, 16))
                canvas.paste(Image.fromarray(view), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), args.label, font=font, fill="white")
                draw.text((10, 31), "audio-only inference | fixture metadata prior | bounded raw controls | diagnostic", font=small, fill="#c1d2e2")
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
        "schema": "lightsem.foundation_v3_demo.v1",
        "protocol_id": "foundation-v3-demo-20261020-v1",
        "status": "diagnostic_only",
        "audit": str(args.audit.resolve()), "audit_sha256": sha(args.audit),
        "prediction": str(args.prediction.resolve()), "prediction_sha256": sha(args.prediction),
        "audio": str(args.audio.resolve()), "audio_sha256": sha(args.audio),
        "patch": str(args.patch.resolve()), "patch_sha256": sha(args.patch),
        "duration_s": frames / fps, "fps": fps, "frames": frames,
        "learned_beams": 8, "presentation_lasers": 5, "presentation_bulbs": 50,
        "presentation_duplicates": True,
        "native_dmx": False,
        "body_inference_input": False,
        "raw_safety_gate": "passed by referenced own-song audit",
        "promotion": "forbidden_until_decoded_dmx_and_human_gates",
        "video": str((args.out / "demo.mp4").resolve()),
        "video_sha256": sha(args.out / "demo.mp4"),
        "frames_inspected": [str((args.out / "frames" / f"frame_{frame:06d}.png").resolve()) for frame in (0, frames // 2, frames - 1)],
        "montage": str((args.out / "frames" / "montage.png").resolve()),
    }
    (args.out / "DEMO.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

