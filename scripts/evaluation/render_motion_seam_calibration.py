#!/usr/bin/env python3
"""Calibrate rendered beam-motion retention with static and known-motion controls.

The known-motion path is a renderer calibration control only; it is never used
as candidate model output or as a fallback choreography source.
"""
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


def calibration_directions(mode: str, frames: int, outputs: int) -> np.ndarray:
    if mode == "static":
        value = np.array([0.0, -1.0, 0.08], dtype=np.float32)
        value /= np.linalg.norm(value)
        return np.repeat(value[None, None, :], frames * outputs, axis=0).reshape(frames, outputs, 3)
    if mode != "known_linear_sweep":
        raise ValueError(mode)
    half = max(1, frames // 2)
    path = np.concatenate((np.linspace(-0.7, 0.7, half), np.linspace(0.7, -0.7, frames - half)))
    result = np.zeros((frames, outputs, 3), dtype=np.float32)
    for t, x in enumerate(path):
        values = np.stack((np.full(outputs, x), np.full(outputs, -1.0), np.full(outputs, 0.08)), axis=1)
        result[t] = values / np.linalg.norm(values, axis=1, keepdims=True)
    return result


def render(mode: str, out: Path, patch: list[dict], audio: Path, seconds: float) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    fps = 30
    frames = int(seconds * fps)
    n = len(patch)
    directions = calibration_directions(mode, frames, n)
    rgb = np.repeat(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), n, axis=0)
    dimmer = np.full(n, 0.6, dtype=np.float32)
    body = np.full((22, 3), 1000.0, np.float32)
    cfg = RenderConfig(width=640, height=600, fps=fps, cam_orbit=0.0, haze=0.25,
                       intensity_gain=2.4, exposure_profile="fixed", fixed_exposure_scale=.4,
                       fixed_role_gains=(1., .3, .03, .2), beam_edge_profile="angular",
                       beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                       require_gpu=True, cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)
    video = out / "calibration.mp4"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "640x600", "-r", str(fps), "-i", "pipe:0", "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frames), "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-t", str(seconds), "-movflags", "+faststart", str(video)]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    with accelerator_lock():
        renderer = LitBodyStageRenderer(patch, cfg, body_offset=(0., 0., 0.), body_scale=1., neutral_fill=0.)
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin was not created")
            for i in range(frames):
                image = renderer.render_controls(directions[i], rgb, dimmer, body)
                canvas = Image.fromarray(image).convert("RGB")
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), f"Renderer calibration | {mode}", font=font, fill="white")
                draw.text((10, 31), "calibration control only | not candidate motion | no fallback", font=small, fill="#c1d2e2")
                encoder.stdin.write(canvas.tobytes())
                if i in (0, frames // 2, frames - 1):
                    canvas.save(out / "frames" / f"frame_{i:06d}.png")
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        finally:
            renderer.close()
    montage = Image.new("RGB", (960, 300))
    for j, i in enumerate((0, frames // 2, frames - 1)):
        montage.paste(Image.open(out / "frames" / f"frame_{i:06d}.png").resize((320, 300)), (j * 320, 0))
    montage.save(out / "frames" / "montage.png")
    return {"mode": mode, "video": str(video.resolve()), "video_sha256": sha(video), "frames": frames,
            "montage": str((out / "frames" / "montage.png").resolve()), "native_dmx": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite calibration root: {args.out}")
    patch = [row for row in json.loads(args.patch.read_text()) if row["kind"] == "beam"]
    if len(patch) != 8:
        raise ValueError(f"expected 8 beam patch rows, got {len(patch)}")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [render(mode, args.out / mode, patch, args.audio, args.seconds)
            for mode in ("static", "known_linear_sweep")]
    receipt = {
        "schema": "lightsem.rendered_motion_calibration.v1",
        "protocol_id": "foundation-rendered-motion-calibration-20261020-v1",
        "source_patch": str(args.patch.resolve()),
        "source_patch_sha256": sha(args.patch),
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha(args.audio),
        "rows": rows,
        "interpretation": "Static and known-linear-sweep renderer controls calibrate the seam only; the known sweep is not candidate choreography or a rescue signal.",
    }
    (args.out / "CALIBRATION.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
