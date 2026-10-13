#!/usr/bin/env python
"""Render a quarantined body-to-DMX diagnostic demo with synced audio."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np

from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream, emitted_light
PARENTS = (-1, 0, 1, 2, 3, 4, 0, 6, 7, 0, 9, 10, 11, 0, 13, 14, 0, 16, 17, 0, 19, 20)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interp_array(t_src, values, t_dst):
    values = np.asarray(values)
    return np.stack([np.interp(t_dst, t_src, values[:, i]) for i in range(values.shape[1])], 1)


def hsv_rgb(h, s, v):
    h = (h % 1.0) * 6.0; i = int(h); f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i % 6]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", type=Path, required=True); ap.add_argument("--audio", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(); demo = args.demo.resolve(); audio = args.audio.resolve(); out = args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    meta = json.loads((demo / "DEMO.json").read_text())
    if meta.get("status") != "diagnostic_only" or not meta.get("finite_dmx") or not meta.get("raw_body", {}).get("finite"):
        raise RuntimeError("Refusing video: demo is not a finite diagnostic package")
    with np.load(demo / "demo.npz", allow_pickle=False) as z:
        t10, world10, dmx10 = z["time"].astype(float), z["world"].astype(np.float32), z["dmx"].astype(np.uint8)
    duration = min(float(t10[-1]), float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio)])))
    n = int(np.floor(duration * args.fps)); t = np.arange(n, dtype=float) / args.fps
    # Decode actual emitted bytes first, then interpolate decoded parameters.
    rig = club_rig(); decoded = decode_stream(rig, dmx10); emitted = emitted_light(decoded)
    moving = np.array([i for i, f in enumerate(rig.fixtures) if f.kind == "moving_head"])
    body = np.stack([np.stack([np.interp(t, t10, world10[:, j, k]) for k in range(3)], 1) for j in range(22)], 1)
    pan = interp_array(t10, decoded["pan"][:, moving], t); tilt = interp_array(t10, decoded["tilt"][:, moving], t)
    dim = interp_array(t10, decoded["dimmer"][:, moving], t); sat = interp_array(t10, decoded["sat"][:, moving], t); val = interp_array(t10, decoded["val"][:, moving], t)
    hue_s = interp_array(t10, np.sin(2 * np.pi * decoded["hue"][:, moving]), t); hue_c = interp_array(t10, np.cos(2 * np.pi * decoded["hue"][:, moving]), t); hue = np.arctan2(hue_s, hue_c) / (2 * np.pi) % 1.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lightsem_dmx_demo_") as td:
        silent = Path(td) / "silent.mp4"
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
        if not writer.isOpened(): raise RuntimeError("OpenCV video writer failed")
        for frame in range(n):
            canvas = np.zeros((args.height, args.width, 3), np.uint8); canvas[:] = (7, 9, 16)
            # Stage and truss.
            cv2.rectangle(canvas, (55, 100), (args.width - 55, args.height - 75), (13, 16, 28), -1)
            cv2.line(canvas, (70, 120), (args.width - 70, 120), (90, 90, 110), 4); cv2.line(canvas, (70, args.height - 75), (args.width - 70, args.height - 75), (50, 50, 65), 3)
            glow = np.zeros_like(canvas)
            for h, fixture_idx in enumerate(moving):
                sx = int(150 + h * (args.width - 300) / max(len(moving) - 1, 1)); sy = 125
                rgb = np.array(hsv_rgb(hue[frame, h], np.clip(sat[frame, h], 0, 1), np.clip(val[frame, h] * dim[frame, h], 0, 1))) * 255
                color = tuple(int(x) for x in rgb[::-1])
                tx = int(np.clip(sx + (pan[frame, h] - .5) * 760, 70, args.width - 70)); ty = int(np.clip(300 + (1 - tilt[frame, h]) * 280, 170, args.height - 80))
                cv2.line(glow, (sx, sy), (tx, ty), color, 24); cv2.line(canvas, (sx, sy), (tx, ty), color, 4); cv2.circle(canvas, (sx, sy), 10, color, -1)
            canvas = cv2.addWeighted(canvas, 1.0, cv2.GaussianBlur(glow, (0, 0), 18), .25, 0)
            # Body in stage coordinates: x horizontal, y vertical, z controls depth scale.
            pts = []
            for j in range(22):
                x, y, z = body[frame, j]; px = int(args.width / 2 + x * 180); py = int(args.height - 120 - y * 180); pts.append((px, py))
            for j, parent in enumerate(PARENTS):
                if parent >= 0: cv2.line(canvas, pts[j], pts[parent], (225, 230, 240), 3)
            for px, py in pts: cv2.circle(canvas, (px, py), 5, (240, 245, 255), -1)
            cv2.putText(canvas, "LIGHTSEM BODY-TO-DMX | DIAGNOSTIC ONLY", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, .8, (235, 235, 245), 2)
            cv2.putText(canvas, f"t={t[frame]:6.2f}s | emitted DMX | moving heads={len(moving)}", (70, 78), cv2.FONT_HERSHEY_SIMPLEX, .55, (170, 175, 195), 1)
            writer.write(canvas)
        writer.release()
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(audio), "-t", f"{duration:.6f}", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
        subprocess.run(cmd, check=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)]))
    v = next(x for x in probe["streams"] if x["codec_type"] == "video"); a = next(x for x in probe["streams"] if x["codec_type"] == "audio")
    report = {"schema_version": 1, "status": "diagnostic_only", "reason": meta["reason"], "video": str(out), "video_sha256": sha(out), "source_demo": str(demo / "demo.npz"), "source_dmx_sha256": meta["dmx_sha256"], "audio": str(audio), "audio_sha256": sha(audio), "duration_s": float(probe["format"]["duration"]), "video_frames": int(v.get("nb_frames", 0) or 0), "video_fps": v.get("r_frame_rate"), "audio_codec": a.get("codec_name"), "video_codec": v.get("codec_name"), "raw_body_finite": meta["raw_body"]["finite"], "raw_body_jerk_p95_m_s3": meta["raw_body"]["raw_relative_jerk_p95_m_s3"], "pcm_identity": "source WAV muxed once; ffprobe stream present"}
    out.with_suffix(".VIDEO_AUDIT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
