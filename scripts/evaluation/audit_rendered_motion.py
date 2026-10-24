#!/usr/bin/env python3
"""Measure perceptible spatial motion in an actual rendered video.

This is deliberately separate from native aim/latent/RGB audits. It samples
luminance frames, removes the text header, extracts the brightest sparse beam
regions, matches component centroids between adjacent samples, and reports
pixel-change controls. It is a diagnostic measurement; thresholds must be
frozen with static/known-motion controls before promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frames(path: Path, count: int, width: int, height: int) -> list[np.ndarray]:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='not(mod(n\\, {max(1, count)}))',setpts=N/FRAME_RATE/TB",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = width * height
    values = np.frombuffer(raw, dtype=np.uint8)
    n = len(values) // frame_bytes
    return [values[i * frame_bytes:(i + 1) * frame_bytes].reshape(height, width) for i in range(n)]


def bright_centroid(frame: np.ndarray, header_rows: int = 55) -> tuple[np.ndarray | None, int]:
    image = frame[header_rows:].astype(np.float32)
    threshold = float(np.quantile(image, 0.995))
    mask = image >= max(2.0, threshold)
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None, int(len(xs))
    return np.array([float(xs.mean()), float(ys.mean() + header_rows)]), int(len(xs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    frames = load_frames(args.video, args.samples, args.width, args.height)
    centroids, bright_counts = [], []
    for frame in frames:
        centroid, count = bright_centroid(frame)
        centroids.append(centroid)
        bright_counts.append(count)
    valid = [x for x in centroids if x is not None]
    centroid_steps = [float(np.linalg.norm(valid[i] - valid[i - 1])) for i in range(1, len(valid))]
    frame_diffs = [float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)))) for i in range(1, len(frames))]
    report = {
        "schema": "lightsem.rendered_motion_audit.v1",
        "protocol_id": "foundation-rendered-beam-spatial-motion-20261020-v1",
        "video": str(args.video.resolve()),
        "video_sha256": sha(args.video),
        "samples": len(frames),
        "frame_shape": [args.height, args.width],
        "header_rows_excluded": 55,
        "bright_pixel_count": bright_counts,
        "valid_bright_centroids": len(valid),
        "bright_centroid_step_px": {
            "mean": float(np.mean(centroid_steps)) if centroid_steps else None,
            "p50": float(np.quantile(centroid_steps, .5)) if centroid_steps else None,
            "p95": float(np.quantile(centroid_steps, .95)) if centroid_steps else None,
            "max": float(np.max(centroid_steps)) if centroid_steps else None,
        },
        "frame_mae": {
            "mean": float(np.mean(frame_diffs)) if frame_diffs else None,
            "p50": float(np.quantile(frame_diffs, .5)) if frame_diffs else None,
            "p95": float(np.quantile(frame_diffs, .95)) if frame_diffs else None,
        },
        "interpretation": "Measurement only. Bright-centroid travel and pixel MAE require frozen static and known-motion controls before a pass/fail threshold is assigned.",
        "promotion": "blocked until perceptible rendered beam travel is validated against controls and decoded-DMX provenance",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
