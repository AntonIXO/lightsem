#!/usr/bin/env python3
"""Render full-song before/after comparisons for the SSL residual overlay."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from lightsem.experiments.stage11.stage11_common import gpu_lock, require_space
from lightsem.experiments.audio_light_foundation.data import sha


OLD_ROOT = Path("/root/lightsem/experiments/native_bf16_v7b/demo_videos_v6")
AFTER_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/body_light_ssl_native_overlay_20261015_v5")
OUT_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/body_light_ssl_native_overlay_20261015_full_v1")


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(path)}) + "\n")


def render_full(song: str, source_root: Path, destination: Path, audio: Path, label: str) -> dict:
    from scripts.rendering.render_body_transfer import render_one
    destination.mkdir(parents=True, exist_ok=False)
    source = source_root / song
    os.symlink(source / "timeline.npz", destination / "timeline.npz")
    os.symlink(source / "patch.json", destination / "patch.json")
    with np.load(source / "timeline.npz", allow_pickle=False) as z:
        duration = float(z["time"][-1] + 1.0 / 40.0)
    track = {"id": song, "title": f"{song} — {label}", "audio": str(audio)}
    plan = {"review_start_s": 0.0, "review_duration_s": duration}
    movie = render_one(destination, track, plan)
    from scripts.motion.run_rhythm_companion import audit_movie
    audit = audit_movie(destination, movie)
    return {"label": label, "movie": str(movie), "movie_sha256": sha(movie),
            "duration_s": duration, "audit": audit, "source_timeline": str(source / "timeline.npz"),
            "source_timeline_sha256": sha(source / "timeline.npz")}


def comparison(song: str, before: Path, after: Path, destination: Path, audio: Path) -> dict:
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(after)
    ], text=True).strip())
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(before), "-i", str(after),
           "-filter_complex",
           "[0:v]scale=960:540,drawtext=text='BEFORE - saved mapper 01':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[a];"
           "[1:v]scale=960:540,drawtext=text='AFTER - per-fixture SSL residual':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[b];"
           "[a][b]hstack=inputs=2[v]",
           "-map", "[v]", "-map", "1:a:0", "-t", f"{duration:.6f}",
           "-c:v", "av1_nvenc", "-preset", "p7", "-b:v", "2200k", "-maxrate", "2400k",
           "-bufsize", "4000k", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(destination)]
    subprocess.run(cmd, check=True)
    return {"song": song, "path": str(destination), "sha256": sha(destination),
            "duration_s": duration, "audio": str(audio), "audio_sha256": sha(audio), "command": cmd,
            "scope": "full-song side-by-side; same saved body, P2 RGB and source audio"}


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    require_space(Path("/mnt/lightsem-data"), 20)
    out.mkdir(parents=True)
    write_json(out / "PROTOCOL.json", {"schema_version": 1, "status": "diagnostic_only",
        "songs": ["shpongle", "cure_for_me"], "before_root": str(OLD_ROOT),
        "after_root": str(AFTER_ROOT), "render_scope": "full source duration",
        "render_fps": 30, "lighting_fps": 40, "P2_fixed": True,
        "body_generation_changed": False, "promotion": "forbidden"})
    rows = []; movies = []
    audio = {"shpongle": Path("/root/shpongle.flac"),
             "cure_for_me": Path("/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav")}
    with gpu_lock():
        for song in ("shpongle", "cure_for_me"):
            folder = out / song; folder.mkdir()
            before = render_full(song, OLD_ROOT, folder / "before", audio[song], "saved mapper 01")
            after = render_full(song, AFTER_ROOT, folder / "after", audio[song], "per-fixture SSL residual")
            pair = comparison(song, Path(before["movie"]), Path(after["movie"]), folder / f"{song}_full_before_after.mp4", audio[song])
            rows.append({"song": song, "before": before, "after": after, "comparison": pair})
            movies.append(pair)
    write_json(out / "DELIVERY.json", {"schema_version": 1, "status": "diagnostic_only", "rows": rows,
        "movies": movies, "body_generation_changed": False, "promotion": "forbidden"})
    print(json.dumps({"out": str(out), "status": "diagnostic_only", "movies": movies}, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--out", default=str(OUT_ROOT)); args = parser.parse_args(argv)
    run(Path(args.out))


if __name__ == "__main__":
    main()
