"""Prepare the October 12 native-baseline 15×2 body-to-light campaign.

This script performs the CPU-side freeze, raw-body admission, source-dependent
mapper compilation, and timeline construction. Native body generation itself is
done by ``lightsem.cli dance`` using the active fixed-C configuration; GPU
rendering is deliberately handled by the sequential curator after this script
has admitted the saved bodies and actual emitted controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import CATALOG, make_outputs
from lightsem.conductor.body_transfer_v2.diagnostics import measure
from lightsem.dance.baseline import load_config, sha
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.experiments.native_repair.assessment import raw_metrics

ROOT = Path("/root/lightsem")
OUT = Path(os.environ.get("NATIVE_PORTFOLIO_ROOT", "/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v2"))
SONGS = {
    "shpongle": Path("/root/shpongle.flac"),
    "cure_for_me": Path("/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav"),
}
REFERENCE_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/native_stereo_counterfactual_20261011_v1/production/active_baseline_showcase/bodies/dual_cross_attention")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha_array(value) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def hash_files(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(set(map(Path, paths)))}


def load_world_npz(path: Path):
    """Read sealed raw body arrays without creating sidecars beside them."""
    with np.load(path, allow_pickle=False) as values:
        key = "world" if "world" in values.files else "body"
        if key not in values.files:
            raise ValueError(f"body NPZ has no world/body array: {path}")
        world = values[key].astype(np.float64, copy=True)
        time = values["time"].astype(np.float64, copy=True) if "time" in values.files else np.arange(len(world), dtype=np.float64) / 30.
    if world.ndim != 3 or world.shape[1:] != (22, 3) or not np.isfinite(world).all():
        raise ValueError(f"invalid raw body shape/values: {path}")
    return world, time


def active_inputs(config):
    paths = [ROOT / "configs/dance/fixed_c.json", ROOT / "configs/light_layers/baseline.json",
             ROOT / "scripts/motion/rebuild_native_portfolio_20261012.py",
             ROOT / "scripts/rendering/render_body_portfolio.py",
             ROOT / "lightsem/dance/pipeline.py", ROOT / "lightsem/dance/lighting.py",
             ROOT / "lightsem/dance/baseline.py", ROOT / "lightsem/conductor/body_transfer.py",
             ROOT / "lightsem/conductor/body_transfer_v2/__init__.py"]
    paths += [Path(config["checkpoint"]["path"]), Path(config["muq"]["weights_path"]) if config["muq"].get("weights_path") else Path(config["checkpoint"]["path"])]
    paths += [Path(config["stereo_conditioning"]["context_manifest"]), Path(config["lighting"]["color_config"])]
    for row in config["cached_tracks"].values():
        paths += [Path(row["audio_sha256"] and row["timeline"]["path"]), Path(row["base92"]["path"]), Path(row["colors"]["path"]), Path(row["color_report"]["path"]), Path(row["loudness_receipt"]["path"])]
    return paths


def freeze():
    if (OUT / "PLAN.json").exists():
        raise FileExistsError("campaign already frozen; choose a new revision")
    config, config_path = load_config(ROOT / "configs/dance/fixed_c.json")
    OUT.mkdir(parents=True)
    tracks = []
    for song, audio in SONGS.items():
        row = config["cached_tracks"][song]
        reference_audio = OUT / "audio_reference" / f"{song}.m4a"
        reference_audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-ss", "60", "-i", str(audio), "-map", "0:a:0", "-vn", "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-af", "atrim=duration=120,asetpts=PTS-STARTPTS", "-t", "120", "-movflags", "+faststart", str(reference_audio)], check=True)
        tracks.append({"id": song, "title": "Shpongle" if song == "shpongle" else "AURORA — Cure For Me",
            "audio": str(audio), "audio_sha256": sha(audio), "audio_reference": str(reference_audio), "audio_reference_sha256": sha(reference_audio), "timeline": row["timeline"], "base92": row["base92"],
            "colors": row["colors"], "color_report": row["color_report"], "loudness": row["loudness_receipt"],
            "reference_body": str(REFERENCE_ROOT / f"{song}.npz"), "reference_body_sha256": sha(REFERENCE_ROOT / f"{song}.npz")})
    files = hash_files([config_path, ROOT / "configs/light_layers/baseline.json", ROOT / "scripts/motion/rebuild_native_portfolio_20261012.py", ROOT / "scripts/rendering/render_body_portfolio.py", ROOT / "lightsem/dance/pipeline.py", ROOT / "lightsem/dance/lighting.py", ROOT / "lightsem/dance/baseline.py", ROOT / "lightsem/conductor/body_transfer.py", ROOT / "lightsem/conductor/body_transfer_v2/__init__.py", Path(config["checkpoint"]["path"]), Path(config["stereo_conditioning"]["context_manifest"]), *[Path(t["audio"]) for t in tracks], *[Path(t["audio_reference"]) for t in tracks], *[Path(t["timeline"]["path"]) for t in tracks], *[Path(t["base92"]["path"]) for t in tracks], *[Path(t["colors"]["path"]) for t in tracks], *[Path(t["color_report"]["path"]) for t in tracks], *[Path(t["loudness"]["path"]) for t in tracks], *[Path(t["reference_body"]) for t in tracks]])
    plan = {"schema_version": 1, "id": OUT.name, "status": "frozen_latest_native_baseline", "baseline_config": str(config_path), "baseline_config_sha256": sha(config_path), "baseline_id": config["id"], "checkpoint": config["checkpoint"], "muq": config["muq"], "stereo_conditioning": config["stereo_conditioning"], "tracks": tracks, "algorithms": [{"id": k, "name": v[0], "papers": ["https://arxiv.org/abs/" + p for p in v[1]]} for k, v in sorted(CATALOG.items())], "counts": config["lighting"]["counts"], "prefix_s": 180, "review_start_s": 60, "review_duration_s": 120, "fps": 40., "body_fps": 30., "d4max": False, "d6": False, "motion_limits": False, "mapper_family": "body_transfer_v2 current workspace algorithms", "p2_config": str(ROOT / "configs/light_layers/baseline.json"), "files": files, "raw_safety": {"max_speed_ratio": 1.5, "max_acceleration_ratio": 2.0, "max_jerk_ratio": 2.0, "scope": "raw native 30Hz body before interpolation/mapper/DMX/video"}, "qualification": "user_selected_motion_baseline; health evidence diagnostic where conditioning coverage is incomplete"}
    save(OUT / "PLAN.json", plan)
    save(OUT / "source_manifest.json", {"files": files, "source_hashes_are_workspace_bindings": True})
    print(json.dumps({"stage": "freeze", "root": str(OUT), "file_count": len(files), "checkpoint_sha256": config["checkpoint"]["sha256"]}, indent=2))


def audit_bodies():
    plan = json.loads((OUT / "PLAN.json").read_text())
    rows = {}
    for track in plan["tracks"]:
        song = track["id"]; path = OUT / "bodies" / song / "body.npz"
        body, body_time = load_world_npz(path)
        reference_path = Path(track["reference_body"])
        reference, _ = load_world_npz(reference_path)
        candidate = raw_metrics(body, reference)
        ref = raw_metrics(reference, reference)
        ratios = {k: candidate[k] / max(ref[k], 1e-12) for k in ("raw_relative_jerk_p95_m_s3", "raw_relative_speed_mean_m_s", "high_frequency_residual_RMS_m", "root_vertical_fast_residual_RMS_m")}
        safety = {"finite": candidate["finite"], "constant_body": candidate["constant_body"], "candidate": candidate, "reference": ref, "ratios": ratios, "thresholds": plan["raw_safety"], "pass": bool(candidate["finite"] and not candidate["constant_body"] and ratios["raw_relative_speed_mean_m_s"] <= plan["raw_safety"]["max_speed_ratio"] and ratios["raw_relative_jerk_p95_m_s3"] <= plan["raw_safety"]["max_jerk_ratio"])}
        row = {"song": song, "body": str(path), "body_sha256": sha(path), "generation_sha256": sha(path.parent / "GENERATION.json"), "frames": len(body), "body_fps": 30, "checkpoint": json.loads((path.parent / "GENERATION.json").read_text()).get("checkpoint"), "raw_safety": safety, "audio_sha256": track["audio_sha256"], "source_body_used": False}
        rows[song] = row
        if not safety["pass"]:
            raise RuntimeError(f"rejected_raw_safety: {song}; no mapper or render permitted")
    save(OUT / "RAW_SAFETY.json", {"status": "complete", "rows": rows, "scope": "Raw native 30Hz body before interpolation, mapper, DMX, D4, D6 or video"})
    print(json.dumps({"stage": "audit_bodies", "status": "complete", "songs": list(rows), "passes": 2}, indent=2))


def map_all():
    plan = json.loads((OUT / "PLAN.json").read_text())
    safety = json.loads((OUT / "RAW_SAFETY.json").read_text())
    config, _ = load_config(ROOT / "configs/dance/fixed_c.json")
    outputs = make_outputs(**plan["counts"])
    for track in plan["tracks"]:
        song = track["id"]; body_path = OUT / "bodies" / song / "body.npz"
        native, body_time = load_world_npz(body_path); native = native.astype(np.float32)
        frames = int(plan["prefix_s"] * 40); t40 = np.arange(frames, dtype=np.float64) / 40.
        if body_time[-1] < plan["prefix_s"] - 1/30: raise RuntimeError(f"body shorter than prefix: {song}")
        world40 = _resample_body_full_interval(native, body_time, t40)
        # P2 and LUFS receipts cover the complete song. Authenticate the full
        # streams first, then crop exactly the declared native 0–180 s prefix;
        # passing ``frames`` to the loader would incorrectly reinterpret a
        # complete-song receipt as a prefix artifact.
        left, right, palette, color_active, identity = _load_p2_colors(Path(track["colors"]["path"]), Path(track["audio"]), None)
        loudness, loud_identity = load_verified_loudness(Path(track["loudness"]["path"]), Path(track["audio"]))
        left, right, palette, color_active = left[:frames], right[:frames], palette[:frames], color_active[:frames]
        # Keep the authenticated full-track LUFS dictionary. The historical
        # brightness law interpolates the complete curve onto the cropped
        # lighting clock and uses the full-track anchor quantile.
        for algorithm in plan["algorithms"]:
            aid = algorithm["id"]; folder = OUT / "timelines" / aid / song; folder.mkdir(parents=True, exist_ok=True)
            dmx, patch, intent, brightness, hard = compile_world(world40, t40, outputs, left, right, loudness, config["lighting"]["transfer"], mapper_id=aid)
            controls = decode(dmx, patch)
            if not all(np.isfinite(np.asarray(v)).all() for v in controls.values()): raise RuntimeError(f"nonfinite emitted controls {aid}/{song}")
            np.savez_compressed(folder / "timeline.npz", time=t40, body=world40, universes=dmx, targets=intent["targets"], dimmer=intent["dimmer"], source_points=intent["source_points"], field_points=intent["field_points"], attribution_mask=np.zeros((frames, len(outputs), 22), dtype=np.uint8))
            (folder / "patch.json").write_text(json.dumps(patch, indent=2) + "\n")
            diagnostics = measure(world40, intent, dmx, patch, fps=40., start_s=0., duration_s=plan["prefix_s"])
            report = {"status": "diagnostic", "algorithm": aid, "song": song, "baseline_id": plan["baseline_id"], "checkpoint_sha256": plan["checkpoint"]["sha256"], "stereo_context_manifest_sha256": plan["stereo_conditioning"]["context_manifest_sha256"], "body_sha256": sha(body_path), "timeline_sha256": sha(folder / "timeline.npz"), "patch_sha256": sha(folder / "patch.json"), "audio_sha256": track["audio_sha256"], "p2_sha256": sha(track["colors"]["path"]), "loudness_receipt_sha256": sha(track["loudness"]["path"]), "frames": frames, "fixtures": len(outputs), "hard_checks": hard, "diagnostics": diagnostics, "raw_safety": safety["rows"][song]["raw_safety"], "d4max": False, "d6": False, "motion_limits": False, "source_body_used": False, "own_song_body": True, "scope": "Latest native baseline body mapped through current deterministic research-inspired mapper; health remains diagnostic"}
            save(folder / "report.json", report)
            print(f"MAPPED {aid} {song}", flush=True)
    print(json.dumps({"stage": "map", "pairs": 30}, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("freeze", "audit_bodies", "map")); a = p.parse_args()
    {"freeze": freeze, "audit_bodies": audit_bodies, "map": map_all}[a.stage]()


if __name__ == "__main__": main()
