#!/usr/bin/env python3
"""Apply the trained SSL representation as a bounded light residual.

The native body, audio, P2 RGB, and saved mapper-01 DMX remain fixed.  The SSL
checkpoint predicts a semantic-light residual from the body/audio history; only
small pan/tilt and dimmer deltas are applied to a copy of the saved emitted DMX.
This is a diagnostic overlay, not a new native body or a promoted baseline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from lightsem.conductor.body_transfer import decode
from lightsem.experiments.audio_light_foundation.data import _motion_proxy, load_rows, sha
from lightsem.experiments.body_light_ssl import ControlledJepaConfig, ControlledJepaModel
from lightsem.experiments.stage11.stage11_common import gpu_lock, require_space
from scripts.experiments.train_body_light_ssl import collect_probe_data, deterministic_windows, normalize


OLD_ROOT = Path("/root/lightsem/experiments/native_bf16_v7b")
SSL_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/body_light_ssl_20261015_v1")
OUT_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/body_light_ssl_native_overlay_20261015_v1")
FRAME_RATE = 40.0
SSL_RATE = 10.0
WINDOW = 192
STRIDE = 96
MAX_RESIDUAL_DEG = 0.05
MAX_DIMMER_RESIDUAL = 0.025
MAX_RATIO = 1.25


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"preserving existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(path)}) + "\n")


def nearest(values: np.ndarray, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    return values[choose]


def load_ssl(device: torch.device):
    saved = torch.load(SSL_ROOT / "checkpoint.pt", map_location=device, weights_only=False)
    cfg_keys = ("audio_dim", "body_dim", "light_dim", "width", "embedding_dim", "heads", "layers",
                "feedforward_mult", "dropout", "ema_decay", "mask_ratio", "action_semantics")
    config = ControlledJepaConfig(**{key: saved["config"][key] for key in cfg_keys})
    model = ControlledJepaModel(config).to(device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in saved["stats"].items()}
    return model, stats, config


def fit_probe(model, stats, device: torch.device):
    rows = prepare_rows_robust(load_rows())
    train_rows = [row for row in rows if row["id"] in ("1023", "10dcc", "1181f", "11b01", "1672d", "210e3")]
    windows = deterministic_windows(train_rows, stats, frames=WINDOW, device=device)
    x, y_light, _ = collect_probe_data(model, windows)
    design = torch.cat((x, torch.ones((len(x), 1), device=device)), -1)
    regularizer = torch.eye(design.shape[-1], device=device) * 1e-2
    regularizer[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ y_light)
    return weights


def prepare_rows_robust(rows: list[dict]) -> list[dict]:
    """Align cache streams before packing; current caches may differ by one frame."""
    prepared = []
    for row in rows:
        audio = np.asarray(row["audio"], dtype=np.float32)
        body = np.asarray(row["body"], dtype=np.float32)
        def pooled(name):
            value = np.asarray(row[name], dtype=np.float32)
            return value.mean(axis=1) if value.ndim == 3 else value
        parts = (pooled("motion"), pooled("speed"), pooled("acceleration"),
                 pooled("rgb"), pooled("brightness"), np.asarray(row["event"], np.float32))
        n = min(len(audio), len(body), *(len(part) for part in parts))
        light = np.concatenate(tuple(part[:n] for part in parts), axis=-1).astype(np.float32)
        prepared.append({"id": row["id"], "audio": audio[:n], "body": body[:n], "light": light,
                         "raw_world": row["raw_world"], "paths": row["paths"]})
    return prepared


def song_inputs(song: str):
    body_path = OLD_ROOT / "demo_bodies" / "joint_seed0" / f"{song}.npz"
    base_path = OLD_ROOT / "demo_features" / "features" / "base92" / f"{song}.npz"
    timeline_path = OLD_ROOT / "demo_features" / "features" / "timelines" / f"{song}.npz"
    old_timeline_path = OLD_ROOT / "demo_videos_v6" / song / "timeline.npz"
    patch_path = OLD_ROOT / "demo_videos_v6" / "patch.json"
    p2_path = OLD_ROOT / "demo_colors_v2" / song / "P2.npz"
    audio_path = Path("/root/shpongle.flac") if song == "shpongle" else OLD_ROOT / "demo_audio" / "Cure_For_Me.wav"
    with np.load(body_path, allow_pickle=False) as z:
        body30 = np.asarray(z["body"], np.float32)
    with np.load(base_path, allow_pickle=False) as z:
        base30 = np.asarray(z["x30"], np.float32)
        base_time = np.asarray(z["time30"], np.float64)
    with np.load(timeline_path, allow_pickle=False) as z:
        muq = np.asarray(z["muq"], np.float32)
        muq_time = np.asarray(z["muq_time"], np.float64)
    with np.load(old_timeline_path, allow_pickle=False) as z:
        old_dmx = np.asarray(z["universes"], np.uint8)
        world40 = np.asarray(z["body"], np.float32)
        time40 = np.asarray(z["time"], np.float64)
    patch = json.loads(patch_path.read_text())
    p2 = np.load(p2_path, allow_pickle=False)
    audio30 = np.concatenate((nearest(muq, muq_time, base_time), base30), axis=-1).astype(np.float32)
    n = min(len(body30) // 3, len(audio30) // 3, len(old_dmx) // 4)
    body10 = body30[::3][:n]
    audio10 = audio30[::3][:n]
    decoded = decode(old_dmx, patch)
    beam_indices = [i for i, item in enumerate(patch) if item["kind"] == "beam"][:8]
    sample40 = np.arange(n, dtype=int) * 4
    rgb = decoded["rgb"][sample40][:, beam_indices]
    brightness = decoded["dimmer"][sample40][:, beam_indices, None]
    event = np.asarray(p2["color_active40"][sample40], np.float32)[:, None]
    rel, aim, speed, acceleration = _motion_proxy(body10, n)
    semantic = np.concatenate((aim.mean(1), speed.mean(1), acceleration.mean(1),
                               rgb.mean(1), brightness.mean(1), event), axis=-1).astype(np.float32)
    return {"song": song, "body_path": body_path, "base_path": base_path, "timeline_path": timeline_path,
            "old_timeline_path": old_timeline_path, "patch_path": patch_path, "p2_path": p2_path,
            "audio_path": audio_path, "body10": body10, "audio10": audio10, "semantic": semantic,
            "world40": world40, "time40": time40, "old_dmx": old_dmx, "patch": patch,
            "decoded": decoded, "n10": n, "p2": p2}


@torch.no_grad()
def predict_semantic(model, stats, weights, item, device: torch.device) -> np.ndarray:
    n = item["n10"]
    audio = (item["audio10"] - stats["audio_mean"]) / stats["audio_std"]
    rel = item["body10"].reshape(n, -1).astype(np.float32)
    body = (rel - stats["body_mean"]) / stats["body_std"]
    light = (item["semantic"] - stats["light_mean"]) / stats["light_std"]
    accum = np.zeros((n, 9), np.float32); counts = np.zeros(n, np.float32)
    starts = list(range(0, max(1, n - WINDOW + 1), STRIDE))
    last = max(0, n - WINDOW)
    if not starts or starts[-1] != last:
        starts.append(last)
    for start in sorted(set(starts)):
        stop = start + WINDOW
        if stop > n:
            continue
        a = torch.from_numpy(audio[start:stop][None]).float().to(device)
        b = torch.from_numpy(body[start:stop][None]).float().to(device)
        l = torch.from_numpy(light[start:stop][None]).float().to(device)
        ts = torch.arange(start, stop, dtype=torch.float32, device=device)[None] / SSL_RATE
        valid = torch.ones((1, WINDOW), dtype=torch.bool, device=device)
        mask = torch.zeros_like(valid)
        output = model(a, b, l, ts, valid, mask)
        features = output["causal_context"][0, :-1]
        design = torch.cat((features, torch.ones((len(features), 1), device=device)), -1)
        prediction = (design @ weights).cpu().numpy().astype(np.float32)
        left, right = start + 1, stop
        accum[left:right] += prediction
        counts[left:right] += 1
    predicted = item["semantic"].copy()
    valid = counts > 0
    predicted[valid] = accum[valid] / counts[valid, None]
    return predicted


def residual_dmx(item, predicted):
    old = item["old_dmx"]
    candidate = old.copy()
    decoded = item["decoded"]
    n10 = item["n10"]
    times10 = np.arange(n10, dtype=np.float64) / SSL_RATE
    dx = np.clip(predicted[:, 0] - item["semantic"][:, 0], -.15, .15)
    dy = np.clip(predicted[:, 1] - item["semantic"][:, 1], -.15, .15)
    db = np.clip(predicted[:, 7] - item["semantic"][:, 7], -.15, .15)
    t40 = item["time40"]
    pan_delta = np.interp(t40, times10, dx) * (MAX_RESIDUAL_DEG / .15)
    tilt_delta = np.interp(t40, times10, dy) * (MAX_RESIDUAL_DEG / .15)
    dim_delta = np.interp(t40, times10, db) * (MAX_DIMMER_RESIDUAL / .15)
    moving = [i for i, fixture in enumerate(item["patch"]) if fixture["kind"] != "bulb"]
    for index in moving:
        fixture = item["patch"][index]
        angle = decoded["angles_deg"][:, index].copy()
        angle[:, 0] += pan_delta
        angle[:, 1] += tilt_delta
        ranges = np.asarray([fixture["pan_range_deg"], fixture["tilt_range_deg"]], float)
        angle = np.clip(angle, -ranges / 2, ranges / 2)
        ticks = np.rint((angle / ranges + .5) * 65535).clip(0, 65535).astype(np.uint16)
        address = fixture["address"] - 1; universe = fixture["universe"]
        candidate[:, universe, address] = (ticks[:, 0] >> 8).astype(np.uint8)
        candidate[:, universe, address + 1] = (ticks[:, 0] & 255).astype(np.uint8)
        candidate[:, universe, address + 2] = (ticks[:, 1] >> 8).astype(np.uint8)
        candidate[:, universe, address + 3] = (ticks[:, 1] & 255).astype(np.uint8)
        dimmer = np.clip(decoded["dimmer"][:, index] + dim_delta, 0, 1)
        candidate[:, universe, address + 4] = np.rint(dimmer * 255).astype(np.uint8)
    return candidate


def kinematic_report(decoded, patch):
    moving = np.asarray([p["kind"] != "bulb" for p in patch])
    angles = decoded["angles_deg"][:, moving]
    speed = np.linalg.norm(np.diff(angles, axis=0) * FRAME_RATE, axis=-1)
    acceleration = np.linalg.norm(np.diff(angles, n=2, axis=0) * FRAME_RATE ** 2, axis=-1)
    return {"speed_p99": float(np.quantile(speed, .99)), "speed_max": float(speed.max()),
            "acceleration_p99": float(np.quantile(acceleration, .99)),
            "acceleration_max": float(acceleration.max())}


def render_compare(song_root: Path, song: str, old_movie: Path, audio_path: Path) -> dict:
    candidate_movie = next(song_root.glob("*_body_to_63_P2_1080p_av1.mp4"))
    comparison = song_root / f"{song}_before_after.mp4"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(old_movie), "-i", str(candidate_movie),
           "-filter_complex", "[0:v]scale=960:540,drawtext=text='BEFORE - saved mapper 01':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[a];[1:v]scale=960:540,drawtext=text='AFTER - SSL residual':x=16:y=16:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[b];[a][b]hstack=inputs=2[v]",
           "-map", "[v]", "-map", "1:a:0", "-t", "60", "-c:v", "av1_nvenc", "-preset", "p7",
           "-b:v", "2200k", "-maxrate", "2400k", "-bufsize", "4000k", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", str(comparison)]
    subprocess.run(cmd, check=True)
    return {"path": str(comparison), "sha256": sha(comparison), "command": cmd,
            "scope": "side-by-side presentation; same saved body and source audio"}


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    require_space(Path("/mnt/lightsem-data"), 20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, config = load_ssl(device)
    weights = fit_probe(model, stats, device)
    out.mkdir(parents=True)
    write_json(out / "PROTOCOL.json", {"schema_version": 1, "status": "diagnostic_only",
        "checkpoint": str(SSL_ROOT / "checkpoint.pt"), "checkpoint_sha256": sha(SSL_ROOT / "checkpoint.pt"),
        "songs": ["shpongle", "cure_for_me"], "body_source": "saved native joint_seed0",
        "baseline": "saved mapper-01 DMX from demo_videos_v6", "color": "P2 exact from saved DMX",
        "residual_cap_deg": MAX_RESIDUAL_DEG, "dimmer_cap": MAX_DIMMER_RESIDUAL,
        "max_candidate_baseline_ratio": MAX_RATIO, "body_generation": False,
        "no_posthoc_body_smoothing": True, "promotion": "forbidden"})
    reports = []; demo_movies = []
    with gpu_lock():
        for song in ("shpongle", "cure_for_me"):
            item = song_inputs(song)
            pred = predict_semantic(model, stats, weights, item, device)
            candidate = residual_dmx(item, pred)
            baseline_decoded = decode(item["old_dmx"], item["patch"])
            candidate_decoded = decode(candidate, item["patch"])
            baseline_motion = kinematic_report(baseline_decoded, item["patch"])
            candidate_motion = kinematic_report(candidate_decoded, item["patch"])
            ratios = {key: candidate_motion[key] / max(baseline_motion[key], 1e-6)
                      for key in ("speed_p99", "speed_max", "acceleration_p99", "acceleration_max")}
            if max(ratios.values()) > MAX_RATIO:
                raise RuntimeError(f"rejected_raw_safety {song}: {ratios}")
            folder = out / song; folder.mkdir()
            np.savez_compressed(folder / "timeline.npz", time=item["time40"], body=item["world40"], universes=candidate)
            (folder / "patch.json").write_text(json.dumps(item["patch"], indent=2) + "\n")
            np.savez_compressed(folder / "SSL_RESIDUAL.npz", time10=np.arange(item["n10"])/SSL_RATE,
                                baseline_semantic=item["semantic"], predicted_semantic=pred,
                                candidate_pan_delta_deg=np.interp(item["time40"], np.arange(item["n10"])/SSL_RATE,
                                                                  np.clip(pred[:, 0]-item["semantic"][:, 0], -.15, .15)) * (MAX_RESIDUAL_DEG/.15),
                                candidate_dimmer_delta=np.interp(item["time40"], np.arange(item["n10"])/SSL_RATE,
                                                                 np.clip(pred[:, 7]-item["semantic"][:, 7], -.15, .15)) * (MAX_DIMMER_RESIDUAL/.15))
            report = {"song": song, "status": "diagnostic_only", "before_dmx_sha256": sha(item["old_timeline_path"]),
                      "candidate_dmx_sha256": sha(folder / "timeline.npz"), "body_sha256": sha(item["body_path"]),
                      "audio_sha256": sha(item["audio_path"]), "p2_sha256": sha(item["p2_path"]),
                      "baseline_motion": baseline_motion, "candidate_motion": candidate_motion,
                      "candidate_baseline_ratios": ratios, "finite_candidate": bool(np.isfinite(candidate).all()),
                      "exact_rgb_preserved": bool(np.array_equal(baseline_decoded["rgb"], candidate_decoded["rgb"])),
                      "body_generation_changed": False, "promotion": "forbidden"}
            write_json(folder / "REPORT.json", report); reports.append(report)
            track = {"id": song, "title": f"{song} — SSL residual diagnostic", "audio": str(item["audio_path"])}
            plan = {"review_start_s": 0.0, "review_duration_s": 60.0}
            from scripts.rendering.render_body_transfer import render_one
            movie = render_one(folder, track, plan)
            from scripts.motion.run_rhythm_companion import audit_movie
            audit = audit_movie(folder, movie)
            before = OLD_ROOT / "demo_videos_v6" / song / ("shpongle_body_to_63_P2_1080p_av1.mp4" if song == "shpongle" else "cure_for_me_body_to_63_P2_1080p_av1.mp4")
            comparison = render_compare(folder, song, before, item["audio_path"])
            demo_movies.append({"song": song, "before": str(before), "after": str(movie), "comparison": comparison,
                                "after_audit": audit})
    write_json(out / "DELIVERY.json", {"schema_version": 1, "status": "diagnostic_only", "reports": reports,
        "movies": demo_movies, "body_generation_changed": False, "promotion": "forbidden"})
    print(json.dumps({"out": str(out), "status": "diagnostic_only", "movies": demo_movies}, indent=2, sort_keys=True))


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--out", default=str(OUT_ROOT)); args = p.parse_args(argv); run(args)


if __name__ == "__main__":
    main()
