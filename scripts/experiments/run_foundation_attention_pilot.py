"""Run the first one-seed continuous-attention foundation decoder pilot.

This script deliberately consumes saved, provenance-bound bodies.  It does not
generate dance or write DMX/video.  The output is a diagnostic checkpoint and
machine receipt under a new external artifact root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from lightsem.experiments.foundation_audio_light import FoundationAudioLightModel, FoundationConfig
from lightsem.experiments.foundation_audio_light.evaluation import raw_body_metrics, sha256, write_receipt
from lightsem.experiments.foundation_audio_light.losses import foundation_losses


DATA_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013")
VISUAL_ROOT = Path("/mnt/lightsem-data/cache/beatsaver_visual")
LIGHT_ROOT = Path("/mnt/lightsem-data/cache/beatsaver_light")
AUDIO_ROOT = Path("/mnt/lightsem-data/cache/beatsaver_audio")
SONG_IDS = ("1023", "10dcc", "1181f", "11b01", "1672d", "210e3", "21c3a", "3b4ab")


def _find_cache(root: Path, suffix: str):
    paths = sorted(root.glob(f"*={suffix}.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one {root} cache for {suffix}, found {len(paths)}")
    return paths[0]


def _body_features(world):
    # Root-relative body coordinates keep the decoder from learning the global
    # stage origin.  The source itself remains the saved native C body.
    world = np.asarray(world, dtype=np.float32)
    rel = world - world[:, :1]
    return rel.reshape(len(rel), -1)


def _downsample_light(path, n):
    z = np.load(path, allow_pickle=False)
    rgb = np.asarray(z["rgb"], dtype=np.float32)
    inten = np.asarray(z["inten"], dtype=np.float32)
    blocks = min(n, rgb.shape[1] // 4)
    rgb = rgb[:, :blocks * 4].reshape(rgb.shape[0], blocks, 4, 3).mean(2).transpose(1, 0, 2)
    inten = inten[:, :blocks * 4].reshape(inten.shape[0], blocks, 4).mean(2).transpose(1, 0)[..., None]
    # The pilot has eight virtual fixtures.  Repeating authored groups is an
    # explicit arbitrary-N diagnostic, not a claim of eight independent cues.
    fixture_group = np.arange(8) % max(1, rgb.shape[1])
    return rgb[:blocks, fixture_group], np.clip(inten[:blocks, fixture_group], 0, 1)


def load_song(suffix):
    body_path = DATA_ROOT / f"body_{suffix}" / "body.npz"
    meta_path = DATA_ROOT / f"body_{suffix}" / "body.json"
    visual_path = _find_cache(VISUAL_ROOT, suffix)
    light_path = _find_cache(LIGHT_ROOT, suffix)
    audio_path = _find_cache(AUDIO_ROOT, suffix)
    body = np.load(body_path, allow_pickle=False)
    visual = np.load(visual_path, allow_pickle=False)
    # The raw body safety audit happens before any decoder or fixture mapping.
    body_metrics = raw_body_metrics(body["world"])
    if not body_metrics["finite"]:
        raise RuntimeError(f"nonfinite saved body rejected: {suffix}")
    body_10 = _body_features(body["world"])[::3]
    # Use the pinned MuQ layer-9 timeline bound in body.json.  The 10 Hz pilot
    # clock selects nearest native MuQ timestamps; it does not manufacture a
    # higher-rate representation or substitute the BeatSaver bus.
    timeline_path = Path(json.loads(meta_path.read_text())["metadata"]["audio"]["timeline"])
    timeline = np.load(timeline_path, allow_pickle=False)
    muq = np.asarray(timeline["muq"], dtype=np.float32)
    muq_time = np.asarray(timeline["muq_time"], dtype=np.float32)
    audio = muq
    scene = np.asarray(visual["y"], dtype=np.float32)
    event = np.asarray(visual["ev_any"], dtype=np.float32)[..., None]
    n = min(len(body_10), len(scene), len(event), int(np.floor(float(muq_time[-1]) * 10)) + 1)
    target_times = np.arange(n, dtype=np.float32) / 10.0
    right = np.searchsorted(muq_time, target_times, side="left").clip(0, len(muq_time) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(muq_time[left] - target_times) <= np.abs(muq_time[right] - target_times)
    nearest = np.where(choose_left, left, right)
    audio = audio[nearest]
    rgb, brightness = _downsample_light(light_path, n)
    n = min(n, len(rgb))
    return {
        "id": suffix,
        "audio": audio[:n],
        "body": body_10[:n],
        "scene": scene[:n],
        "event": event[:n],
        "rgb": rgb[:n],
        "brightness": brightness[:n],
        "body_metrics": body_metrics,
        "paths": {"body": str(body_path), "body_meta": str(meta_path),
                  "visual": str(visual_path), "light": str(light_path),
                  "audio": str(audio_path), "muq_timeline": str(timeline_path)},
        "body_meta": json.loads(meta_path.read_text()),
    }


def fixture_metadata(n=8):
    rows = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        rows.append([np.cos(theta), np.sin(theta), 0.5, 1., 0., 0., 1., 1.])
    return np.asarray(rows, dtype=np.float32)


def batch(rows, window, device):
    # Fixed leading windows make the pilot deterministic and keep the receipt
    # easy to replay.  Held-out rows are never sampled into train windows.
    rows = [r for r in rows if len(r["audio"]) >= window]
    if not rows:
        raise RuntimeError("no song has enough frames for the requested window")
    def stack(name):
        return torch.from_numpy(np.stack([r[name][:window] for r in rows])).to(device)
    return {
        "audio": stack("audio").float(), "body": stack("body").float(),
        "scene": stack("scene").float(), "event": stack("event").float(),
        "rgb": stack("rgb").float(), "brightness": stack("brightness").float(),
        "valid": torch.ones((len(rows), window), dtype=torch.bool, device=device),
    }, rows


def run_chunked(model, data, fixtures, chunk, duration):
    state = None
    scenes = []
    with torch.no_grad():
        for start in range(0, data["audio"].shape[1], chunk):
            end = min(data["audio"].shape[1], start + chunk)
            out = model(data["audio"][:, start:end], data["body"][:, start:end], fixtures,
                        timestamps=torch.arange(start, end, device=data["audio"].device, dtype=torch.float32)[None].expand(data["audio"].shape[0], -1) / 10.,
                        valid=data["valid"][:, start:end], state=state,
                        duration_seconds=duration)
            state = out["state"]
            scenes.append(out["scene"])
    return torch.cat(scenes, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--out", default="/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_20261015_v1")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows = [load_song(s) for s in SONG_IDS]
    train_rows, val_rows = rows[:6], rows[6:]
    train, train_used = batch(train_rows, args.window, device)
    val, val_used = batch(val_rows, args.window, device)
    fixtures = torch.from_numpy(fixture_metadata()).to(device)
    config = FoundationConfig(audio_dim=int(train["audio"].shape[-1]))
    model = FoundationAudioLightModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    started = time.time()
    model.train()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(train["audio"], train["body"], fixtures[None].expand(len(train_used), -1, -1),
                    timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(train_used), -1) / 10.,
                    valid=train["valid"], duration_seconds=args.window / 10.)
        losses = foundation_losses(out, train, train["valid"])
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            history.append({"step": step + 1, **{k: float(v.detach().cpu()) for k, v in losses.items()}})
    model.eval()
    with torch.no_grad():
        pred = model(val["audio"], val["body"], fixtures[None].expand(len(val_used), -1, -1),
                     timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(val_used), -1) / 10.,
                     valid=val["valid"], duration_seconds=args.window / 10.)
        val_losses = foundation_losses(pred, val, val["valid"])
        full_scene = pred["scene"]
        chunk_scene = run_chunked(model, val, fixtures[None].expand(len(val_used), -1, -1), args.chunk, args.window / 10.)
        chunk_error = float((full_scene - chunk_scene).abs().max().cpu())
        perm = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4], device=device)
        p1 = model(val["audio"], val["body"], fixtures[None].expand(len(val_used), -1, -1),
                   timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(val_used), -1) / 10.,
                   valid=val["valid"], duration_seconds=args.window / 10.)
        p2 = model(val["audio"], val["body"], fixtures[perm][None].expand(len(val_used), -1, -1),
                   timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(val_used), -1) / 10.,
                   valid=val["valid"], duration_seconds=args.window / 10.)
        inverse = torch.argsort(perm)
        perm_error = float((p1["brightness"] - p2["brightness"][:, :, inverse]).abs().max().cpu())
        frozen = val["body"][:, :1].expand_as(val["body"])
        frozen_out = model(val["audio"], frozen, fixtures[None].expand(len(val_used), -1, -1),
                           timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(val_used), -1) / 10.,
                           valid=val["valid"], duration_seconds=args.window / 10.)
        frozen_angle = float(frozen_out["angle"].abs().max().cpu())
        perturbed = val["body"].clone(); perturbed[..., -3:] += 0.05
        perturbed_out = model(val["audio"], perturbed, fixtures[None].expand(len(val_used), -1, -1),
                              timestamps=torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(val_used), -1) / 10.,
                              valid=val["valid"], duration_seconds=args.window / 10.)
        limb_response = float((p1["angle"] - perturbed_out["angle"]).abs().mean().cpu())
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(), "seed": seed}, out_root / "model.pt")
    provenance = []
    for row in rows:
        provenance.append({"id": row["id"], "body_metrics": row["body_metrics"],
                           "paths": {k: {"path": v, "sha256": sha256(v)} for k, v in row["paths"].items()},
                           "body_metadata": row["body_meta"]["metadata"]})
    result = {
        "protocol_id": "foundation-audio-light-continuous-attention-20261015-v1",
        "status": "diagnostic_only",
        "seed": seed, "device": str(device), "steps": args.steps,
        "train_songs": [r["id"] for r in train_used], "validation_songs": [r["id"] for r in val_used],
        "train_history": history,
        "validation_losses": {k: float(v.detach().cpu()) for k, v in val_losses.items()},
        "checks": {"chunk_scene_max_abs": chunk_error, "fixture_permutation_max_abs": perm_error,
                   "frozen_body_angle_max_abs": frozen_angle, "assigned_limb_angle_response": limb_response},
        "runtime_seconds": time.time() - started,
        "model_config": model.config_dict(), "provenance": provenance,
        "safety_note": "saved native 30 Hz bodies were audited before decoder use; no new body, DMX or video was generated",
        "promotion": "forbidden: one seed, eight diagnostic songs, no decoded-DMX safety receipt",
    }
    write_receipt(out_root / "RESULTS.json", result)
    (out_root / "CONFIG.json").write_text(json.dumps({"args": vars(args), "model": model.config_dict()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out_root), "status": result["status"], "checks": result["checks"], "validation": result["validation_losses"]}, indent=2))


if __name__ == "__main__":
    main()
