#!/usr/bin/env python3
"""Train and audit the controlled body-to-light SSL loop on real prepared rows.

This is a one-seed representation experiment.  It produces a held-out semantic
light prediction trace and plot, not a native body, DMX universe, or product
demo.  The raw native bodies remain inputs and are audited separately.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.audio_light_foundation.data import IDS, load_rows, sha
from lightsem.experiments.foundation_audio_light.evaluation import raw_body_metrics
from lightsem.experiments.body_light_ssl import (
    ControlledJepaConfig,
    ControlledSSLLoop,
    audit_input_sensitivity,
    controlled_ssl_loss,
    make_mask,
)
from lightsem.experiments.stage11.stage11_common import gpu_lock, require_space


SEED = 0
TRAIN_IDS = IDS[:6]
VAL_IDS = IDS[6:]
FREE_FLOOR_GB = 20


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"preserving existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(
        json.dumps({"sha256": sha(path)}, sort_keys=True) + "\n"
    )


def semantic_light(row: dict) -> np.ndarray:
    """Pack fixture controls into a normalized, fixture-order-free state."""
    def pooled(name: str) -> np.ndarray:
        value = np.asarray(row[name], dtype=np.float32)
        if value.ndim == 3:
            value = value.mean(axis=1)
        return value

    return np.concatenate(
        (pooled("motion"), pooled("speed"), pooled("acceleration"),
         pooled("rgb"), pooled("brightness"), np.asarray(row["event"], np.float32)),
        axis=-1,
    ).astype(np.float32)


def prepare_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        audio = np.asarray(row["audio"], dtype=np.float32)
        body = np.asarray(row["body"], dtype=np.float32)
        light = semantic_light(row)
        n = min(len(audio), len(body), len(light))
        prepared.append({"id": row["id"], "audio": audio[:n], "body": body[:n],
                         "light": light[:n], "raw_world": row["raw_world"],
                         "paths": row["paths"]})
    return prepared


def normalization(rows: list[dict]) -> dict[str, np.ndarray]:
    stats = {}
    for key in ("audio", "body", "light"):
        values = np.concatenate([row[key] for row in rows], axis=0)
        stats[key + "_mean"] = values.mean(axis=0).astype(np.float32)
        stats[key + "_std"] = np.maximum(values.std(axis=0), 1e-4).astype(np.float32)
    return stats


def normalize(row: dict, stats: dict[str, np.ndarray]) -> dict:
    return {key: (row[key] - stats[key + "_mean"]) / stats[key + "_std"]
            for key in ("audio", "body", "light")}


def sample_batch(rows: list[dict], stats: dict[str, np.ndarray], *, batch_size: int,
                 frames: int, rng: np.random.Generator, device: torch.device) -> tuple:
    values = {"audio": [], "body": [], "light": [], "timestamps": []}
    for _ in range(batch_size):
        row = rows[int(rng.integers(len(rows)))]
        start = int(rng.integers(max(1, len(row["audio"]) - frames + 1)))
        stop = start + frames
        if stop > len(row["audio"]):
            start = len(row["audio"]) - frames
            stop = len(row["audio"])
        norm = normalize(row, stats)
        for key in ("audio", "body", "light"):
            values[key].append(norm[key][start:stop])
        values["timestamps"].append(np.arange(start, stop, dtype=np.float32) / 10.0)
    tensors = [torch.from_numpy(np.stack(values[key])).float().to(device)
               for key in ("audio", "body", "light", "timestamps")]
    valid = torch.ones((batch_size, frames), dtype=torch.bool, device=device)
    return (*tensors, valid)


def deterministic_windows(rows: list[dict], stats: dict[str, np.ndarray], *, frames: int,
                          device: torch.device) -> list[tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    windows = []
    for row in rows:
        norm = normalize(row, stats)
        starts = list(range(0, max(1, len(row["audio"]) - frames + 1), frames))
        if not starts or starts[-1] + frames < len(row["audio"]):
            starts.append(max(0, len(row["audio"]) - frames))
        for start in sorted(set(starts)):
            stop = start + frames
            if stop > len(row["audio"]):
                continue
            tensors = [torch.from_numpy(norm[key][start:stop][None]).float().to(device)
                       for key in ("audio", "body", "light")]
            timestamps = torch.from_numpy((np.arange(start, stop, dtype=np.float32) / 10.0)[None]).to(device)
            valid = torch.ones((1, frames), dtype=torch.bool, device=device)
            windows.append((row, *tensors, timestamps, valid))
    return windows


@torch.no_grad()
def alignment_metrics(model, windows, *, mask_ratio: float, seed: int) -> dict[str, float]:
    model.eval()
    masked, future, count = [], [], 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _, audio, body, light, timestamps, valid in windows:
        mask = make_mask(valid, mask_ratio, generator=generator)
        outputs = model(audio, body, light, timestamps, valid, mask)
        masked_mask = outputs["mask"] & outputs["valid"]
        p = torch.nn.functional.normalize(outputs["predicted"], dim=-1)
        t = torch.nn.functional.normalize(outputs["target"], dim=-1)
        masked.append(float((p[masked_mask] * t[masked_mask]).sum(-1).mean()))
        future_mask = outputs["valid"][:, 1:] & outputs["valid"][:, :-1]
        fp = torch.nn.functional.normalize(outputs["future_predicted"], dim=-1)
        ft = torch.nn.functional.normalize(outputs["future_target"], dim=-1)
        future.append(float((fp[future_mask] * ft[future_mask]).sum(-1).mean()))
        count += int(masked_mask.sum())
    return {"masked_cosine": float(np.mean(masked)), "future_cosine": float(np.mean(future)),
            "masked_frames": count}


@torch.no_grad()
def collect_probe_data(model, windows):
    model.eval()
    features, light_targets, body_targets = [], [], []
    zero_mask = None
    for _, audio, body, light, timestamps, valid in windows:
        if zero_mask is None or zero_mask.shape != valid.shape:
            zero_mask = torch.zeros_like(valid)
        outputs = model(audio, body, light, timestamps, valid, zero_mask)
        features.append(outputs["causal_context"][:, :-1].reshape(-1, outputs["causal_context"].shape[-1]))
        light_targets.append(light[:, 1:].reshape(-1, light.shape[-1]))
        body_targets.append((body[:, 1:] - body[:, :-1]).reshape(-1, body.shape[-1]))
    return (torch.cat(features), torch.cat(light_targets), torch.cat(body_targets))


def ridge_probe(train_data, val_data, ridge: float = 1e-2) -> dict[str, float]:
    x_train, y_light_train, y_body_train = train_data
    x_val, y_light_val, y_body_val = val_data
    def fit_predict(y_train, y_val):
        ones_train = torch.ones((len(x_train), 1), device=x_train.device, dtype=x_train.dtype)
        ones_val = torch.ones((len(x_val), 1), device=x_val.device, dtype=x_val.dtype)
        design = torch.cat((x_train, ones_train), dim=-1)
        regularizer = torch.eye(design.shape[-1], device=design.device, dtype=design.dtype) * ridge
        regularizer[-1, -1] = 0
        weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ y_train)
        prediction = torch.cat((x_val, ones_val), dim=-1) @ weights
        return float(torch.mean((prediction - y_val).square()))
    return {"future_light_probe_mse": fit_predict(y_light_train, y_light_val),
            "future_body_velocity_probe_mse": fit_predict(y_body_train, y_body_val)}


def plot_demo(model, train_windows, val_row, val_window, out: Path, *, seed: int) -> dict:
    import matplotlib.pyplot as plt
    train_data = collect_probe_data(model, train_windows)
    _, audio, body, light, timestamps, valid = val_window
    val_data = collect_probe_data(model, [val_window])
    x_train, y_light_train, _ = train_data
    x_val, _, _ = val_data
    design = torch.cat((x_train, torch.ones((len(x_train), 1), device=x_train.device)), -1)
    reg = torch.eye(design.shape[-1], device=design.device) * 1e-2; reg[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + reg, design.T @ y_light_train)
    pred = torch.cat((x_val, torch.ones((len(x_val), 1), device=x_val.device)), -1) @ weights
    target = light[:, 1:].reshape(-1, light.shape[-1])
    body_speed = torch.linalg.vector_norm(body[:, 1:] - body[:, :-1], dim=-1).reshape(-1).cpu().numpy()
    seconds = timestamps[:, 1:].reshape(-1).cpu().numpy()
    actual = target.cpu().numpy(); predicted = pred.cpu().numpy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(seconds, body_speed, color="#303f9f", lw=1.2); axes[0].set_ylabel("body Δ")
    axes[1].plot(seconds, actual[:, 7], label="actual brightness", color="#111111", lw=1.0)
    axes[1].plot(seconds, predicted[:, 7], label="SSL probe", color="#d84315", lw=1.0); axes[1].set_ylabel("brightness"); axes[1].legend()
    axes[2].plot(seconds, actual[:, 0], label="actual aim-x", color="#00695c", lw=1.0)
    axes[2].plot(seconds, predicted[:, 0], label="SSL probe", color="#8e24aa", lw=1.0); axes[2].set_ylabel("aim-x"); axes[2].set_xlabel("seconds"); axes[2].legend()
    fig.suptitle(f"Controlled body-light SSL diagnostic — held-out {val_row['id']}")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return {"plot": str(out), "song_id": val_row["id"], "frames": int(len(seconds)),
            "label": "semantic-light probe; no native DMX and no body generation"}


def run(args) -> dict:
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"artifact root already exists: {out}")
    require_space(Path("/mnt/lightsem-data"), FREE_FLOOR_GB)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rows = prepare_rows(load_rows())
    train_rows = [row for row in rows if row["id"] in TRAIN_IDS]
    val_rows = [row for row in rows if row["id"] in VAL_IDS]
    shortest_train = min(len(row["audio"]) for row in train_rows)
    if args.frames > shortest_train:
        raise ValueError(f"frames={args.frames} exceeds shortest train row ({shortest_train}); use padding as a new protocol")
    stats = normalization(train_rows)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    config = ControlledJepaConfig(audio_dim=train_rows[0]["audio"].shape[-1], body_dim=train_rows[0]["body"].shape[-1],
                                  light_dim=train_rows[0]["light"].shape[-1], width=args.width,
                                  embedding_dim=args.embedding_dim, heads=args.heads, layers=args.layers,
                                  mask_ratio=args.mask_ratio)
    loop = ControlledSSLLoop(config, seed=SEED, learning_rate=args.learning_rate)
    loop.model.to(device)
    initial_state = copy.deepcopy(loop.model.state_dict())
    rng = np.random.default_rng(SEED + 1)
    started = time.monotonic(); history = []
    with gpu_lock() if device.type == "cuda" else _nullcontext():
        for step in range(args.steps):
            batch = sample_batch(train_rows, stats, batch_size=args.batch_size, frames=args.frames, rng=rng, device=device)
            metrics = loop.step(*batch)
            if step == 0 or (step + 1) % args.log_every == 0:
                history.append(metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
    train_windows = deterministic_windows(train_rows, stats, frames=args.frames, device=device)
    val_windows = deterministic_windows(val_rows, stats, frames=args.frames, device=device)
    candidate_alignment = alignment_metrics(loop.model, val_windows, mask_ratio=args.mask_ratio, seed=SEED + 2)
    baseline_model = copy.deepcopy(loop.model)
    baseline_model.load_state_dict(initial_state, strict=True)
    baseline_alignment = alignment_metrics(baseline_model, val_windows, mask_ratio=args.mask_ratio, seed=SEED + 2)
    candidate_probe = ridge_probe(collect_probe_data(loop.model, train_windows), collect_probe_data(loop.model, val_windows))
    baseline_probe = ridge_probe(collect_probe_data(baseline_model, train_windows), collect_probe_data(baseline_model, val_windows))
    sensitivity = audit_input_sensitivity(loop.model, *val_windows[0][1:])
    body_metrics = {row["id"]: raw_body_metrics(row["raw_world"]) for row in rows}
    out.mkdir(parents=True)
    torch.save({"state_dict": loop.model.state_dict(), "config": loop.model.config_dict(),
                "optimizer": loop.optimizer.state_dict(), "steps": loop.steps, "seed": SEED,
                "stats": stats, "train_ids": TRAIN_IDS, "val_ids": VAL_IDS}, out / "checkpoint.pt")
    np.savez_compressed(out / "NORMALIZATION.npz", **stats)
    write_json(out / "PROTOCOL.json", {"schema_version": 1, "status": "complete", "seed": SEED,
        "steps": args.steps, "frames": args.frames, "batch_size": args.batch_size,
        "train_ids": TRAIN_IDS, "validation_ids": VAL_IDS, "model": config.as_dict(),
        "inputs": ["pinned audio features", "audited native body", "semantic light state"],
        "forbidden": ["raw DMX input", "body replacement", "procedural motion", "posthoc smoothing"],
        "free_space_gb": shutil_free_space(), "device": str(device)})
    result = {"schema_version": 1, "status": "diagnostic_only", "seed": SEED,
              "steps": args.steps, "device": str(device), "elapsed_s": time.monotonic() - started,
              "train_ids": TRAIN_IDS, "validation_ids": VAL_IDS,
              "baseline_alignment": baseline_alignment, "candidate_alignment": candidate_alignment,
              "alignment_delta": {key: candidate_alignment[key] - baseline_alignment[key] for key in ("masked_cosine", "future_cosine")},
              "baseline_probe": baseline_probe, "candidate_probe": candidate_probe,
              "probe_delta": {key: candidate_probe[key] - baseline_probe[key] for key in candidate_probe},
              "input_sensitivity": sensitivity, "raw_body_metrics": body_metrics,
              "body_generation_changed": False, "promotion": "forbidden",
              "history": history, "checkpoint": str(out / "checkpoint.pt"),
              "checkpoint_sha256": sha(out / "checkpoint.pt")}
    write_json(out / "RESULTS.json", result)
    demo = plot_demo(loop.model, train_windows, val_windows[0][0], val_windows[0], out / "semantic_light_demo.png", seed=SEED)
    write_json(out / "DEMO.json", {"schema_version": 1, "status": "diagnostic_only", **demo})
    print(json.dumps({"status": result["status"], "out": str(out), "results": result, "demo": demo}, indent=2, sort_keys=True))
    return result


def shutil_free_space() -> float:
    import shutil
    return float(shutil.disk_usage("/mnt/lightsem-data").free / 2**30)


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--frames", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=.35)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
