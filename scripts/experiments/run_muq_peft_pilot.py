#!/usr/bin/env python3
"""Run one explicit frozen-MuQ/PEFT audio-to-light pilot.

The pilot consumes authenticated cached features. It therefore trains the
downstream temporal adapter and light heads, while recording
``muq_finetuned=false``. A live MuQ backbone is required for any future
last-block or full-backbone arm.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.audio_light_foundation.data import load_rows, make_fixture_metadata, sha
from lightsem.experiments.audio_light_foundation.evaluation import metrics
from lightsem.experiments.audio_light_foundation.losses import losses
from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig, AudioToLightFoundation
from lightsem.experiments.foundation_audio_light.peft import trainable_parameter_report


ARMS = {
    "frozen_heads": "none",
    "local_adapter": "local_conv",
    "causal_adapter": "causal_transformer",
    "hierarchical_adapter": "hierarchical",
}


def _synthetic_rows(count: int = 8, frames: int = 160, dim: int = 1116) -> list[dict[str, np.ndarray]]:
    rows = []
    for index in range(count):
        rng = np.random.default_rng(index)
        audio = rng.normal(size=(frames, dim)).astype(np.float32)
        rgb = rng.uniform(0.05, 0.95, size=(frames, 8, 3)).astype(np.float32)
        brightness = rng.uniform(0.05, 0.95, size=(frames, 8, 1)).astype(np.float32)
        event = rng.integers(0, 2, size=(frames, 1)).astype(np.float32)
        rows.append({"id": f"synthetic_{index}", "audio": audio, "rgb": rgb,
                     "brightness": brightness, "event": event})
    return rows


def _windows(rows, starts, length, device):
    entries = [(row, start) for row in rows for start in starts if start + length <= len(row["audio"])]
    if not entries:
        raise ValueError("no valid training windows")

    def stack(key):
        return torch.from_numpy(np.stack([row[key][start : start + length] for row, start in entries])).float().to(device)

    return {
        "audio": stack("audio"),
        "rgb": stack("rgb"),
        "brightness": stack("brightness"),
        "event": stack("event"),
        "valid": torch.ones((len(entries), length), dtype=torch.bool, device=device),
    }, entries


def _raw_control_metrics(output: dict[str, torch.Tensor], rate_hz: float) -> dict[str, float | bool | int]:
    aim = output["aim"].detach().cpu().numpy().astype(np.float64)
    finite = bool(np.isfinite(aim).all())
    velocity = np.diff(aim, axis=1, prepend=aim[:, :1]) * rate_hz
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1]) * rate_hz
    jerk = np.diff(acceleration, axis=1, prepend=acceleration[:, :1]) * rate_hz
    streams = aim.reshape(aim.shape[0] * aim.shape[1], aim.shape[2], aim.shape[3])
    duplicate = 0
    for frame in streams:
        duplicate += frame.shape[0] - len({np.ascontiguousarray(row).tobytes() for row in frame})
    return {
        "finite": finite,
        "frames": int(aim.shape[1]),
        "fixtures": int(aim.shape[2]),
        "speed_p99": float(np.percentile(np.linalg.norm(velocity, axis=-1), 99)),
        "acceleration_p99": float(np.percentile(np.linalg.norm(acceleration, axis=-1), 99)),
        "jerk_p99": float(np.percentile(np.linalg.norm(jerk, axis=-1), 99)),
        "inactive_fraction": float(np.mean(np.linalg.norm(velocity, axis=-1) <= 1e-8)),
        "duplicate_stream_count": int(duplicate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, choices=(10, 30), default=10)
    parser.add_argument("--muq-root", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    rows = _synthetic_rows() if args.synthetic else load_rows(sample_rate_hz=args.sample_rate, muq_root=str(args.muq_root) if args.muq_root else None)
    if rows[0]["audio"].shape[-1] != 1116:
        raise ValueError("the full-contract pilot requires MuQ1024 + Base92 = 1116 features")
    train_rows, validation_rows = rows[:6], rows[6:]
    starts = (0, 32, 64, 96) if args.sample_rate == 10 else (0, 96, 192, 288)
    train, _ = _windows(train_rows, starts, args.window, device)
    validation, _ = _windows(validation_rows, starts, args.window, device)
    config = AudioToLightConfig(
        audio_dim=1116,
        local_rate_hz=float(args.sample_rate),
        adapter_kind=ARMS[args.arm],
        adapter_bottleneck=32,
        adapter_heads=4,
        adapter_local_window=min(32, args.window),
    )
    model = AudioToLightFoundation(config).to(device)
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)[None].expand(len(train["audio"]), -1, -1)
    validation_fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)[None].expand(len(validation["audio"]), -1, -1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    started = time.time()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(train["audio"]), -1) / float(args.sample_rate)
        output = model(train["audio"], fixtures, timestamps=timestamps, valid=train["valid"], duration_seconds=args.window / float(args.sample_rate))
        current = losses(output, train, train["valid"], auxiliary=False)
        current["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
            history.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in current.items()}})
    model.eval()
    with torch.no_grad():
        timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(len(validation["audio"]), -1) / float(args.sample_rate)
        validation_output = model(validation["audio"], validation_fixtures, timestamps=timestamps, valid=validation["valid"], duration_seconds=args.window / float(args.sample_rate))
        validation_losses = losses(validation_output, validation, validation["valid"], auxiliary=False)
        checks = metrics(model, validation, validation_fixtures)
        raw = _raw_control_metrics(validation_output, float(args.sample_rate))
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.out / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(), "seed": seed, "arm": args.arm}, checkpoint)
    receipt = {
        "schema": "lightsem.muq_peft_pilot.v1",
        "protocol_id": "muq-peft-pilot-20261019-v1",
        "status": "diagnostic_only",
        "arm": args.arm,
        "seed": seed,
        "steps": args.steps,
        "device": str(device),
        "train_songs": [row["id"] for row in train_rows],
        "validation_songs": [row["id"] for row in validation_rows],
        "provenance": [
            {
                "id": row["id"],
                "paths": {
                    key: {"path": value, "sha256": sha(value)}
                    for key, value in row.get("paths", {}).items()
                    if Path(value).exists()
                },
            }
            for row in rows
        ],
        "input_contract": "pinned MuQ layer9 + Base92 = 1116-D; no zero padding",
        "muq_finetuned": False,
        "feature_cache_only": not args.synthetic,
        "adapter_kind": ARMS[args.arm],
        "trainable_parameters": trainable_parameter_report(model),
        "history": history,
        "validation_losses": {key: float(value.detach().cpu()) for key, value in validation_losses.items()},
        "checks": checks,
        "raw_control_metrics": raw,
        "runtime_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "promotion": "forbidden_pending_raw_safety_conditioning_and_dmx_gates",
    }
    (args.out / "RESULTS.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.out / "CONFIG.json").write_text(json.dumps({"args": vars(args), "model": model.config_dict()}, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "arm": args.arm, "status": receipt["status"], "raw": raw}, sort_keys=True))


if __name__ == "__main__":
    main()

