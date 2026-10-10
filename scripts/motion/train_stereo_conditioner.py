#!/usr/bin/env python3
"""Train a compact StereoAudioV1 fused or factorized pilot arm.

This driver trains descriptor probes only.  It does not alter the native-C
checkpoint or declare descriptor reconstruction to be source truth.  Native
integration is exercised by the separate zero-initialized adapter tests and a
future guarded native campaign.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import StereoAudioV1, sha256
from lightsem.experiments.audio_stereo.model import (
    StereoConditioner, StereoConditionerConfig, conditioner_loss,
)


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserving existing training artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


class _Records:
    def __init__(self, manifest_path: Path, split: str = "train"):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("test_access", True):
            raise ValueError("Stereo training manifest must prove test_access=false")
        rows = [r for r in manifest.get("entries", []) if r.get("split") == split]
        if not rows:
            raise ValueError(f"No StereoAudioV1 rows for split {split!r}")
        self.rows = []
        for row in rows:
            descriptor = StereoAudioV1.load(row["feature_path"])
            if descriptor.metadata.get("split") == "test":
                raise ValueError("TEST descriptor reached training loader")
            self.rows.append(descriptor)
        dims = {(r.feature_dim, r.event_dim, r.spatial_10.shape[-1], r.relation_2.shape[-1]) for r in self.rows}
        if len(dims) != 1:
            raise ValueError("Stereo feature dimensions differ across records")
        self.dims = next(iter(dims))

    def batch(self, rng: np.random.Generator, batch_size: int, frames: int) -> tuple[np.ndarray, ...]:
        x, event, spatial, relation = [], [], [], []
        for _ in range(batch_size):
            record = self.rows[int(rng.integers(len(self.rows)))]
            if len(record.time_40) < frames:
                start = 0
            else:
                start = int(rng.integers(len(record.time_40) - frames + 1))
            stop = start + frames
            x.append(_pad(record.features_40[start:stop], frames))
            event.append(_pad(record.events_40[start:stop], frames))
            spatial_start, spatial_stop = start // 4, (stop + 3) // 4
            relation_start, relation_stop = start // 20, (stop + 19) // 20
            spatial.append(_pad(record.spatial_10[spatial_start:spatial_stop], (frames + 3) // 4))
            relation.append(_pad(record.relation_2[relation_start:relation_stop], (frames + 19) // 20))
        return tuple(np.asarray(value, dtype=np.float32) for value in (x, event, spatial, relation))


def _pad(value: np.ndarray, length: int) -> np.ndarray:
    if len(value) >= length:
        return value[:length]
    if len(value) == 0:
        raise ValueError("Cannot pad an empty stereo stream")
    return np.pad(value, ((0, length - len(value)), (0, 0)), mode="edge")


def train(manifest_path: Path, out: Path, *, arm: str, steps: int, batch_size: int,
          frames: int, seed: int, device: str, split: str) -> dict:
    if arm not in ("fused", "factorized"):
        raise ValueError("Pilot arm must be fused or factorized")
    if steps <= 0 or batch_size <= 0 or frames < 20:
        raise ValueError("Invalid training dimensions")
    records = _Records(manifest_path, split=split)
    config = StereoConditionerConfig(arm=arm, input_dim=records.dims[0], event_dim=records.dims[1],
                                     spatial_dim=records.dims[2], relation_dim=records.dims[3])
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = StereoConditioner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    started = time.monotonic()
    losses = []
    model.train()
    for step in range(steps):
        arrays = records.batch(np_rng, batch_size, frames)
        tensors = [torch.from_numpy(value).to(device) for value in arrays]
        x, event, spatial, relation = tensors
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, valid)
                loss, parts = conditioner_loss(outputs, event, spatial, relation, valid)
        else:
            outputs = model(x, valid)
            loss, parts = conditioner_loss(outputs, event, spatial, relation, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite StereoAudioV1 loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    payload = {
        "model": model.state_dict(), "config": model.config_dict(),
        "optimizer": optimizer.state_dict(), "steps": steps, "seed": seed,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "split": split,
    }
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "final.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Preserving existing checkpoint: {checkpoint}")
    torch.save(payload, checkpoint)
    report = {
        "schema_version": 1, "status": "complete", "arm": arm, "seed": seed,
        "steps": steps, "batch_size": batch_size, "frames": frames,
        "device": device, "elapsed_s": time.monotonic() - started,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses), "finite": True, "test_access": False,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "split": split,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "model_config": model.config_dict(),
        "interpretation": "descriptor reconstruction probe; not source truth or a qualified dance model",
    }
    write_exclusive(out / "REPORT.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("fused", "factorized"), required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "dev", "val", "showcase"), default="train")
    args = parser.parse_args(argv)
    result = train(args.manifest, args.out, arm=args.arm, steps=args.steps,
                   batch_size=args.batch_size, frames=args.frames,
                   seed=args.seed, device=args.device, split=args.split)
    print(json.dumps({k: result[k] for k in ("status", "arm", "steps", "loss_first", "loss_last")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
