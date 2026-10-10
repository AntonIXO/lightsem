#!/usr/bin/env python3
"""Train a source-aware stereo conditioner on MUSDB TRAIN/DEV targets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import StereoAudioV1, sha256
from lightsem.experiments.audio_stereo.semantic import (
    SemanticStereoConditioner, SemanticStereoConditionerConfig,
    semantic_conditioner_loss,
)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing semantic training artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


def _stats(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate([np.asarray(x, dtype=np.float64) for x in arrays], axis=0)
    return joined.mean(0).astype(np.float32), np.maximum(joined.std(0), 1e-3).astype(np.float32)


class Records:
    def __init__(self, manifest_path: Path, split: str):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("test_access", True):
            raise ValueError("Semantic training manifest must prove test_access=false")
        rows = [row for row in manifest["entries"] if row.get("split") == split]
        if not rows:
            raise ValueError(f"No semantic targets for split {split!r}")
        self.rows = []
        for row in rows:
            descriptor = StereoAudioV1.load(row["mixture_feature_path"])
            with np.load(row["target_path"], allow_pickle=False) as target:
                if not np.array_equal(target["time_40"], descriptor.time_40):
                    raise ValueError(f"semantic target clock mismatch: {row['id']}")
                self.rows.append({
                    "id": row["id"], "features": descriptor.features_40.astype(np.float32),
                    "event": target["source_event_40"].astype(np.float32),
                    "spatial": target["source_spatial_10"].astype(np.float32),
                    "relation": target["source_relation_2"].astype(np.float32),
                })
        self.input_mean, self.input_std = _stats([row["features"] for row in self.rows])
        self.event_mean, self.event_std = _stats([row["event"] for row in self.rows])
        self.spatial_mean, self.spatial_std = _stats([row["spatial"] for row in self.rows])
        self.relation_mean, self.relation_std = _stats([row["relation"] for row in self.rows])
        self.onset_threshold = np.asarray([
            np.percentile(np.concatenate([row["event"][:, 2 + stem * 4] for row in self.rows]), 85)
            for stem in range(4)
        ], dtype=np.float32)

    def batch(self, rng: np.random.Generator, batch_size: int, frames: int) -> tuple[np.ndarray, ...]:
        values = [[] for _ in range(4)]
        for _ in range(batch_size):
            row = self.rows[int(rng.integers(len(self.rows)))]
            start = 0 if len(row["features"]) <= frames else int(rng.integers(len(row["features"]) - frames + 1))
            stop = start + frames
            event_start, spatial_start, relation_start = start // 4, start // 20, start // 20
            values[0].append(self._pad(row["features"][start:stop], frames))
            values[1].append(self._pad(row["event"][start:stop], frames))
            values[2].append(self._pad(row["spatial"][spatial_start:(stop + 3) // 4], (frames + 3) // 4))
            values[3].append(self._pad(row["relation"][relation_start:(stop + 19) // 20], (frames + 19) // 20))
        return tuple(np.asarray(value, dtype=np.float32) for value in values)

    @staticmethod
    def _pad(value: np.ndarray, length: int) -> np.ndarray:
        if len(value) >= length:
            return value[:length]
        if len(value) == 0:
            raise ValueError("Cannot pad empty semantic target")
        return np.pad(value, ((0, length - len(value)), (0, 0)), mode="edge")


def _lr_swap_batch(features: np.ndarray, event: np.ndarray, spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the stereo metamorphic transform while preserving source labels."""
    x = features.copy(); e = event.copy(); s = spatial.copy()
    names = list(__import__("lightsem.audio.stereo_v1", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES)
    def swap_features(a: str, b: str) -> None:
        ia, ib = names.index(a), names.index(b); x[..., ia], x[..., ib] = features[..., ib], features[..., ia]
    for a, b in (("left_log_rms", "right_log_rms"), ("left_centroid", "right_centroid"),
                 ("left_rolloff", "right_rolloff")):
        swap_features(a, b)
    for i in range(5): swap_features(f"flux_left_band_{i}", f"flux_right_band_{i}")
    for name in ("ild_db", "pan_balance", "phase_sin"):
        x[..., names.index(name)] *= -1.
    for i in range(5): x[..., names.index(f"energy_delta_band_{i}")] *= -1.
    # Source event blocks are [log energy, active, left onset, right onset].
    for stem in range(4): e[..., stem * 4 + 2], e[..., stem * 4 + 3] = event[..., stem * 4 + 3], event[..., stem * 4 + 2]
    # Source spatial blocks are [ILD, width, pan, coherence, log energy].
    for stem in range(4):
        s[..., stem * 5] *= -1.; s[..., stem * 5 + 2] *= -1.
    return x, e, s


def train(manifest_path: Path, out: Path, *, split: str, steps: int, batch_size: int,
          frames: int, seed: int, device: str, lr_augment: bool) -> dict:
    if out.exists():
        raise FileExistsError(f"Preserving existing semantic conditioner root: {out}")
    records = Records(manifest_path, split)
    config = SemanticStereoConditionerConfig()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SemanticStereoConditioner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    started = time.monotonic(); losses = []
    model.train()
    stats = {
        "input_mean": records.input_mean, "input_std": records.input_std,
        "event_mean": records.event_mean, "event_std": records.event_std,
        "spatial_mean": records.spatial_mean, "spatial_std": records.spatial_std,
        "relation_mean": records.relation_mean, "relation_std": records.relation_std,
        "onset_threshold": records.onset_threshold,
    }
    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).to(device)
    for step in range(steps):
        x, event_raw, spatial_raw, relation_raw = records.batch(rng, batch_size, frames)
        original_x = x.copy(); original_event = event_raw.copy(); original_spatial = spatial_raw.copy()
        if lr_augment:
            swapped_x_raw, swapped_event_raw, swapped_spatial_raw = _lr_swap_batch(
                original_x, original_event, original_spatial)
        else:
            swapped_x_raw = swapped_event_raw = swapped_spatial_raw = None
        def normalise(x_raw, event_raw_value, spatial_raw_value, relation_raw_value):
            x_value = (tensor(x_raw) - tensor(records.input_mean)) / tensor(records.input_std)
            activity = tensor(event_raw_value[..., 1::4])
            event_tensor = tensor(event_raw_value)
            onset_values = event_tensor.reshape(*event_tensor.shape[:-1], 4, 4)[..., 2:4].reshape(
                *event_tensor.shape[:-1], 8)
            onset_threshold = tensor(np.repeat(records.onset_threshold, 2))
            onset = (onset_values > onset_threshold).float()
            event_value = (tensor(event_raw_value) - tensor(records.event_mean)) / tensor(records.event_std)
            spatial_value = (tensor(spatial_raw_value) - tensor(records.spatial_mean)) / tensor(records.spatial_std)
            relation_value = (tensor(relation_raw_value) - tensor(records.relation_mean)) / tensor(records.relation_std)
            return x_value, event_value, spatial_value, relation_value, activity, onset
        x, event, spatial, relation, activity_target, onset_target = normalise(
            original_x, original_event, original_spatial, relation_raw)
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
        optimizer.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, valid)
                loss, parts = semantic_conditioner_loss(outputs, event, spatial, relation, valid,
                                                        activity_target=activity_target, onset_target=onset_target)
                if lr_augment:
                    sx, se, ss, sr, sa, so = normalise(swapped_x_raw, swapped_event_raw,
                                                        swapped_spatial_raw, relation_raw)
                    swapped_outputs = model(sx, valid)
                    swapped_loss, _ = semantic_conditioner_loss(swapped_outputs, se, ss, sr, valid,
                                                                activity_target=sa, onset_target=so)
                    consistency = (
                        torch.nn.functional.mse_loss(outputs["activity_logits"], swapped_outputs["activity_logits"]) +
                        torch.nn.functional.mse_loss(outputs["onset_logits"], swapped_outputs["onset_logits"]) +
                        torch.nn.functional.mse_loss(outputs["relation"], swapped_outputs["relation"]))
                    loss = .5 * (loss + swapped_loss) + .1 * consistency
        else:
            outputs = model(x, valid)
            loss, parts = semantic_conditioner_loss(outputs, event, spatial, relation, valid,
                                                    activity_target=activity_target, onset_target=onset_target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite semantic loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    out.mkdir(parents=True)
    checkpoint = out / "final.pt"
    torch.save({"model": model.state_dict(), "config": model.config_dict(), "stats": stats,
                "steps": steps, "seed": seed, "split": split,
                "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)}, checkpoint)
    report = {"schema_version": 1, "status": "complete", "arm": config.arm, "seed": seed,
              "steps": steps, "batch_size": batch_size, "frames": frames, "device": device,
              "elapsed_s": time.monotonic() - started, "loss_first": losses[0],
              "loss_last": losses[-1], "loss_min": min(losses), "finite": True,
              "test_access": False, "lr_augmentation": lr_augment, "manifest": str(manifest_path),
              "manifest_sha256": sha256(manifest_path), "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "model_config": model.config_dict(),
              "interpretation": "source-aware semantic probe; no native dance qualification"}
    write_json(out / "REPORT.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-lr-augment", action="store_true")
    args = parser.parse_args(argv)
    result = train(args.manifest, args.out, split=args.split, steps=args.steps,
                   batch_size=args.batch_size, frames=args.frames, seed=args.seed,
                   device=args.device, lr_augment=not args.no_lr_augment)
    print(json.dumps({k: result[k] for k in ("status", "arm", "steps", "loss_first", "loss_last")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
