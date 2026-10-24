#!/usr/bin/env python3
"""Jointly train the legacy audio-to-light decoder on canonical + open data.

RPMC-L2 contributes appearance-derived HSV/value targets. Roskilde contributes
its released abstract [T,60] lighting tensor through an auxiliary shared-latent
head. Neither source is treated as authored pan/tilt/DMX supervision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.foundation_audio_light.safety_projection import (
    bounded_dynamics_projection,
    smooth_bounded_dynamics_projection,
)
from scripts.experiments.evaluate_foundation_v3_own_songs import align_features, load_model
from scripts.experiments.run_foundation_v3_audio_only import audit_output, batch as canonical_batch
from scripts.experiments.run_foundation_v3_audio_only import load_row, loss_parts
from scripts.experiments.run_open_prior_bridge import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_BASE92,
    DEFAULT_CONTRACT,
    DEFAULT_CURE_AUDIO,
    DEFAULT_MUQ,
    DEFAULT_SHONGLE_AUDIO,
    evaluate_rows,
    appearance_metric,
    infer_model,
    sha,
    zscore,
)

DEFAULT_RPMC = Path("/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5")
DEFAULT_ROSKILDE = Path("/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl")


def interp_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if len(x) == n:
        return x
    old = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(new, old, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:]).astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    out = np.empty(h.shape + (3,), np.float32)
    choices = (
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    )
    for k, values in enumerate(choices):
        mask = i == k
        for c in range(3):
            out[..., c][mask] = values[c][mask]
    return out


def rpmc_pack(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return 36-D input and eight-group appearance targets."""
    music = group["music"]
    mfcc = np.asarray(music["mfcc"][:13], np.float32)
    chroma = np.asarray(music["chroma_stft"][:12], np.float32)
    centroid = np.asarray(music["spectral_centroids"][:], np.float32)
    mel = np.asarray(music["mel_spectrogram"][:], np.float32)
    rms = np.log1p(np.maximum(mel, 0.0)).mean(0, keepdims=True).astype(np.float32)
    delta = np.diff(mfcc[:9], axis=1, prepend=mfcc[:9, :1]).astype(np.float32)
    features = np.concatenate([mfcc, chroma, centroid, rms, delta], 0).T

    curves = []
    for key in sorted(group["light"].keys()):
        raw = np.asarray(group["light"][key][:], np.float32)
        hue_hist, sat_hist, value_hist = raw[:, 0], raw[:, 1], raw[:, 2]
        mass = np.maximum(value_hist.sum(1), 1.0)
        bins = np.arange(value_hist.shape[1], dtype=np.float32)
        value = (value_hist * bins[None]).sum(1) / mass / 255.0
        saturation = (sat_hist * bins[None]).sum(1) / mass / 255.0
        hb = np.arange(min(180, hue_hist.shape[1]), dtype=np.float32) * (2.0 * np.pi / 180.0)
        hm = np.maximum(hue_hist[:, : len(hb)].sum(1), 1.0)
        hx = (hue_hist[:, : len(hb)] * np.cos(hb)[None]).sum(1) / hm
        hy = (hue_hist[:, : len(hb)] * np.sin(hb)[None]).sum(1) / hm
        hue = np.unwrap(np.arctan2(hy, hx)).astype(np.float32) / (2.0 * np.pi)
        curves.append(np.stack([hue, saturation, value], 1))
    if not curves:
        raise ValueError("RPMC sample has no light groups")
    max_len = max(len(x) for x in curves)
    curves = np.stack([interp_1d(x, max_len) for x in curves], 1)
    n = max(32, min(len(features) // 7, len(curves) // 3))
    features = interp_1d(features, n)
    curves = interp_1d(curves, n)
    features = zscore(np.nan_to_num(features))
    group_idx = np.linspace(0, curves.shape[1] - 1, 8).round().astype(int)
    hsv = curves[:, group_idx]
    rgb = hsv_to_rgb(hsv[..., 0], np.clip(hsv[..., 1], 0, 1), np.clip(hsv[..., 2], 0, 1))
    brightness = np.clip(hsv[..., 2, None], 0, 1)
    activity = np.mean(np.abs(np.diff(hsv, axis=0, prepend=hsv[:1])), axis=(1, 2))
    event = np.clip(activity / max(float(np.quantile(activity, 0.95)), 1e-4), 0, 1)[:, None]
    target = {
        "rgb": rgb.astype(np.float32),
        "brightness": brightness.astype(np.float32),
        "event": event.astype(np.float32),
    }
    return features, np.concatenate([target["rgb"].reshape(n, -1), target["brightness"].reshape(n, -1), target["event"]], 1), {
        "source": "RPMC-L2 HSV/value histogram appearance proxy",
        "groups": int(curves.shape[1]),
    }


def unpack_rpmc_target(target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rgb": target[:, : 8 * 3].reshape(len(target), 8, 3),
        "brightness": target[:, 8 * 3 : 8 * 4].reshape(len(target), 8, 1),
        "event": target[:, 8 * 4 : 8 * 4 + 1],
    }


def roskilde_pack(item: dict) -> tuple[np.ndarray, np.ndarray]:
    audio = item["audio_features"]
    mfcc = np.asarray(audio["mfcc"], np.float32)[:, :13]
    chroma = np.asarray(audio["chroma_stft"], np.float32)[:, :12]
    onset = np.asarray(audio["onset_env"], np.float32).reshape(-1, 1)
    rms = np.asarray(audio["rms"], np.float32).reshape(-1, 1)
    delta = np.asarray(audio["mfcc_delta"], np.float32)[:, :9]
    features = np.concatenate([mfcc, chroma, onset, rms, delta], 1)
    light = np.asarray(item["lighting_features"], np.float32)
    n = min(len(features), len(light))
    return zscore(np.nan_to_num(features[:n])), light[:n].astype(np.float32)


class RPMCProvider:
    def __init__(self, path: Path):
        self.path = path
        self.handle = h5py.File(path, "r")
        self.keys = sorted(self.handle.keys())

    @lru_cache(maxsize=8)
    def get(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        features, packed, _ = rpmc_pack(self.handle[key])
        return features, packed

    def close(self) -> None:
        self.handle.close()


class JointOpenLegacyModel(nn.Module):
    def __init__(self, base: nn.Module, open_dim: int = 36, aux_dim: int = 60):
        super().__init__()
        self.base = base
        width = base.foundation.config.width
        self.open_encoder = nn.Sequential(
            nn.Conv1d(open_dim, width, 5, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, width),
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
        )
        self.open_aux = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, aux_dim))

    def _decode_project(self, hidden: torch.Tensor, fixtures: torch.Tensor, state=None) -> dict[str, torch.Tensor]:
        if isinstance(state, dict):
            motion_state = state.get("motion")
            foundation_state = state.get("foundation")
        else:
            motion_state = None
            foundation_state = state
        latent_seed = self.base.foundation.context.latent_seed[None].expand(hidden.shape[0], -1, -1)
        out = self.base.foundation._decode(hidden, fixtures, latent_seed)
        raw = out["aim"] * float(self.base.config.residual_gain)
        if self.base.config.dynamics_mode == "hard_clipped":
            projection = bounded_dynamics_projection
        else:
            projection = smooth_bounded_dynamics_projection
        kwargs = (
            {
                "drive_scale": self.base.config.dynamics_drive_scale,
                "position_gain": self.base.config.dynamics_position_gain,
                "velocity_gain": self.base.config.dynamics_velocity_gain,
                "acceleration_gain": self.base.config.dynamics_acceleration_gain,
                "radial_bounds": self.base.config.dynamics_radial_bounds,
            }
            if self.base.config.dynamics_mode == "smooth_jerk"
            else {}
        )
        (delta, velocity, acceleration, jerk), new_motion = projection(
            raw, motion_state, self.base.config.safety, **kwargs
        )
        anchor = self.base._anchor(fixtures)
        out.update(
            {
                "raw_aim": out["aim"],
                "anchor": anchor,
                "aim": anchor[:, None] + delta,
                "velocity": velocity,
                "acceleration": acceleration,
                "jerk": jerk,
                "audio_latent": hidden,
                "state": {"foundation": foundation_state, "motion": new_motion},
            }
        )
        return out

    def forward_open(
        self,
        features: torch.Tensor,
        fixtures: torch.Tensor,
        *,
        valid: torch.Tensor,
        duration_seconds: float,
        auxiliary: bool,
    ) -> dict[str, torch.Tensor]:
        tokens = self.open_encoder(features.transpose(1, 2)).transpose(1, 2)
        hidden, new_state = self.base.foundation.context(tokens, valid, None, return_state=True)
        out = self._decode_project(hidden, fixtures, {"foundation": new_state, "motion": None})
        if auxiliary:
            out["open_aux"] = self.open_aux(hidden)
        return out


def appearance_target_tensor(target: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value[None]).float().to(device) for key, value in target.items()} | {
        "valid": torch.ones((1, len(target["event"])), dtype=torch.bool, device=device)
    }


def sample_open_batch(provider: RPMCProvider, keys: list[str], batch_size: int, window: int, device: torch.device):
    xs, ys = [], []
    while len(xs) < batch_size:
        key = random.choice(keys)
        features, packed = provider.get(key)
        if len(features) < window:
            continue
        start = random.randrange(len(features) - window + 1)
        xs.append(features[start : start + window])
        ys.append(packed[start : start + window])
    x = torch.from_numpy(np.stack(xs)).float().to(device)
    packed = np.stack(ys)
    target = unpack_rpmc_target(packed.reshape(-1, packed.shape[-1]))
    target = {key: value.reshape(batch_size, window, *value.shape[1:]) for key, value in target.items()}
    return x, {key: torch.from_numpy(value).float().to(device) for key, value in target.items()}


def sample_roskilde(cache: list[tuple[np.ndarray, np.ndarray]], indices: list[int], batch_size: int, window: int, device: torch.device):
    xs, ys = [], []
    while len(xs) < batch_size:
        features, light = cache[random.choice(indices)]
        if len(features) < window:
            continue
        start = random.randrange(len(features) - window + 1)
        xs.append(features[start : start + window])
        ys.append(light[start : start + window])
    return (
        torch.from_numpy(np.stack(xs)).float().to(device),
        torch.from_numpy(np.stack(ys)).float().to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--rpmc", type=Path, default=DEFAULT_RPMC)
    parser.add_argument("--roskilde", type=Path, default=DEFAULT_ROSKILDE)
    parser.add_argument("--shpongle-muq", type=Path, default=DEFAULT_MUQ / "shpongle.npz")
    parser.add_argument("--shpongle-base92", type=Path, default=DEFAULT_BASE92 / "shpongle.npz")
    parser.add_argument("--cure-muq", type=Path, default=DEFAULT_MUQ / "cure_for_me.npz")
    parser.add_argument("--cure-base92", type=Path, default=DEFAULT_BASE92 / "cure_for_me.npz")
    parser.add_argument("--shpongle-audio", type=Path, default=DEFAULT_SHONGLE_AUDIO)
    parser.add_argument("--cure-audio", type=Path, default=DEFAULT_CURE_AUDIO)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    for path in [args.contract, args.base_checkpoint, args.rpmc, args.roskilde, args.shpongle_audio, args.cure_audio]:
        if not path.exists():
            raise FileNotFoundError(path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=False)
    cache_dir = args.out / "roskilde_cache"
    cache_dir.mkdir()

    contract = json.loads(args.contract.read_text())
    rows = []
    for source in contract["rows"]:
        row = load_row(source)
        row["wav_path"] = source["wav_path"]
        rows.append(row)
    groups = np.asarray(sorted({row["wav_sha256"] for row in rows}))
    groups = groups[np.random.default_rng(args.seed).permutation(len(groups))]
    split = max(1, int(0.8 * len(groups)))
    train_groups = set(groups[:split])
    train_rows = [row for row in rows if row["wav_sha256"] in train_groups]
    val_rows = [row for row in rows if row["wav_sha256"] not in train_groups]
    train_windows = [(i, start) for i, row in enumerate(train_rows) for start in range(0, len(row["audio"]) - args.window + 1, args.window)]

    roskilde_cache: list[tuple[np.ndarray, np.ndarray]] = []
    cached_files = sorted(cache_dir.glob("*.npz"))
    if len(cached_files) == 224:
        for path in cached_files:
            with np.load(path, allow_pickle=False) as bundle:
                roskilde_cache.append((np.asarray(bundle["features"], np.float32), np.asarray(bundle["light"], np.float32)))
    else:
        with args.roskilde.open("rb") as handle:
            roskilde_items = pickle.load(handle)
        for index, item in enumerate(roskilde_items):
            features, light = roskilde_pack(item)
            np.savez_compressed(cache_dir / f"{index:04d}.npz", features=features, light=light)
            roskilde_cache.append((features, light))
        del roskilde_items
    ros_perm = np.random.default_rng(args.seed).permutation(len(roskilde_cache))
    ros_split = max(1, int(0.8 * len(roskilde_cache)))
    ros_train = ros_perm[:ros_split].tolist()
    ros_val = ros_perm[ros_split:].tolist()
    provider = RPMCProvider(args.rpmc)
    rpmc_perm = np.random.default_rng(args.seed).permutation(len(provider.keys))
    rpmc_split = max(1, int(0.8 * len(provider.keys)))
    rpmc_train = [provider.keys[i] for i in rpmc_perm[:rpmc_split]]
    rpmc_val = [provider.keys[i] for i in rpmc_perm[rpmc_split:]]

    baseline = load_model(args.base_checkpoint, device)
    baseline.eval()
    base = load_model(args.base_checkpoint, device)
    model = JointOpenLegacyModel(base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)
    history: list[dict[str, float | int | str]] = []
    model.train()
    for step in range(args.steps):
        source = (step % 4)
        if source in (0, 1):
            picked = [train_windows[(step * args.batch_size + j) % len(train_windows)] for j in range(args.batch_size)]
            target = canonical_batch(train_rows, picked, args.window, device)
            timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(args.batch_size, -1) / 10.0
            output = model.base(target["audio"], fixtures[None].expand(args.batch_size, -1, -1), timestamps=timestamps, valid=target["valid"], duration_seconds=args.window / 10.0)
            parts = loss_parts(output, target)
            loss = parts["total"]
            record = {"source": "canonical", **{key: float(value.detach().cpu()) for key, value in parts.items()}}
        elif source == 2:
            features, target = sample_open_batch(provider, rpmc_train, args.batch_size, args.window, device)
            valid = torch.ones((args.batch_size, args.window), dtype=torch.bool, device=device)
            output = model.forward_open(features, fixtures[None].expand(args.batch_size, -1, -1), valid=valid, duration_seconds=args.window / 10.0, auxiliary=False)
            open_target = {**target, "valid": valid}
            parts = loss_parts(output, open_target)
            loss = parts["total"]
            record = {"source": "rpmc", **{key: float(value.detach().cpu()) for key, value in parts.items()}}
        else:
            features, light = sample_roskilde(roskilde_cache, ros_train, args.batch_size, args.window, device)
            valid = torch.ones((args.batch_size, args.window), dtype=torch.bool, device=device)
            output = model.forward_open(features, fixtures[None].expand(args.batch_size, -1, -1), valid=valid, duration_seconds=args.window / 10.0, auxiliary=True)
            target_light = (light - light.mean(0, keepdim=True)) / light.std(0, keepdim=True).clamp_min(1e-4)
            pred_light = (output["open_aux"] - output["open_aux"].mean((0, 1), keepdim=True)) / output["open_aux"].std((0, 1), keepdim=True).clamp_min(1e-4)
            aux_loss = F.mse_loss(pred_light, target_light)
            loss = aux_loss
            record = {"source": "roskilde", "total": float(loss.detach().cpu()), "aux": float(aux_loss.detach().cpu())}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 9) == 0:
            record["step"] = step + 1
            history.append(record)

    model.eval()
    baseline_metric, _ = evaluate_rows(baseline, val_rows, device, False, args.window)
    treatment_metric, _ = evaluate_rows(model.base, val_rows, device, False, args.window)

    def eval_rpmc(keys: list[str]) -> float:
        values = []
        with torch.no_grad():
            for key in keys[: min(80, len(keys))]:
                features, packed = provider.get(key)
                n = min(len(features), args.window * 4)
                features_t = torch.from_numpy(features[:n][None]).float().to(device)
                target_np = unpack_rpmc_target(packed[:n])
                target = appearance_target_tensor(target_np, device)
                valid = target["valid"]
                output = model.forward_open(features_t, fixtures[None], valid=valid, duration_seconds=n / 10.0, auxiliary=False)
                values.append(float(appearance_metric(output, target)["composite"]))
        return float(np.mean(values)) if values else float("nan")

    rpmc_treatment = eval_rpmc(rpmc_val)
    rpmc_mean_baseline = eval_rpmc(rpmc_val)  # replaced below with constant-target baseline
    rpmc_constant_values = []
    for key in rpmc_val[: min(80, len(rpmc_val))]:
        _, packed = provider.get(key)
        target = unpack_rpmc_target(packed)
        means = {k: v.mean(0, keepdims=True) for k, v in target.items()}
        rgb = np.broadcast_to(means["rgb"], target["rgb"].shape)
        bright = np.broadcast_to(means["brightness"], target["brightness"].shape)
        event = np.broadcast_to(means["event"], target["event"].shape)
        rgb_l1 = float(np.mean(np.abs(target["rgb"] - rgb)))
        brightness_l1 = float(np.mean(np.abs(target["brightness"] - bright)))
        event_bce = float(np.mean(-(target["event"] * np.log(np.clip(event, 1e-6, 1 - 1e-6)) + (1 - target["event"]) * np.log(np.clip(1 - event, 1e-6, 1 - 1e-6)))))
        rpmc_constant_values.append((rgb_l1 + brightness_l1 + event_bce) / 3.0)
    rpmc_mean_baseline = float(np.mean(rpmc_constant_values)) if rpmc_constant_values else float("nan")

    ros_errors = []
    ros_mean_errors = []
    with torch.no_grad():
        for i in ros_val:
            features, light = roskilde_cache[i]
            n = min(len(features), args.window * 4)
            x = torch.from_numpy(features[:n][None]).float().to(device)
            valid = torch.ones((1, n), dtype=torch.bool, device=device)
            output = model.forward_open(x, fixtures[None], valid=valid, duration_seconds=n / 10.0, auxiliary=True)
            pred = output["open_aux"][0, :n].cpu().numpy()
            truth = light[:n]
            pred_norm = (pred - pred.mean(0, keepdims=True)) / np.maximum(pred.std(0, keepdims=True), 1e-4)
            truth_norm = (truth - truth.mean(0, keepdims=True)) / np.maximum(truth.std(0, keepdims=True), 1e-4)
            ros_errors.append(float(np.mean((pred_norm - truth_norm) ** 2)))
            ros_mean_errors.append(float(np.mean(truth_norm**2)))
    ros_treatment = float(np.mean(ros_errors)) if ros_errors else float("nan")
    ros_mean_baseline = float(np.mean(ros_mean_errors)) if ros_mean_errors else float("nan")

    songs = {"shpongle": (args.shpongle_audio, args.shpongle_muq, args.shpongle_base92), "cure_for_me": (args.cure_audio, args.cure_muq, args.cure_base92)}
    predictions = {}
    song_reports = {}
    for name, (audio_path, muq_path, base92_path) in songs.items():
        audio, times = align_features(muq_path, base92_path)
        out = {key: value[0] for key, value in infer_model(model.base, audio, device).items()}
        predictions[name] = out
        song_reports[name] = audit_output({key: torch.from_numpy(value[None]) for key, value in out.items()}, 10.0) | {"frames": int(len(audio)), "duration_s": float(times[-1])}
    n = min(len(predictions["shpongle"]["rgb"]), len(predictions["cure_for_me"]["rgb"]))
    swap = float(np.mean(np.abs(predictions["shpongle"]["rgb"][:n] - predictions["cure_for_me"]["rgb"][:n])))

    args.out.mkdir(exist_ok=True)
    checkpoint = args.out / "model.pt"
    torch.save({"state_dict": model.state_dict(), "seed": args.seed, "base_checkpoint": str(args.base_checkpoint.resolve()), "base_checkpoint_sha256": sha(args.base_checkpoint), "open_input_dim": 36, "roskilde_aux_dim": 60}, checkpoint)
    for name, out in predictions.items():
        np.savez_compressed(args.out / f"PREDICTIONS_{name}.npz", time=np.arange(len(out["aim"])) / 10.0, **out)
    raw_pass = all(song_reports[name]["raw_safety_pass"] for name in song_reports)
    collapse_pass = all(song_reports[name]["rgb_std"] > 0.01 and song_reports[name]["brightness_std"] > 0.01 and song_reports[name]["duplicate_stream_count"] == 0 for name in song_reports)
    appearance_gain = baseline_metric["composite"] - treatment_metric["composite"]
    ros_gain = ros_mean_baseline - ros_treatment
    status = "diagnostic_candidate" if appearance_gain >= 0.005 and ros_gain >= 0.01 and raw_pass and collapse_pass and swap > 0.001 else "rejected_appearance_gain" if appearance_gain < 0.005 else "rejected_open_auxiliary" if ros_gain < 0.01 else "rejected_raw_safety" if not raw_pass else "output_collapsed" if not collapse_pass else "unconditioned_or_shared_prior"
    receipt = {"schema": "lightsem.foundation_joint_open_legacy.v1", "protocol_id": "foundation-joint-open-legacy-20261022-v1", "status": status, "seed": args.seed, "device": str(device), "steps": args.steps, "window": args.window, "batch_size": args.batch_size, "contract": str(args.contract.resolve()), "contract_sha256": sha(args.contract), "base_checkpoint": str(args.base_checkpoint.resolve()), "base_checkpoint_sha256": sha(args.base_checkpoint), "rpmc": str(args.rpmc.resolve()), "rpmc_sha256": sha(args.rpmc), "roskilde": str(args.roskilde.resolve()), "roskilde_sha256": sha(args.roskilde), "train_groups": len(train_groups), "validation_groups": len(groups) - len(train_groups), "train_rows": len(train_rows), "validation_rows": len(val_rows), "rpmc_train_samples": len(rpmc_train), "rpmc_validation_samples": len(rpmc_val), "roskilde_train_samples": len(ros_train), "roskilde_validation_samples": len(ros_val), "history": history, "baseline_canonical": baseline_metric, "treatment_canonical": treatment_metric, "canonical_gain": appearance_gain, "rpmc_treatment": rpmc_treatment, "rpmc_constant_baseline": rpmc_mean_baseline, "rpmc_gain": rpmc_mean_baseline - rpmc_treatment, "roskilde_treatment": ros_treatment, "roskilde_mean_baseline": ros_mean_baseline, "roskilde_gain": ros_gain, "own_song_rgb_swap": swap, "songs": song_reports, "raw_safety_pass": raw_pass, "output_collapse_pass": collapse_pass, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "native_dmx": False, "motion_target_claim": False, "promotion": "appearance and abstract-light diagnostic only; no authored physical-light claim"}
    (args.out / "TRAINING.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    provider.close()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
