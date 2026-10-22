#!/usr/bin/env python3
"""Old BodyToDMX scene baseline plus a masked Roskilde abstract-light branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig, masked_loss
from scripts.light_layers.train_body_to_dmx import load_pairs


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def moments(arrays):
    x = np.concatenate(arrays)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


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
    features = features[:n]
    mean, std = features.mean(0), features.std(0)
    std[std < 1e-5] = 1.0
    return ((features - mean) / std).astype(np.float32), light[:n].astype(np.float32)


class RoskildeBodyToDMX(BodyToDMX):
    """Old causal scene trunk plus a source-specific Roskilde head."""

    def __init__(self, cfg: BodyToDMXConfig, roskilde_audio_dim: int = 36, roskilde_scene_dim: int = 60):
        super().__init__(cfg)
        d = cfg.d_model
        self.roskilde_audio = nn.Sequential(nn.LayerNorm(roskilde_audio_dim), nn.Linear(roskilde_audio_dim, d), nn.SiLU())
        self.roskilde_scene = nn.Linear(d, roskilde_scene_dim)

    def forward_roskilde(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        b = torch.zeros((audio.shape[0], audio.shape[1], self.cfg.d_model), device=audio.device, dtype=audio.dtype)
        a = self.roskilde_audio(audio)
        h = self.fuse(torch.cat([b, a, torch.tanh(self.mul_a(b)) * torch.sigmoid(self.mul_b(a))], dim=-1))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return {"h": h, "scene60": self.roskilde_scene(h)}


def sample_roskilde(cache, indices, batch_size, window, device):
    xs, ys = [], []
    while len(xs) < batch_size:
        features, light = cache[random.choice(indices)]
        if len(features) < window:
            continue
        start = random.randrange(len(features) - window + 1)
        xs.append(features[start : start + window])
        ys.append(light[start : start + window])
    return torch.from_numpy(np.stack(xs)).float().to(device), torch.from_numpy(np.stack(ys)).float().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-root", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013"))
    parser.add_argument("--ids", default="1023,10dcc,1181f,11b01,1672d,210e3,21c3a,3b4ab")
    parser.add_argument("--rpmc-dir", type=Path, default=Path("/mnt/lightsem-data/cache/rpmc_all"))
    parser.add_argument("--roskilde", type=Path, default=Path("/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl"))
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/lightsem-data/beatsaver/manifest.json"))
    parser.add_argument("--cache", type=Path, default=Path("/mnt/lightsem-data/cache"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013/pilot_seed0_v7_multisource/REPORT.json"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--open-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not np.isfinite(args.open_loss_weight) or args.open_loss_weight <= 0:
        raise ValueError("--open-loss-weight must be finite and positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True)
    roskilde_cache_dir = args.out / "roskilde_cache"
    roskilde_cache_dir.mkdir()

    beat = load_pairs(args.body_root, args.ids.split(","), args.cache, args.manifest)
    if len(beat) < 4:
        raise RuntimeError(f"need at least four BeatSaver pairs, got {len(beat)}")
    beat.sort(key=lambda p: p["id"])
    n_val = max(2, int(np.ceil(len(beat) * 0.2)))
    beat_train, beat_val = beat[:-n_val], beat[-n_val:]

    rpmc = sorted(args.rpmc_dir.glob("*.npz"))
    if len(rpmc) < 10:
        raise RuntimeError(f"need at least ten RPMC compact files, got {len(rpmc)}")
    rpmc_perm = np.random.default_rng(args.seed).permutation(len(rpmc))
    rpmc_val_n = max(10, len(rpmc) // 5)
    rpmc_train = [rpmc[i] for i in rpmc_perm[:-rpmc_val_n]]
    rpmc_val = [rpmc[i] for i in rpmc_perm[-rpmc_val_n:]]

    with args.roskilde.open("rb") as handle:
        items = pickle.load(handle)
    roskilde_cache = []
    for index, item in enumerate(items):
        features, light = roskilde_pack(item)
        np.savez_compressed(roskilde_cache_dir / f"{index:04d}.npz", x=features, y=light)
        roskilde_cache.append((features, light))
    del items
    ros_perm = np.random.default_rng(args.seed).permutation(len(roskilde_cache))
    ros_val_n = max(10, len(roskilde_cache) // 5)
    ros_train = ros_perm[:-ros_val_n].tolist()
    ros_val = ros_perm[-ros_val_n:].tolist()

    body_mean, body_std = moments([p["body"] for p in beat_train])
    audio_mean, audio_std = moments([p["audio"] for p in beat_train])
    rpmc_mean, rpmc_std = moments([np.load(p, allow_pickle=False)["x"].astype(np.float32) for p in rpmc_train[:40]])
    ros_light_mean, ros_light_std = moments([roskilde_cache[i][1] for i in ros_train])
    cfg = BodyToDMXConfig(audio_dim=int(audio_mean.shape[0]), rpmc_audio_dim=int(rpmc_mean.shape[0]))
    model = RoskildeBodyToDMX(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    stats = [torch.tensor(v, device=args.device) for v in (body_mean, body_std, audio_mean, audio_std, rpmc_mean, rpmc_std)]
    body_mean_t, body_std_t, audio_mean_t, audio_std_t, rpmc_mean_t, rpmc_std_t = stats
    history = []

    for step in range(args.steps):
        source = step % 4
        if source in (0, 1):
            p = beat_train[step % len(beat_train)]
            n = len(p["body"])
            start = 0 if n <= args.window else random.randrange(n - args.window)
            sl = slice(start, min(start + args.window, n))
            body = torch.from_numpy(p["body"][sl]).to(args.device)
            audio = torch.from_numpy(p["audio"][sl]).to(args.device)
            scene = torch.from_numpy(p["scene"][sl]).to(args.device)
            event = torch.from_numpy(p["event"][sl]).to(args.device)
            angle = torch.from_numpy(p["angle"][sl]).to(args.device)
            pred = model(((body - body_mean_t) / body_std_t)[None], ((audio - audio_mean_t) / audio_std_t)[None])
            losses = masked_loss(pred, scene[None], event[None], angle=angle[None])
            loss = losses["total"]
            source_name = "beatsaver"
        elif source == 2:
            p = rpmc_train[step % len(rpmc_train)]
            with np.load(p, allow_pickle=False) as z:
                audio_np, scene_np = z["x"].astype(np.float32), z["y"].astype(np.float32)
            n = len(audio_np)
            start = 0 if n <= args.window else random.randrange(n - args.window)
            sl = slice(start, min(start + args.window, n))
            audio = torch.from_numpy(audio_np[sl]).to(args.device)
            scene = torch.from_numpy(scene_np[sl]).to(args.device)
            body = torch.zeros((len(audio), cfg.body_dim), device=args.device)
            pred = model(((body - body_mean_t) / body_std_t)[None], ((audio - rpmc_mean_t) / rpmc_std_t)[None], audio_kind="rpmc", body_present=False)
            losses = masked_loss(pred, scene[None])
            loss = losses["total"] * args.open_loss_weight
            source_name = "rpmc"
        else:
            audio, target = sample_roskilde(roskilde_cache, ros_train, 1, args.window, args.device)
            target = (target - torch.tensor(ros_light_mean, device=args.device)) / torch.tensor(ros_light_std, device=args.device)
            pred = model.forward_roskilde(audio)
            loss = torch.nn.functional.smooth_l1_loss(pred["scene60"], target)
            losses = {"total": loss, "roskilde": loss}
            loss = loss * args.open_loss_weight
            source_name = "roskilde"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            history.append({"step": step, "source": source_name, **{k: float(v.detach().cpu()) for k, v in losses.items()}})

    model.eval()

    def eval_beat(rows):
        values = []
        with torch.no_grad():
            for p in rows:
                body = torch.from_numpy(p["body"]).to(args.device)
                audio = torch.from_numpy(p["audio"]).to(args.device)
                target = torch.from_numpy(p["scene"]).to(args.device)
                pred = model(((body - body_mean_t) / body_std_t)[None], ((audio - audio_mean_t) / audio_std_t)[None])["scene"][0]
                values.append(float(torch.mean(torch.abs(pred - target)).cpu()))
        return float(np.mean(values))

    def eval_rpmc(rows):
        values = []
        with torch.no_grad():
            for p in rows:
                with np.load(p, allow_pickle=False) as z:
                    audio_np, target_np = z["x"].astype(np.float32), z["y"].astype(np.float32)
                audio = torch.from_numpy(audio_np).to(args.device)
                target = torch.from_numpy(target_np).to(args.device)
                body = torch.zeros((len(audio), cfg.body_dim), device=args.device)
                pred = model(torch.zeros_like(body)[None], ((audio - rpmc_mean_t) / rpmc_std_t)[None], audio_kind="rpmc", body_present=False)["scene"][0]
                values.append(float(torch.mean(torch.abs(pred - target)).cpu()))
        return float(np.mean(values))

    def eval_roskilde(indices):
        values = []
        baseline_values = []
        mean = torch.tensor(ros_light_mean, device=args.device)
        std = torch.tensor(ros_light_std, device=args.device)
        with torch.no_grad():
            for index in indices:
                features, target_np = roskilde_cache[index]
                target = torch.from_numpy(target_np).to(args.device)
                pred = model.forward_roskilde(torch.from_numpy(features).to(args.device).unsqueeze(0))["scene60"][0]
                target_norm = (target - mean) / std
                values.append(float(torch.mean(torch.abs(pred - target_norm)).cpu()))
                baseline_values.append(float(torch.mean(torch.abs(target_norm)).cpu()))
        return float(np.mean(values)), float(np.mean(baseline_values))

    beatsaver_scene = eval_beat(beat_val)
    rpmc_scene = eval_rpmc(rpmc_val)
    roskilde_scene, roskilde_baseline = eval_roskilde(ros_val)
    baseline_report = json.loads(args.baseline_report.read_text())
    baseline_beatsaver = float(baseline_report["beatsaver_val_scene_mae"])
    baseline_rpmc = float(baseline_report["rpmc_val_scene_mae"])
    percent = {
        "beatsaver_scene_mae_change_pct": 100.0 * (baseline_beatsaver - beatsaver_scene) / baseline_beatsaver,
        "rpmc_scene_mae_change_pct": 100.0 * (baseline_rpmc - rpmc_scene) / baseline_rpmc,
        "roskilde_vs_mean_baseline_change_pct": 100.0 * (roskilde_baseline - roskilde_scene) / roskilde_baseline,
    }
    predictions = {}
    for p in beat_val:
        body = torch.from_numpy(p["body"]).to(args.device)
        audio = torch.from_numpy(p["audio"]).to(args.device)
        with torch.no_grad():
            pred = model(((body - body_mean_t) / body_std_t)[None], ((audio - audio_mean_t) / audio_std_t)[None])
        predictions[p["id"]] = {
            "time": np.arange(len(p["scene"]), dtype=np.float32) / 10.0,
            "scene": pred["scene"][0].cpu().numpy(),
            "event_logit": pred["event_logit"][0, :, 0].cpu().numpy(),
            "angle_logit": pred["angle_logit"][0].cpu().numpy(),
            "target_scene": p["scene"],
            "target_event": p["event"],
            "body_path": p["body_path"],
            "target_path": p["target_path"],
        }
        np.savez_compressed(args.out / f"PREDICTIONS_{p['id']}.npz", **{k: v for k, v in predictions[p["id"]].items() if isinstance(v, np.ndarray)})
    checkpoint = args.out / "model.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "stats": {"body_mean": body_mean.tolist(), "body_std": body_std.tolist(), "audio_mean": audio_mean.tolist(), "audio_std": audio_std.tolist(), "rpmc_mean": rpmc_mean.tolist(), "rpmc_std": rpmc_std.tolist(), "roskilde_mean": ros_light_mean.tolist(), "roskilde_std": ros_light_std.tolist()}, "seed": args.seed, "roskilde_audio_dim": 36, "roskilde_scene_dim": 60}, checkpoint)
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "reason": "Roskilde has abstract lighting only; fixed-C native motion baseline is unchanged",
        "config": cfg.__dict__,
        "params": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "steps": args.steps,
        "open_loss_weight": args.open_loss_weight,
        "beatsaver_train_ids": [p["id"] for p in beat_train],
        "beatsaver_val_ids": [p["id"] for p in beat_val],
        "rpmc_train_files": len(rpmc_train),
        "rpmc_val_files": len(rpmc_val),
        "roskilde_train_files": len(ros_train),
        "roskilde_val_files": len(ros_val),
        "history": history,
        "baseline_report": str(args.baseline_report.resolve()),
        "baseline_report_sha256": sha(args.baseline_report),
        "baseline_metrics": {"beatsaver_scene_mae": baseline_beatsaver, "rpmc_scene_mae": baseline_rpmc},
        "treatment_metrics": {"beatsaver_scene_mae": beatsaver_scene, "rpmc_scene_mae": rpmc_scene, "roskilde_abstract_mae": roskilde_scene, "roskilde_mean_baseline_mae": roskilde_baseline},
        "percent_difference_vs_baseline": percent,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha(checkpoint),
        "fixed_c_native_motion_changed": False,
        "physical_angle_target_from_roskilde": False,
        "native_dmx": False,
    }
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
