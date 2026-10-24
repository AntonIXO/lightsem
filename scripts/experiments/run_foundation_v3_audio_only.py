#!/usr/bin/env python3
"""Train and audit the audio-only foundation-v3 repair on canonical 300-row data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightsem.dance.pipeline import accelerator_lock
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.foundation_audio_light.direct_model import AudioOnlyDirectLightModel, DirectLightConfig
from lightsem.experiments.foundation_audio_light.layer_selection import effective_rank
from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig
from lightsem.experiments.foundation_audio_light.peft import trainable_parameter_report


CONTRACT = Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json")
MUQ_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3")
CACHE_ROOT = Path("/mnt/lightsem-data/cache")


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one(root: Path, suffix: str) -> Path:
    paths = sorted(root.glob(f"*={suffix}.npz"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one cache for {suffix} in {root}, found {len(paths)}")
    return paths[0]


def nearest(values: np.ndarray, times: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, target, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return values[np.where(np.abs(times[left] - target) <= np.abs(times[right] - target), left, right)]


def load_row(row: dict) -> dict:
    sid = row["id"]
    audio_path = one(CACHE_ROOT / "beatsaver_audio", sid)
    visual_path = one(CACHE_ROOT / "beatsaver_visual", sid)
    light_path = one(CACHE_ROOT / "beatsaver_light", sid)
    muq_path = MUQ_ROOT / f"{sid}.npz"
    scalar = (
        "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid", "flux_hf",
        "energy_lf", "energy_lmid", "energy_mid", "energy_hmid", "energy_hf", "onset", "onset_ema",
        "flux_ema", "flux_lf_ema", "flux_hf_ema", "centroid", "flatness", "rolloff", "percussive_ratio",
        "pulse_clarity", "beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s",
    )
    with np.load(audio_path, allow_pickle=False) as z:
        mel = np.asarray(z["mel"], np.float32).T
        base = np.concatenate([mel, np.stack([z["ch__" + name] for name in scalar], 1).astype(np.float32)], 1)
        base_time = np.arange(len(base), dtype=np.float32) / 40.0
    with np.load(muq_path, allow_pickle=False) as z:
        muq = np.asarray(z["muq"], np.float32)
        muq_time = np.asarray(z["muq_time"], np.float32)
    with np.load(visual_path, allow_pickle=False) as z:
        events = np.asarray(z["ev_any"], np.float32)
        event_time = np.arange(len(events), dtype=np.float32) / 10.0
    with np.load(light_path, allow_pickle=False) as z:
        rgb = np.asarray(z["rgb"], np.float32)
        intensity = np.asarray(z["inten"], np.float32)
    n = min(len(events), len(rgb[0]) // 4, len(muq), int(np.floor(float(muq_time[-1]) * 10)) + 1)
    clock = np.arange(n, dtype=np.float32) / 10.0
    audio = np.concatenate([nearest(muq, muq_time, clock), nearest(base, base_time, clock)], 1)
    rgb = rgb[:, : n * 4].reshape(rgb.shape[0], n, 4, 3).mean(2).transpose(1, 0, 2)
    intensity = intensity[:, : n * 4].reshape(intensity.shape[0], n, 4).mean(2).transpose(1, 0)[..., None]
    groups = np.arange(8) % max(1, rgb.shape[1])
    return {
        "id": sid,
        "wav_sha256": row["wav_sha256"],
        "audio": audio,
        "rgb": np.clip(rgb[:, groups], 0, 1),
        "brightness": np.clip(intensity[:, groups], 0, 1),
        "event": nearest(events, event_time, clock)[:, None],
        "paths": {"audio_cache": str(audio_path), "muq": str(muq_path), "visual": str(visual_path), "light": str(light_path)},
    }


def batch(rows, indices, window, device):
    return {
        "audio": torch.from_numpy(np.stack([rows[index]["audio"][start : start + window] for index, start in indices])).float().to(device),
        "rgb": torch.from_numpy(np.stack([rows[index]["rgb"][start : start + window] for index, start in indices])).float().to(device),
        "brightness": torch.from_numpy(np.stack([rows[index]["brightness"][start : start + window] for index, start in indices])).float().to(device),
        "event": torch.from_numpy(np.stack([rows[index]["event"][start : start + window] for index, start in indices])).float().to(device),
        "valid": torch.ones((len(indices), window), dtype=torch.bool, device=device),
    }


def loss_parts(output, target):
    valid = target["valid"]
    mask = valid[..., None, None]
    mean = lambda value: torch.where(mask.expand_as(value), value, torch.zeros_like(value)).sum() / mask.expand_as(value).sum().clamp_min(1)
    rgb = mean((output["rgb"] - target["rgb"]).abs())
    rgb_bce = mean(F.binary_cross_entropy_with_logits(output["rgb_logits"], target["rgb"], reduction="none"))
    brightness = mean((output["brightness"] - target["brightness"]).abs())
    event = torch.where(valid[..., None], F.binary_cross_entropy_with_logits(output["event"], target["event"], reduction="none"), torch.zeros_like(output["event"])).sum() / valid.sum().clamp_min(1)
    if output["rgb"].shape[2] > 1:
        pred_pair = torch.cdist(output["rgb"], output["rgb"])
        target_pair = torch.cdist(target["rgb"], target["rgb"])
        upper = torch.triu(torch.ones((pred_pair.shape[-1], pred_pair.shape[-1]), device=pred_pair.device, dtype=torch.bool), diagonal=1)
        spread_error = (pred_pair[..., upper] - target_pair[..., upper]).square()
        spread_mask = valid[..., None].expand_as(spread_error)
        spread = torch.where(spread_mask, spread_error, torch.zeros_like(spread_error)).sum() / spread_mask.sum().clamp_min(1)
    else:
        spread = rgb.new_zeros(())
    latent = output["audio_latent"]
    latent_flat = latent.reshape(-1, latent.shape[-1])
    latent_std = latent_flat.std(0)
    latent_floor = torch.relu(0.02 - latent_std).mean()
    latent_delta = torch.relu(0.002 - latent_flat[1:].diff(dim=0).abs().mean(0)).mean() if len(latent_flat) > 1 else latent_floor.new_zeros(())
    total = rgb + rgb_bce + 2.5 * brightness + 1.5 * event + 1.0 * spread + 0.5 * latent_floor + 0.1 * latent_delta
    return {"total": total, "rgb": rgb, "rgb_bce": rgb_bce, "brightness": brightness, "event": event, "rgb_spread": spread, "latent_floor": latent_floor, "latent_delta": latent_delta}


def audit_output(output: dict[str, torch.Tensor], rate_hz: float) -> dict:
    aim = output["aim"].detach().cpu().numpy().astype(np.float64)
    rgb = output["rgb"].detach().cpu().numpy().astype(np.float64)
    brightness = output["brightness"].detach().cpu().numpy().astype(np.float64)
    event = torch.sigmoid(output["event"]).detach().cpu().numpy().astype(np.float64)
    latent = output["audio_latent"].detach().cpu().numpy().astype(np.float64)
    parameterized_velocity = output["velocity"].detach().cpu().numpy().astype(np.float64)
    parameterized_acceleration = output["acceleration"].detach().cpu().numpy().astype(np.float64)
    parameterized_jerk = output["jerk"].detach().cpu().numpy().astype(np.float64)
    velocity = np.diff(aim, axis=1, prepend=aim[:, :1]) * rate_hz
    acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1]) * rate_hz
    jerk = np.diff(acceleration, axis=1, prepend=acceleration[:, :1]) * rate_hz
    pair = np.linalg.norm(aim[:, :, :, None, :] - aim[:, :, None, :, :], axis=-1)
    upper = np.triu_indices(aim.shape[2], 1)
    streams = aim.reshape(-1, aim.shape[2], aim.shape[3])
    duplicate = sum(frame.shape[0] - len({np.ascontiguousarray(row).tobytes() for row in frame}) for frame in streams)
    event_mean = float(event.mean())
    raw = {
        "finite": bool(np.isfinite(aim).all() and np.isfinite(rgb).all() and np.isfinite(brightness).all() and np.isfinite(event).all()),
        "speed_p99": float(np.percentile(np.linalg.norm(velocity, axis=-1), 99)),
        "acceleration_p99": float(np.percentile(np.linalg.norm(acceleration, axis=-1), 99)),
        "jerk_p99": float(np.percentile(np.linalg.norm(jerk, axis=-1), 99)),
        "parameterized_speed_component_p99": float(np.percentile(np.abs(parameterized_velocity), 99)),
        "parameterized_acceleration_component_p99": float(np.percentile(np.abs(parameterized_acceleration), 99)),
        "parameterized_jerk_component_p99": float(np.percentile(np.abs(parameterized_jerk), 99)),
        "rgb_std": float(rgb.std()),
        "rgb_below001_fraction": float(np.mean(rgb < 0.01)),
        "brightness_std": float(brightness.std()),
        "event_mean": event_mean,
        "event_entropy": float(-(event_mean * np.log(max(event_mean, 1e-8)) + (1 - event_mean) * np.log(max(1 - event_mean, 1e-8)))),
        "aim_spread_p50": float(np.median(pair[:, :, upper[0], upper[1]])),
        "aim_spread_min": float(np.min(pair[:, :, upper[0], upper[1]])),
        "duplicate_stream_count": int(duplicate),
        "latent_effective_rank": float(effective_rank(latent.reshape(-1, latent.shape[-1]))),
        "latent_constant_channel_fraction": float(np.mean(latent.reshape(-1, latent.shape[-1]).std(0) <= 1e-8)),
    }
    raw["raw_safety_pass"] = bool(raw["finite"] and raw["parameterized_speed_component_p99"] <= 0.18 + 1e-6 and raw["parameterized_acceleration_component_p99"] <= 0.5 + 1e-6 and raw["parameterized_jerk_component_p99"] <= 1.5 + 1e-6)
    raw["output_collapse_pass"] = bool(raw["rgb_std"] >= 0.01 and raw["brightness_std"] >= 0.01 and raw["duplicate_stream_count"] == 0 and raw["latent_effective_rank"] >= 2)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = json.loads(CONTRACT.read_text())
    rows = [load_row(row) for row in contract["rows"]]
    groups = {}
    for row in rows:
        groups.setdefault(row["wav_sha256"], []).append(row)
    group_keys = np.array(sorted(groups))
    group_keys = group_keys[np.random.default_rng(args.seed).permutation(len(group_keys))]
    split = max(1, int(0.8 * len(group_keys)))
    train_groups = set(group_keys[:split])
    train_rows = [row for row in rows if row["wav_sha256"] in train_groups]
    val_rows = [row for row in rows if row["wav_sha256"] not in train_groups]
    train_windows = [(index, start) for index, row in enumerate(train_rows) for start in range(0, max(1, len(row["audio"]) - args.window + 1), args.window) if start + args.window <= len(row["audio"])]
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)
    model_config = AudioToLightConfig(audio_dim=1116, width=96, heads=4, local_window=32, latent_slots=8,
                                      adapter_kind="hierarchical", adapter_bottleneck=32, adapter_heads=4,
                                      adapter_local_window=32)
    model = AudioOnlyDirectLightModel(DirectLightConfig(foundation=model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    started = time.time()
    with accelerator_lock():
        model.train()
        for step in range(args.steps):
            batch_indices = [train_windows[(step * args.batch_size + j) % len(train_windows)] for j in range(args.batch_size)]
            target = batch(train_rows, batch_indices, args.window, device)
            timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(args.batch_size, -1) / 10.0
            output = model(target["audio"], fixtures[None].expand(args.batch_size, -1, -1), timestamps=timestamps,
                           valid=target["valid"], duration_seconds=args.window / 10.0)
            parts = loss_parts(output, target)
            optimizer.zero_grad(set_to_none=True); parts["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 0 or (step + 1) % max(1, args.steps // 4) == 0:
                history.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in parts.items()}})
        model.eval()
        val_audits = []
        predictions = {}
        with torch.no_grad():
            for row in val_rows[:4]:
                n = min(len(row["audio"]), 512)
                audio = torch.from_numpy(row["audio"][:n][None]).float().to(device)
                valid = torch.ones((1, n), dtype=torch.bool, device=device)
                timestamps = torch.arange(n, device=device, dtype=torch.float32)[None] / 10.0
                output = model(audio, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                val_audits.append({"id": row["id"], **audit_output(output, 10.0)})
                predictions[row["id"]] = {key: value.detach().cpu().numpy() for key, value in output.items() if key in {"aim", "rgb", "brightness", "event", "audio_latent", "anchor"}}
            if len(val_rows) >= 2:
                n = min(len(val_rows[0]["audio"]), len(val_rows[1]["audio"]), 512)
                a0 = torch.from_numpy(val_rows[0]["audio"][:n][None]).float().to(device)
                a1 = torch.from_numpy(val_rows[1]["audio"][:n][None]).float().to(device)
                valid = torch.ones((1, n), dtype=torch.bool, device=device)
                timestamps = torch.arange(n, device=device, dtype=torch.float32)[None] / 10.0
                own = model(a0, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                repeat = model(a0, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                swapped = model(a1, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=n / 10.0)
                deterministic = max(float((own[key] - repeat[key]).abs().max().cpu()) for key in ("aim", "rgb", "brightness", "event"))
                audio_swap = {key: float((own[key] - swapped[key]).abs().mean().cpu()) for key in ("aim", "rgb", "brightness", "event")}
            else:
                deterministic, audio_swap = None, {}
    args.out.mkdir(parents=True, exist_ok=False)
    for sid, values in predictions.items():
        np.savez_compressed(args.out / f"PREDICTIONS_{sid}.npz", time=np.arange(len(values["aim"][0])) / 10.0, **values)
    checkpoint = args.out / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(), "seed": args.seed,
                "input_contract": "canonical MuQ1024 layer9 + Base92 92 = 1116; audio-only inference"}, checkpoint)
    all_raw = all(item["raw_safety_pass"] for item in val_audits)
    all_collapse = all(item["output_collapse_pass"] for item in val_audits)
    status = "diagnostic_candidate" if all_raw and all_collapse else "rejected_raw_safety" if not all_raw else "output_collapsed"
    receipt = {
        "schema": "lightsem.foundation_v3_audio_only.v1",
        "protocol_id": "foundation-v3-audio-only-20261020-v1",
        "status": status,
        "promotion": "forbidden_until_own_song_mmca_decoded_dmx_and_human_gates",
        "seed": args.seed, "steps": args.steps, "device": str(device),
        "train_groups": len(train_groups), "validation_groups": len(group_keys) - len(train_groups),
        "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "input_contract": "canonical MuQ1024 layer9 + Base92 92 = 1116; no zero padding",
        "body_inference_input": False,
        "muq_finetuned": False,
        "adapter": "hierarchical_local_middle_coarse",
        "trainable_parameters": trainable_parameter_report(model),
        "history": history, "validation_audits": val_audits,
        "deterministic_max_abs_diff": deterministic, "audio_swap_mean_abs_delta": audio_swap,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "muq_receipt": str(MUQ_ROOT / "COMPLETE_V2.json"), "muq_receipt_sha256": sha(MUQ_ROOT / "COMPLETE_V2.json"),
        "runtime_seconds": time.time() - started,
        "promotion_reason": "raw safety and output-collapse gates are prerequisites; no mapper or video was run",
    }
    (args.out / "TRAINING.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "status": status, "validation": val_audits,
                      "audio_swap": audio_swap}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

