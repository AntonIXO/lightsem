#!/usr/bin/env python3
"""Evaluate an audio-only v3 checkpoint on the two required own-song inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.foundation_audio_light.direct_model import AudioOnlyDirectLightModel, DirectLightConfig
from lightsem.experiments.foundation_audio_light.safety_projection import SafetyConfig
from scripts.experiments.run_foundation_v3_audio_only import audit_output


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align_features(muq_path: Path, base92_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(muq_path, allow_pickle=False) as z:
        muq = np.asarray(z["embeddings" if "embeddings" in z.files else "muq"], np.float32)
        muq_time = np.asarray(z["timestamps" if "timestamps" in z.files else "muq_time"], np.float64)
    with np.load(base92_path, allow_pickle=False) as z:
        base = np.asarray(z["x30" if "x30" in z.files else "x"], np.float32)
        base_time = np.asarray(z["time30" if "time30" in z.files else "time"], np.float64)
    end = min(float(muq_time[-1]), float(base_time[-1]))
    target = np.arange(0.0, end + 1e-6, 0.1, dtype=np.float64)
    muq_idx = np.searchsorted(muq_time, target, side="left").clip(0, len(muq_time) - 1)
    base_idx = np.searchsorted(base_time, target, side="left").clip(0, len(base_time) - 1)
    return np.concatenate([muq[muq_idx], base[base_idx]], 1).astype(np.float32), target


def load_model(checkpoint: Path, device: torch.device) -> AudioOnlyDirectLightModel:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["config"]
    foundation_payload = cfg["foundation"]
    allowed = {field.name for field in fields(AudioToLightConfig)}
    foundation = AudioToLightConfig(**{key: value for key, value in foundation_payload.items() if key in allowed})
    model = AudioOnlyDirectLightModel(DirectLightConfig(
        foundation=foundation,
        anchor_pan_scale=float(cfg.get("anchor_pan_scale", 0.90)),
        anchor_tilt_scale=float(cfg.get("anchor_tilt_scale", 0.25)),
        residual_gain=float(cfg.get("residual_gain", 0.10)),
        safety=SafetyConfig(**cfg.get("safety", {})),
        dynamics_mode=str(cfg.get("dynamics_mode", "hard_clipped")),
        dynamics_drive_scale=float(cfg.get("dynamics_drive_scale", 4.0)),
        dynamics_position_gain=float(cfg.get("dynamics_position_gain", 1.0)),
        dynamics_velocity_gain=float(cfg.get("dynamics_velocity_gain", 2.0)),
        dynamics_acceleration_gain=float(cfg.get("dynamics_acceleration_gain", 2.0)),
        dynamics_radial_bounds=bool(cfg.get("dynamics_radial_bounds", False)),
    )).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    return model


def infer(model, audio: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    fixtures = torch.from_numpy(make_fixture_metadata()[None]).float().to(device)
    total = len(audio)
    state = None
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in ("aim", "velocity", "acceleration", "jerk", "rgb", "brightness", "event", "audio_latent", "anchor")}
    with torch.no_grad():
        for start in range(0, total, 128):
            end = min(total, start + 128)
            x = torch.from_numpy(audio[start:end][None]).float().to(device)
            valid = torch.ones((1, end - start), dtype=torch.bool, device=device)
            timestamps = torch.arange(start, end, device=device, dtype=torch.float32)[None] / 10.0
            output = model(x, fixtures, timestamps=timestamps, valid=valid, state=state, duration_seconds=total / 10.0)
            state = output["state"]
            for key in chunks:
                chunks[key].append(output[key].detach().cpu().numpy())
    return {key: np.concatenate(values, 1) for key, values in chunks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shpongle-muq", type=Path, required=True)
    parser.add_argument("--shpongle-base92", type=Path, required=True)
    parser.add_argument("--cure-muq", type=Path, required=True)
    parser.add_argument("--cure-base92", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = load_model(args.checkpoint, device)
    sources = {
        "shpongle": (args.shpongle_muq, args.shpongle_base92),
        "cure_for_me": (args.cure_muq, args.cure_base92),
    }
    reports = {}
    predictions = {}
    for name, (muq, base92) in sources.items():
        audio, times = align_features(muq, base92)
        output = infer(model, audio, device)
        report = audit_output({key: torch.from_numpy(value) for key, value in output.items()}, 10.0)
        report.update({"frames": int(len(audio)), "duration_s": float(times[-1]), "audio_feature_sha256": sha(muq), "base92_sha256": sha(base92)})
        reports[name] = report
        predictions[name] = output
    n = min(len(predictions["shpongle"]["aim"][0]), len(predictions["cure_for_me"]["aim"][0]))
    swap = {}
    for key in ("aim", "rgb", "brightness", "event"):
        swap[key] = float(np.mean(np.abs(predictions["shpongle"][key][0, :n] - predictions["cure_for_me"][key][0, :n])))
    raw_pass = all(row["raw_safety_pass"] for row in reports.values())
    collapse_pass = all(row["output_collapse_pass"] for row in reports.values())
    conditioning_pass = bool(swap["aim"] > 1e-4 and swap["rgb"] > 1e-4)
    status = "diagnostic_candidate" if raw_pass and collapse_pass and conditioning_pass else "rejected_raw_safety" if not raw_pass else "output_collapsed" if not collapse_pass else "unconditioned_or_shared_prior"
    args.out.mkdir(parents=True, exist_ok=False)
    for name, output in predictions.items():
        np.savez_compressed(args.out / f"PREDICTIONS_{name}.npz", time=np.arange(len(output["aim"][0])) / 10.0, **{key: value[0] for key, value in output.items()})
    receipt = {
        "schema": "lightsem.foundation_v3_own_song_audit.v1",
        "protocol_id": "foundation-v3-own-song-audit-20261020-v1",
        "status": status,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
        "inference_contract": "audio-only; fixture metadata only; 10 Hz control clock; no body input; no native DMX",
        "songs": reports,
        "audio_swap_mean_abs_delta": swap,
        "raw_safety_pass": raw_pass,
        "output_collapse_pass": collapse_pass,
        "conditioning_pass": conditioning_pass,
        "promotion": "forbidden_until_decoded_dmx_and_human_gates",
    }
    (args.out / "OWN_SONG_AUDIT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

