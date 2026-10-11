#!/usr/bin/env python3
"""Fit TRAIN-only affine translators from external spatial descriptors to R2.

The native R2 adapter was trained on the 64-D factorized StereoAudioV1
context. Feeding SpaIn/CRW/oBSRNN descriptors directly into that adapter is an
invalid distribution swap. This driver makes the comparison explicit: each
method is extracted from the same deterministic MUSDB TRAIN repan clips as the
factorized target, then a frozen affine translator is fit on TRAIN only. The
translator is a feature-domain bridge, not a new dance checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
import soundfile as sf

from lightsem.audio.stereo_v1 import extract_stereo_v1
from lightsem.experiments.audio_stereo.model import StereoConditioner, StereoConditionerConfig
from lightsem.experiments.audio_stereo.spain_pilot import SpaInNet, SpaInNetConfig, equal_power_repan
from lightsem.experiments.audio_stereo.crw_pilot import StereoCRW, StereoCRWConfig
from lightsem.experiments.audio_stereo.bsrnn_teacher import DEFAULT_SOURCE_ROOT, extract_streamed, load_teacher


TRAIN_MANIFEST = Path("/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/train_npz/MANIFEST.json")
TARGET_CHECKPOINT = Path("/mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/learning/R2/final.pt")
ANGLES = (-45.0, -15.0, 15.0, 45.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interp(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target, time, values[:, i]) for i in range(values.shape[1])], 1).astype(np.float32)


def mixture(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as z:
        stems = z["stems"].astype(np.float32)
        sample_rate = int(z["sample_rate"])
    if stems.shape[0] != 4 or not np.isfinite(stems).all():
        raise ValueError(f"Invalid TRAIN stems: {path}")
    values = torch.from_numpy(stems[None])
    angles = torch.tensor([ANGLES], dtype=values.dtype)
    rendered, _ = equal_power_repan(values, angles)
    return rendered[0].numpy().astype(np.float32), sample_rate


class TargetEncoder:
    def __init__(self, checkpoint: Path, device: str):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        keys = ("arm", "input_dim", "event_dim", "spatial_dim", "relation_dim", "width", "fused_dim", "spatial_hop", "relation_hop")
        config = StereoConditionerConfig(**{key: saved["config"][key] for key in keys})
        self.model = StereoConditioner(config).to(device).eval()
        self.model.load_state_dict(saved["model"], strict=True)
        self.device = device

    def __call__(self, wave: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        descriptor = extract_stereo_v1(wave, sample_rate)
        with torch.inference_mode():
            fused = self.model(torch.from_numpy(descriptor.features_40[None]).to(self.device))["fused"][0]
        fused = fused.float().cpu().numpy()
        time30 = np.arange(int(np.ceil(float(descriptor.time_40[-1]) * 30.0)) + 1, dtype=np.float64) / 30.0
        return time30, interp(descriptor.time_40, fused, time30)


def load_method(method: str, checkpoint: Path, device: str):
    if method == "spain":
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = SpaInNetConfig(**raw["config"]["config"])
        model = SpaInNet(cfg).to(device).eval()
        model.load_state_dict(raw["state_dict"], strict=True)
        angles = torch.tensor([ANGLES], dtype=torch.float32, device=device)

        def run(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with torch.inference_mode():
                _, out = model(torch.from_numpy(wave[None]).to(device), angles)
            feat = out["native_descriptor"][0].float().cpu().numpy()
            return np.arange(len(feat), dtype=np.float64) / 30.0, feat

        return run
    if method == "crw":
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        keys = ("sample_rate", "patch_size", "patch_stride", "patch_num", "n_fft", "win_length", "hop_length", "feature_dim", "tau", "cycle_num", "smooth_weight")
        cfg = StereoCRWConfig(**{key: raw["config"][key] for key in keys})
        model = StereoCRW(cfg).to(device).eval()
        model.load_state_dict(raw["state_dict"], strict=True)

        def run(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            step = round(cfg.sample_rate / 30.0)
            req = cfg.required_samples
            starts = np.arange(0, max(1, wave.shape[-1] - req + 1), step, dtype=int)
            values, times = [], []
            for begin in range(0, len(starts), 64):
                starts_batch = starts[begin:begin + 64]
                chunk = np.stack([wave[:, start:start + req] for start in starts_batch])
                if chunk.shape[-1] < req:
                    chunk = np.pad(chunk, ((0, 0), (0, 0), (0, req - chunk.shape[-1])))
                with torch.inference_mode():
                    values.append(model(torch.from_numpy(chunk).to(device))["native_features"].float().cpu().numpy())
                times.append(starts_batch / cfg.sample_rate)
            return np.concatenate(times), np.concatenate(values)

        return run
    if method == "bsrnn":
        teacher = load_teacher(checkpoint, device, source_root=DEFAULT_SOURCE_ROOT)

        def run(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            target_len = int(round(wave.shape[1] * 44100 / 16000))
            hi = np.stack([resample_poly(wave[c], 44100, 16000)[:target_len] for c in range(2)])
            temp = Path("/tmp") / f"lightsem_spatial_translator_{hashlib.sha256(wave.tobytes()).hexdigest()[:16]}.wav"
            temp.unlink(missing_ok=True)
            sf.write(temp, hi.T, 44100, subtype="PCM_16")
            try:
                with sf.SoundFile(str(temp), "r") as handle:
                    written_peak = 0.0
                    for block in handle.blocks(blocksize=1 << 20, dtype="float32", always_2d=True):
                        if block.size:
                            written_peak = max(written_peak, float(np.max(np.abs(block))))
                if not np.isfinite(written_peak) or written_peak <= 0.0:
                    raise ValueError(f"TRAIN repan WAV was silent after write: {temp}")
                extracted = extract_streamed(temp, teacher, split="train", chunk_seconds=10, hop_seconds=9, fps=40)
                return extracted.time_40, extracted.context_64
            finally:
                temp.unlink(missing_ok=True)

        return run
    raise ValueError(method)


def fit(method: str, method_checkpoint: Path, target_checkpoint: Path, train_manifest: Path, out: Path, device: str, ridge: float = 1e-2) -> dict:
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(train_manifest.read_text())
    rows = manifest["items"]
    target = TargetEncoder(target_checkpoint, device)
    source = load_method(method, method_checkpoint, device)
    source_rows, target_rows, ids = [], [], []
    for row in rows:
        wave, sample_rate = mixture(Path(row["path"]))
        if sample_rate != 16000:
            raise ValueError(f"Unexpected TRAIN rate {sample_rate}")
        source_time, source_value = source(wave)
        target_time, target_value = target(wave, sample_rate)
        lo, hi = max(source_time[0], target_time[0]), min(source_time[-1], target_time[-1])
        keep = (source_time >= lo) & (source_time <= hi)
        source_rows.append(source_value[keep].astype(np.float32))
        target_rows.append(interp(target_time, target_value, source_time[keep]))
        ids.append(row["source_id"])
    x = np.concatenate(source_rows, 0).astype(np.float64)
    y = np.concatenate(target_rows, 0).astype(np.float64)
    if x.shape[1] != 64 or y.shape[1] != 64 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError((x.shape, y.shape))
    input_mean, input_std = x.mean(0), np.maximum(x.std(0), 1e-4)
    output_mean, output_std = y.mean(0), np.maximum(y.std(0), 1e-4)
    x_norm, y_norm = (x - input_mean) / input_std, (y - output_mean) / output_std
    design = np.concatenate((x_norm, np.ones((len(x_norm), 1))), 1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ y_norm)
    prediction = design @ weights
    train_mse = float(np.mean((prediction - y_norm) ** 2))
    target_variance = float(np.mean((y_norm - y_norm.mean(0)) ** 2))
    out.mkdir(parents=True)
    np.savez_compressed(out / "translator.npz", weights=weights.astype(np.float32), input_mean=input_mean.astype(np.float32), input_std=input_std.astype(np.float32), output_mean=output_mean.astype(np.float32), output_std=output_std.astype(np.float32))
    report = {
        "schema_version": 1, "status": "complete", "method": method, "train_only": True, "test_access": False,
        "rows": len(rows), "frames": len(x), "ridge": ridge, "train_normalized_mse": train_mse, "target_variance": target_variance,
        "checkpoint": str(method_checkpoint), "checkpoint_sha256": sha(method_checkpoint),
        "target_checkpoint": str(target_checkpoint), "target_checkpoint_sha256": sha(target_checkpoint),
        "train_manifest": str(train_manifest), "train_manifest_sha256": sha(train_manifest), "source_ids": ids,
        "translator": str(out / "translator.npz"), "translator_sha256": sha(out / "translator.npz"),
        "scope": "frozen feature-domain translator; no native body training or method qualification",
    }
    report_path = out / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "REPORT.json.sha256.json").write_text(json.dumps({"sha256": sha(report_path)}) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("spain", "crw", "bsrnn"), required=True)
    parser.add_argument("--method-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, default=TARGET_CHECKPOINT)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = fit(args.method, args.method_checkpoint, args.target_checkpoint, args.train_manifest, args.out, args.device)
    print(json.dumps({key: result[key] for key in ("status", "method", "rows", "frames", "train_normalized_mse", "translator_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
