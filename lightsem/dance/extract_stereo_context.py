"""Build an authenticated 128-D stereo context for one complete waveform.

This is an inference-only helper for the native CLI.  It deliberately runs in
the audio/Torch environment, where torchaudio and the pinned oBSRNN teacher
are installed, while the native body model remains in the canonical Torch
environment.  The output contains raw SpaIn and oBSRNN 64-D streams
concatenated on the SpaIn 30 Hz clock; the native checkpoint applies its own
TRAIN normalizers later.
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

from lightsem.experiments.audio_stereo.bsrnn_teacher import (
    DEFAULT_SOURCE_ROOT,
    extract_streamed,
    load_teacher,
)
from lightsem.experiments.audio_stereo.spain_pilot import SpaInNet, SpaInNetConfig


SPAIN_CHECKPOINT = Path(
    "/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/learning/spain_seed0_repaired"
)
BSRNN_CHECKPOINT = Path(
    "/mnt/lightsem-data/checkpoints/three_spatial_methods/simo-bsrnn-opt/simo-bsrnn-opt/separator.ckpt"
)
SCHEMA = "spain_obsrnn_concat128_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.T
    if wave.shape[0] != 2 or not np.isfinite(wave).all():
        raise ValueError("Expected finite mastered stereo input")
    if rate != sample_rate:
        wave = np.stack(
            [resample_poly(wave[channel], sample_rate, rate) for channel in range(2)]
        )
    return wave.astype(np.float32)


def prepare_teacher_audio(path: Path, out: Path) -> tuple[Path, dict]:
    """Normalize only the stereo-teacher input; preserve the source waveform hash."""
    info = sf.info(str(path))
    if info.channels not in (1, 2):
        raise ValueError(f"Stereo teachers accept mono or stereo input, got {info.channels} channels")
    if info.samplerate == 44100 and info.channels == 2:
        return path, {"resampled": False, "channels_in": 2, "sample_rate_in": 44100}
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if wave.shape[1] == 1:
        wave = np.repeat(wave, 2, axis=1)
    if rate != 44100:
        wave = np.stack([resample_poly(wave[:, channel], 44100, rate) for channel in range(2)], axis=1)
    if not np.isfinite(wave).all() or not np.any(np.abs(wave) > 0):
        raise ValueError("Teacher-normalized audio is empty or non-finite")
    teacher_path = out / "teacher_input_44100_stereo.wav"
    sf.write(teacher_path, wave.astype(np.float32), 44100, subtype="PCM_16")
    return teacher_path, {"resampled": True, "channels_in": int(info.channels),
                          "sample_rate_in": int(info.samplerate), "teacher_path": str(teacher_path),
                          "teacher_sha256": sha(teacher_path)}


def spain_features(path: Path, device: str) -> tuple[np.ndarray, np.ndarray, str]:
    raw = torch.load(SPAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg_values = raw["config"]["config"]
    cfg = SpaInNetConfig(
        **{
            key: cfg_values[key]
            for key in (
                "sample_rate",
                "n_fft",
                "hop_length",
                "win_length",
                "width",
                "position_dim",
                "num_layers",
                "sources",
            )
            if key in cfg_values
        }
    )
    model = SpaInNet(cfg).to(device).eval()
    model.load_state_dict(raw["state_dict"], strict=True)
    wave = read_audio(path, cfg.sample_rate)
    angles = torch.tensor([[-45.0, -15.0, 15.0, 45.0]], dtype=torch.float32, device=device)
    chunk = cfg.sample_rate * 8
    times, values = [], []
    for start in range(0, wave.shape[-1], chunk):
        core = wave[:, start : min(wave.shape[-1], start + chunk)]
        if core.shape[-1] < chunk:
            core = np.pad(core, ((0, 0), (0, chunk - core.shape[-1])))
        with torch.inference_mode():
            _, features = model(torch.from_numpy(core[None]).to(device), angles)
        value = features["native_descriptor"][0].float().cpu().numpy()
        time = start / cfg.sample_rate + np.arange(len(value), dtype=np.float64) / 30.0
        keep = time < wave.shape[-1] / cfg.sample_rate
        times.append(time[keep])
        values.append(value[keep])
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(times), np.concatenate(values), sha(SPAIN_CHECKPOINT)


def bsrnn_features(path: Path, device: str) -> tuple[np.ndarray, np.ndarray, dict]:
    teacher = load_teacher(BSRNN_CHECKPOINT, device, source_root=DEFAULT_SOURCE_ROOT)
    result = extract_streamed(path, teacher, split="showcase", chunk_seconds=10, hop_seconds=9, fps=40)
    metadata = dict(result.metadata)
    metadata["source_root"] = str(DEFAULT_SOURCE_ROOT)
    return result.time_40, result.context_64, metadata


def interpolate(source_time: np.ndarray, source: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=np.float64)
    source = np.asarray(source, dtype=np.float32)
    order = np.argsort(source_time)
    source_time, source = source_time[order], source[order]
    keep = np.r_[True, np.diff(source_time) > 1e-9]
    source_time, source = source_time[keep], source[keep]
    if target_time[0] < source_time[0] - 0.1 or target_time[-1] > source_time[-1] + 0.1:
        raise ValueError("Stereo teachers do not cover the same source interval")
    return np.stack(
        [np.interp(target_time, source_time, source[:, index]) for index in range(source.shape[1])],
        axis=1,
    ).astype(np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    audio = args.wav.expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stereo context extraction requested CUDA but no GPU is available")
    args.out.mkdir(parents=True)
    teacher_audio, teacher_audio_meta = prepare_teacher_audio(audio, args.out)
    spain_time, spain, spain_sha = spain_features(teacher_audio, args.device)
    bsrnn_time, bsrnn, bsrnn_meta = bsrnn_features(teacher_audio, args.device)
    # Interpolate the oBSRNN stream on the shared 30 Hz SpaIn clock.
    bsrnn_on_spain = interpolate(bsrnn_time, np.asarray(bsrnn, dtype=np.float32), spain_time)
    fused = np.concatenate((np.asarray(spain, dtype=np.float32), bsrnn_on_spain), axis=1)
    if fused.shape != (len(spain_time), 128) or not np.isfinite(fused).all():
        raise ValueError(f"Invalid fused stereo context shape/values: {fused.shape}")
    context_path = args.out / "context.npz"
    np.savez_compressed(context_path, time30=spain_time, fused30=fused)
    manifest = {
        "schema_version": 2,
        "schema": SCHEMA,
        "method": "spain_obsrnn_concat128_v2",
        "feature_dim": 128,
        "status": "complete",
        "split": "showcase",
        "test_access": False,
        "automatic_for_waveform": True,
        "rows": [
            {
                "id": "input",
                "split": "showcase",
                "context_path": str(context_path),
                "context_sha256": sha(context_path),
                "audio_sha256": sha(audio),
                "feature_dim": 128,
                "frames30": int(len(spain_time)),
            }
        ],
        "audio_path": str(audio),
        "audio_sha256": sha(audio),
        "teacher_audio": teacher_audio_meta,
        "spain_checkpoint": str(SPAIN_CHECKPOINT),
        "spain_checkpoint_sha256": spain_sha,
        "obsrnn_checkpoint": str(BSRNN_CHECKPOINT),
        "obsrnn_checkpoint_sha256": bsrnn_meta["checkpoint_sha256"],
        "obsrnn_source_root": str(DEFAULT_SOURCE_ROOT),
        "obsrnn_source_commit": bsrnn_meta["source_root_commit"],
        "normalization": "raw teacher streams; native checkpoint TRAIN normalizers",
        "stereo_semantics": "mastered stereo image descriptors; not physical 3D location",
        "clock": "SpaIn 30 Hz; oBSRNN interpolated from 40 Hz",
    }
    manifest_path = args.out / "CONTEXT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "CONTEXT_MANIFEST.json.sha256.json").write_text(
        json.dumps({"sha256": sha(manifest_path)}) + "\n"
    )
    print(json.dumps({"status": "complete", "manifest": str(manifest_path), "audio_sha256": sha(audio)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
