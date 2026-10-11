#!/usr/bin/env python3
"""Extract external spatial-method contexts for paired native TRAIN clips.

The output is consumed by the native end-to-end trainer. Every row comes from
an admitted native TRAIN pose/audio pair; no showcase or TEST audio is allowed.
MuQ/Base92 remain the incumbent inputs and this branch adds one external 64-D
method context.
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

from lightsem.experiments.audio_stereo.bsrnn_teacher import DEFAULT_SOURCE_ROOT, context_64, extract_streamed, load_teacher
from lightsem.experiments.audio_stereo.crw_pilot import StereoCRW, StereoCRWConfig
from lightsem.experiments.audio_stereo.spain_pilot import SpaInNet, SpaInNetConfig

NATIVE_MANIFEST = Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b_stereo_v1/STEREO_NATIVE_MANIFEST.json")
SPAIN_CHECKPOINT = Path("/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/learning/spain_seed0_repaired")
CRW_CHECKPOINT = Path("/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/learning/crw_seed0/final.pt")
BSRNN_CHECKPOINT = Path("/mnt/lightsem-data/checkpoints/three_spatial_methods/simo-bsrnn-opt/simo-bsrnn-opt/separator.ckpt")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.T
    if rate != sample_rate:
        wave = np.stack([resample_poly(wave[c], sample_rate, rate) for c in range(wave.shape[0])])
    if wave.shape[0] != 2 or not np.isfinite(wave).all():
        raise ValueError(f"Expected finite mastered stereo audio: {path}")
    return wave.astype(np.float32)


def load_spain(checkpoint: Path, device: str):
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = SpaInNetConfig(**raw["config"]["config"])
    model = SpaInNet(cfg).to(device).eval()
    model.load_state_dict(raw["state_dict"], strict=True)
    angles = torch.tensor([[-45.0, -15.0, 15.0, 45.0]], dtype=torch.float32, device=device)

    def extract(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, times = [], []
        chunk = cfg.sample_rate * 8
        for start in range(0, wave.shape[-1], chunk):
            core = wave[:, start:min(wave.shape[-1], start + chunk)]
            if core.shape[-1] < chunk:
                core = np.pad(core, ((0, 0), (0, chunk - core.shape[-1])))
            with torch.inference_mode():
                _, features = model(torch.from_numpy(core[None]).to(device), angles)
            value = features["native_descriptor"][0].float().cpu().numpy()
            time = start / cfg.sample_rate + np.arange(len(value), dtype=np.float64) / 30.0
            keep = time < wave.shape[-1] / cfg.sample_rate
            values.append(value[keep]); times.append(time[keep])
        return np.concatenate(times), np.concatenate(values)

    return extract


def load_crw(checkpoint: Path, device: str):
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    keys = ("sample_rate", "patch_size", "patch_stride", "patch_num", "n_fft", "win_length", "hop_length", "feature_dim", "tau", "cycle_num", "smooth_weight")
    cfg = StereoCRWConfig(**{key: raw["config"][key] for key in keys})
    model = StereoCRW(cfg).to(device).eval()
    model.load_state_dict(raw["state_dict"], strict=True)

    def extract(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        req, step = cfg.required_samples, round(cfg.sample_rate / 30.0)
        starts = np.arange(0, max(1, wave.shape[-1] - req + 1), step, dtype=int)
        values, times = [], []
        for begin in range(0, len(starts), 64):
            starts_batch = starts[begin:begin + 64]
            batch = np.stack([wave[:, start:start + req] for start in starts_batch])
            if batch.shape[-1] < req:
                batch = np.pad(batch, ((0, 0), (0, 0), (0, req - batch.shape[-1])))
            with torch.inference_mode():
                values.append(model(torch.from_numpy(batch).to(device))["native_features"].float().cpu().numpy())
            times.append(starts_batch / cfg.sample_rate)
        return np.concatenate(times), np.concatenate(values)

    return extract


def load_bsrnn(checkpoint: Path, device: str):
    teacher = load_teacher(checkpoint, device, source_root=DEFAULT_SOURCE_ROOT)

    def extract(path: Path) -> tuple[np.ndarray, np.ndarray]:
        result = extract_streamed(path, teacher, split="train", chunk_seconds=10, hop_seconds=9, fps=40)
        return result.time_40, result.context_64

    return extract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("spain", "crw", "bsrnn"), required=True)
    parser.add_argument("--native-manifest", type=Path, default=NATIVE_MANIFEST)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.native_manifest.read_text())
    if manifest.get("test_access", True):
        raise ValueError("Native context extraction requires test_access=false")
    rows = manifest["entries"]
    checkpoint = args.checkpoint or {"spain": SPAIN_CHECKPOINT, "crw": CRW_CHECKPOINT, "bsrnn": BSRNN_CHECKPOINT}[args.method]
    extractor = load_spain(checkpoint, args.device) if args.method == "spain" else load_crw(checkpoint, args.device) if args.method == "crw" else load_bsrnn(checkpoint, args.device)
    out_rows = []
    args.out.mkdir(parents=True)
    for row in rows:
        audio = Path(row["wav_path"])
        if row.get("partition") != "train" or not audio.is_file():
            raise ValueError(f"Non-TRAIN or missing native row: {row['key']}")
        if args.method == "bsrnn":
            time, context = extractor(audio)
        else:
            wave = read_audio(audio)
            time, context = extractor(wave)
        if context.ndim != 2 or context.shape[1] != 64 or not np.isfinite(context).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f"Invalid {args.method} context for {row['key']}: {context.shape}")
        destination = args.out / f"{row['key']}.npz"
        np.savez_compressed(destination, time30=time.astype(np.float64), fused30=context.astype(np.float32))
        out_rows.append({'id':row['key'],'native_key':row['key'],'group':row['group'],'split':'train','context_path':str(destination),'context_sha256':sha(destination),'audio_path':str(audio),'audio_sha256':sha(audio),'frames30':len(time),'feature_dim':64})
    output = {'schema_version':1,'status':'complete','method':args.method,'rows':out_rows,'source_manifest':str(args.native_manifest),'source_manifest_sha256':sha(args.native_manifest),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'split':'train','test_access':False,'scope':'external spatial features extracted on paired native TRAIN audio; no showcase/TEST access'}
    manifest_path = args.out / 'CONTEXT_MANIFEST.json'
    manifest_path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    (args.out/'CONTEXT_MANIFEST.json.sha256.json').write_text(json.dumps({'sha256':sha(manifest_path)})+'\n')
    print(json.dumps({'status':'complete','method':args.method,'rows':len(out_rows),'manifest':str(manifest_path)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
