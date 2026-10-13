#!/usr/bin/env python3
"""Build authenticated local complex-Fourier contexts for native pilots.

The context is deliberately an isolated feature intervention.  It is computed
from the already authenticated 30 Hz MusicTimeline descriptors, so it does not
change the waveform, MuQ, source masks, pose split, or native clock.  Each
selected channel receives real and imaginary Hann-windowed DFT coordinates at
two temporal scales.  These are context features, not authored lighting labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_fourier(x: np.ndarray) -> np.ndarray:
    """Return 128 real/imag local DFT channels on the input 30 Hz grid."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError(f"expected finite [T,8] selected descriptors, got {x.shape}")
    outputs = []
    # 2 s: .5, 1, 2, 4 Hz; .5 s: 2, 4, 8, 16 Hz.  The latter deliberately
    # exposes fast transients and is retained as a diagnostic even when the
    # downstream raw-jerk gate rejects a candidate.
    for width, harmonics in ((60, (1, 2, 4, 8)), (15, (1, 2, 4, 7))):
        window = np.hanning(width).astype(np.float32)
        window /= max(float(np.linalg.norm(window)), 1e-8)
        padded = np.pad(x, ((width // 2, width - 1 - width // 2), (0, 0)), mode="edge")
        frames = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
        # sliding_window_view returns [T,8,width] for axis=0; transpose once
        # so the contraction below is explicit and shape-checked.
        frames = np.moveaxis(frames, -1, 1)
        if frames.shape != (len(x), width, 8):
            raise ValueError(f"unexpected local window shape {frames.shape}")
        for harmonic in harmonics:
            phase = 2.0 * np.pi * harmonic * np.arange(width, dtype=np.float32) / width
            cosine = (window * np.cos(phase))[None, :, None]
            sine = (window * np.sin(phase))[None, :, None]
            outputs.append(np.sum(frames * cosine, axis=1))
            outputs.append(np.sum(frames * sine, axis=1))
    result = np.concatenate(outputs, axis=1).astype(np.float32)
    if result.shape != (len(x), 128) or not np.isfinite(result).all():
        raise ValueError(f"invalid Fourier context shape/values {result.shape}")
    return result


def build(args: argparse.Namespace) -> dict:
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"preserving existing Fourier context root: {out}")
    out.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads(args.context_manifest.read_text())
    feature_manifest = json.loads(args.feature_manifest.read_text())
    feature_rows = {row["key"]: row for row in feature_manifest["entries"]}
    base_manifest = json.loads(args.data_manifest.read_text())
    data_rows = {row["key"]: row for row in base_manifest["entries"]}
    rows = source_manifest["rows"]
    if source_manifest.get("split") != "train" or any(row.get("split") != "train" for row in rows):
        raise ValueError("Fourier pilot admits TRAIN context rows only")
    context_rows = []
    for row in rows:
        key = row["id"]
        data = data_rows.get(key)
        if data is None or data.get("partition") != "train":
            raise ValueError(f"missing TRAIN data row for {key}")
        feature = feature_rows.get(key)
        if feature is None:
            raise ValueError(f"missing authenticated feature row for {key}")
        timeline_path = Path(feature["timeline_path"]).resolve()
        if sha(timeline_path) != feature["timeline_sha256"]:
            raise ValueError(f"changed timeline cache for {key}")
        timeline = MusicTimeline.load(timeline_path)
        time = np.asarray(timeline.time, dtype=np.float64)
        extra, names = model_features(timeline, time)
        # Mixture 4 channels at 1040:1044 and source flux at every fourth
        # source descriptor (1024 + source*4 + 2).
        selected = np.concatenate((extra[:, 1040:1044], extra[:, [1026, 1030, 1034, 1038]]), axis=1)
        context = local_fourier(selected)
        destination = out / "contexts" / "train" / f"{key}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(stream, time30=time, fused30=context)
        context_rows.append({
            "id": key,
            "group": data["group"],
            "split": "train",
            "audio_sha256": data["wav_sha256"],
            "context_path": str(destination),
            "context_sha256": sha(destination),
            "feature_key": feature["feature_key"],
            "timeline_sha256": feature["timeline_sha256"],
            "selected_feature_names": [names[i] for i in (1040, 1041, 1042, 1043, 1026, 1030, 1034, 1038)],
        })
    manifest = {
        "schema_version": 1,
        "status": "train_only_local_fourier_v1",
        "method": "local_complex_fourier",
        "schema": "local_fourier_128_v1",
        "feature_dim": 128,
        "windows_frames": [60, 15],
        "frequencies_hz": {"60": [0.5, 1.0, 2.0, 4.0], "15": [2.0, 4.0, 8.0, 14.0]},
        "channels": "mixture_rms/onset/centroid/flux plus four source_flux channels",
        "normalization": "native TRAIN-only adapter normalization",
        "split": "train",
        "test_access": False,
        "source_manifests": {
            "contexts": str(args.context_manifest),
            "contexts_sha256": sha(args.context_manifest),
            "features": str(args.feature_manifest),
            "features_sha256": sha(args.feature_manifest),
            "data": str(args.data_manifest),
            "data_sha256": sha(args.data_manifest),
        },
        "rows": context_rows,
    }
    manifest_path = out / "CONTEXT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest_path.with_name(manifest_path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(manifest_path)}) + "\n")
    return {"out": str(out), "rows": len(context_rows), "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
