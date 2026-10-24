#!/usr/bin/env python3
"""Generate fixed-weight audio swap, shift and same-BPM donor controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.evaluate_foundation_v3_own_songs import align_features, infer, load_model, sha
from scripts.experiments.run_foundation_v3_audio_only import CONTRACT, load_row


def bpm_from_timeline(path: Path) -> float:
    with np.load(path, allow_pickle=False) as z:
        beats = np.asarray(z["beats"], dtype=np.float64)
    return float(60.0 / np.median(np.diff(beats)))


def bpm_from_row(row: dict) -> float:
    period = np.asarray(row["audio"][:, 1115], dtype=np.float64)
    return float(60.0 / np.median(period))


def save_prediction(path: Path, values: dict[str, np.ndarray], total: int) -> None:
    np.savez_compressed(path, time=np.arange(total, dtype=np.float64) / 10.0,
                        **{key: value[0] for key, value in values.items() if key != "state"})


def mean_abs_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], key: str) -> float:
    n = min(a[key].shape[1], b[key].shape[1])
    return float(np.mean(np.abs(a[key][:, :n] - b[key][:, :n])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shpongle-muq", type=Path, required=True)
    parser.add_argument("--shpongle-base92", type=Path, required=True)
    parser.add_argument("--cure-muq", type=Path, required=True)
    parser.add_argument("--cure-base92", type=Path, required=True)
    parser.add_argument("--shpongle-timeline", type=Path, required=True)
    parser.add_argument("--cure-timeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing control root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = load_model(args.checkpoint, device)
    rows = [load_row(row) for row in json.loads(CONTRACT.read_text())["rows"]]
    by_target = {}
    for target_bpm, min_len in ((bpm_from_timeline(args.shpongle_timeline), 1800), (bpm_from_timeline(args.cure_timeline), 1800)):
        candidates = [row for row in rows if len(row["audio"]) >= min_len]
        by_target[target_bpm] = min(candidates, key=lambda row: abs(bpm_from_row(row) - target_bpm))
    sources = {
        "shpongle": (args.shpongle_muq, args.shpongle_base92, args.shpongle_timeline),
        "cure_for_me": (args.cure_muq, args.cure_base92, args.cure_timeline),
    }
    controls: dict[str, dict[str, object]] = {}
    for label, (muq_path, base_path, timeline_path) in sources.items():
        own_audio, own_time = align_features(muq_path, base_path)
        own = infer(model, own_audio, device)
        donor = by_target[bpm_from_timeline(timeline_path)]
        donor_audio = donor["audio"][:len(own_audio)].astype(np.float32, copy=False)
        shifted_audio = np.roll(own_audio, 30, axis=0)
        same_bpm = infer(model, donor_audio, device)
        shifted = infer(model, shifted_audio, device)
        own_path = args.out / f"PREDICTIONS_{label}_own.npz"
        same_path = args.out / f"PREDICTIONS_{label}_same_bpm.npz"
        shift_path = args.out / f"PREDICTIONS_{label}_circular_shift_3s.npz"
        save_prediction(own_path, own, len(own_audio))
        save_prediction(same_path, same_bpm, len(donor_audio))
        save_prediction(shift_path, shifted, len(shifted_audio))
        controls[label] = {
            "own_bpm": bpm_from_timeline(timeline_path),
            "same_bpm_donor_id": donor["id"],
            "same_bpm_donor_bpm": bpm_from_row(donor),
            "own_muq_sha256": sha(muq_path),
            "own_base92_sha256": sha(base_path),
            "own_prediction": str(own_path.resolve()),
            "same_bpm_prediction": str(same_path.resolve()),
            "shift_prediction": str(shift_path.resolve()),
            "deltas_own_vs_same_bpm": {key: mean_abs_delta(own, same_bpm, key) for key in ("aim", "rgb", "brightness", "event", "audio_latent")},
            "deltas_own_vs_shift": {key: mean_abs_delta(own, shifted, key) for key in ("aim", "rgb", "brightness", "event", "audio_latent")},
        }
    receipt = {
        "schema": "lightsem.foundation_v3_controls.v1",
        "protocol_id": "foundation-v3-jerk-safe-controls-20261020-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "fixed_inputs": "same checkpoint, seed/noise, fixture metadata, clock and inference code; only audio changed",
        "controls": controls,
        "promotion": "diagnostic only; controls do not establish musicality or decoded-DMX retention",
    }
    (args.out / "CONTROLS.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
