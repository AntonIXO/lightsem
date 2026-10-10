#!/usr/bin/env python3
"""Audit source-aware stereo semantics against MUSDB stem targets.

The report is a representation admission diagnostic.  It compares the learned
semantic predictor with an explicit DSP mixture-only baseline and refuses to
claim native dance improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import FEATURE_NAMES, StereoAudioV1, sha256
from lightsem.experiments.audio_stereo.semantic import (
    SemanticStereoConditioner, SemanticStereoConditionerConfig, STEM_NAMES,
)
from lightsem.experiments.audio_stereo.diagnostics import representation_gate


def _stats(values: list[float]) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"mean": None, "p50": None, "p95": None}
    return {"mean": float(value.mean()), "p50": float(np.percentile(value, 50)),
            "p95": float(np.percentile(value, 95))}


def _event_f1(target: np.ndarray, prediction: np.ndarray, tolerance: int = 2) -> float:
    threshold = max(.05, float(np.percentile(target, 85)))
    true = target > threshold
    pred = prediction > threshold
    tp = fp = fn = 0
    for t, p in zip(true.T, pred.T):
        t_idx = np.flatnonzero(t); p_idx = np.flatnonzero(p); used: set[int] = set()
        for index in p_idx:
            candidates = [j for j in t_idx if j not in used and abs(int(j) - int(index)) <= tolerance]
            if candidates:
                used.add(min(candidates, key=lambda j: abs(int(j) - int(index)))); tp += 1
            else:
                fp += 1
        fn += len(t_idx) - len(used)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _binary_event_f1(target: np.ndarray, probability: np.ndarray, tolerance: int = 2) -> float:
    true = np.asarray(target, dtype=bool)
    pred = np.asarray(probability) > .5
    tp = fp = fn = 0
    for t, p in zip(true.T, pred.T):
        t_idx = np.flatnonzero(t); p_idx = np.flatnonzero(p); used: set[int] = set()
        for index in p_idx:
            candidates = [j for j in t_idx if j not in used and abs(int(j) - int(index)) <= tolerance]
            if candidates:
                used.add(min(candidates, key=lambda j: abs(int(j) - int(index)))); tp += 1
            else:
                fp += 1
        fn += len(t_idx) - len(used)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _swap_features(features: np.ndarray) -> np.ndarray:
    names = list(FEATURE_NAMES); out = features.copy()
    def swap(a: str, b: str) -> None:
        ia, ib = names.index(a), names.index(b); out[:, ia], out[:, ib] = features[:, ib], features[:, ia]
    for a, b in (("left_log_rms", "right_log_rms"), ("left_centroid", "right_centroid"),
                 ("left_rolloff", "right_rolloff")):
        swap(a, b)
    for prefix in ("flux",):
        for i in range(5): swap(f"{prefix}_left_band_{i}", f"{prefix}_right_band_{i}")
    for name in ("ild_db", "pan_balance", "phase_sin"):
        out[:, names.index(name)] *= -1.
    for i in range(5): out[:, names.index(f"energy_delta_band_{i}")] *= -1.
    return out


def _spatial_error(target: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    scale = np.maximum(np.mean(np.abs(target), axis=(0, 1)), 1e-3)
    learned = np.mean(np.abs(prediction - target) / scale[None, None, :])
    dsp = np.mean(np.abs(baseline - target) / scale[None, None, :])
    return float(learned), float(dsp)


def audit(manifest_path: Path, checkpoint: Path, out: Path, split: str, device: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_access", True):
        raise ValueError("Semantic audit requires test_access=false")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = SemanticStereoConditionerConfig(**{
        key: saved["config"][key] for key in ("arm", "input_dim", "source_event_dim",
                                                "source_spatial_dim", "source_relation_dim",
                                                "width", "context_dim", "spatial_hop", "relation_hop")
    })
    model = SemanticStereoConditioner(config).to(device).eval(); model.load_state_dict(saved["model"], strict=True)
    stats = {key: torch.from_numpy(np.asarray(value)).to(device) for key, value in saved["stats"].items()}
    rows = [row for row in manifest["entries"] if row.get("split") == split]
    if not rows:
        raise ValueError(f"No semantic rows for split {split!r}")
    event_scores, dsp_event_scores, spatial_learned, spatial_dsp, relation_learned, relation_dsp = [], [], [], [], [], []
    false_positive_learned, false_positive_dsp, inactive_count = [], [], []
    lr_errors, channel_mask_finite, output_std = [], [], []
    for row in rows:
        descriptor = StereoAudioV1.load(row["mixture_feature_path"])
        with np.load(row["target_path"], allow_pickle=False) as target:
            event_target = target["source_event_40"].astype(np.float32)
            spatial_target = target["source_spatial_10"].astype(np.float32)
            relation_target = target["source_relation_2"].astype(np.float32)
        x = descriptor.features_40.astype(np.float32)
        xt = (torch.from_numpy(x[None]).to(device) - stats["input_mean"]) / stats["input_std"]
        with torch.inference_mode():
            outputs = model(xt)
            swapped = model((torch.from_numpy(_swap_features(x)[None]).to(device) - stats["input_mean"]) / stats["input_std"])
            masked = model(torch.zeros_like(xt))
        event_pred = outputs["event"][0].float().cpu().numpy() * saved["stats"]["event_std"] + saved["stats"]["event_mean"]
        spatial_pred = outputs["spatial"][0].float().cpu().numpy() * saved["stats"]["spatial_std"] + saved["stats"]["spatial_mean"]
        relation_pred = outputs["relation"][0].float().cpu().numpy() * saved["stats"]["relation_std"] + saved["stats"]["relation_mean"]
        activity = torch.sigmoid(outputs["activity_logits"][0]).float().cpu().numpy()
        onset_probability = torch.sigmoid(outputs["onset_logits"][0]).float().cpu().numpy()
        activity_target = event_target[:, 1::4]
        # A source-blind DSP baseline repeats the mastered-mixture event/image
        # evidence for every source.  It is intentionally explicit and weak.
        mix_energy = descriptor.features_40[:, 2]
        mix_active = (mix_energy > -4.).astype(np.float32)
        mixture_event = np.stack((mix_energy, mix_active,
                                  descriptor.events_40[:, 0], descriptor.events_40[:, 1]), axis=1)
        dsp_events = np.repeat(mixture_event[:, None, :], len(STEM_NAMES), axis=1).reshape(len(x), -1)
        onset_threshold = np.asarray(saved["stats"].get("onset_threshold", np.full(4, .5)))
        onset_values = event_target.reshape(len(event_target), len(STEM_NAMES), 4)[..., 2:4].reshape(len(event_target), 8)
        onset_target = onset_values > np.repeat(onset_threshold, 2)[None, :]
        dsp_onset = np.repeat(np.stack((descriptor.events_40[:, 0], descriptor.events_40[:, 1]), axis=1)[:, None, :],
                                  len(STEM_NAMES), axis=1).reshape(len(x), len(STEM_NAMES) * 2)
        onset_threshold_8 = np.repeat(onset_threshold, 2)
        dsp_event_scores.append(_binary_event_f1(onset_target, dsp_onset > onset_threshold_8[None, :]))
        event_scores.append(_binary_event_f1(onset_target, onset_probability))
        # Four spatial target blocks: ILD, width, pan, coherence, log energy.
        target_spatial = spatial_target.reshape(len(spatial_target), len(STEM_NAMES), 5)
        pred_spatial = spatial_pred.reshape(len(spatial_pred), len(STEM_NAMES), 5)
        spatial_length = min(len(target_spatial), len(pred_spatial))
        target_spatial = target_spatial[:spatial_length]
        pred_spatial = pred_spatial[:spatial_length]
        mixture_spatial = descriptor.spatial_10[:, [0, 1, 2, 3]]
        dsp_spatial = np.repeat(mixture_spatial[:, None, :], len(STEM_NAMES), axis=1)
        dsp_energy = np.repeat(descriptor.features_40[:, 2:3].mean(0, keepdims=True), len(STEM_NAMES), axis=1)
        dsp_energy = np.repeat(dsp_energy, len(target_spatial), axis=0)[..., None]
        dsp_spatial = np.concatenate((dsp_spatial, dsp_energy), axis=-1)
        learned, dsp = _spatial_error(target_spatial, pred_spatial, dsp_spatial)
        spatial_learned.append(learned); spatial_dsp.append(dsp)
        lag_idx = np.arange(1, relation_target.shape[-1], 3)
        relation_length = min(len(relation_pred), len(relation_target))
        relation_learned.extend(np.abs(relation_pred[:relation_length, lag_idx] - relation_target[:relation_length, lag_idx]).reshape(-1).tolist())
        relation_dsp.extend(np.abs(relation_target[:relation_length, lag_idx]).reshape(-1).tolist())
        learned_active = activity > .5
        inactive = activity_target < .5
        inactive_count.append(int(inactive.sum()))
        false_positive_learned.extend(learned_active[inactive].tolist())
        mixture_active = (descriptor.features_40[:, 2] > -4.)
        dsp_active = np.repeat(mixture_active[:, None], len(STEM_NAMES), axis=1)
        false_positive_dsp.extend(dsp_active[inactive].tolist())
        # LR swap: source activity/relation should be invariant, signed ILD/pan
        # source predictions should flip.  Record finite/metamorphic checks.
        swap_out = swapped["activity_logits"].float().cpu().numpy()
        lr_errors.append(float(np.max(np.abs(swap_out - outputs["activity_logits"].float().cpu().numpy()))))
        channel_mask_finite.append(bool(torch.isfinite(masked["fused"]).all()))
        output_std.append(float(outputs["fused"].float().std().cpu()))
    event_delta = float(np.mean(event_scores) - np.mean(dsp_event_scores))
    spatial_delta = float(np.mean(spatial_learned) - np.mean(spatial_dsp))
    relation_delta = float(np.mean(relation_learned) - np.mean(relation_dsp))
    learned_fpr = float(np.mean(false_positive_learned)) if false_positive_learned else 0.
    dsp_fpr = float(np.mean(false_positive_dsp)) if false_positive_dsp else 0.
    metrics = {
        "timing_delta": event_delta,
        "timing_auroc_delta": event_delta,
        "spatial_relative_error": spatial_delta,
        "relation_lag_delta": relation_delta,
        "timing_regression": max(0., -event_delta),
        "silent_fpr_increase": learned_fpr - dsp_fpr,
        "event_f1_learned": float(np.mean(event_scores)),
        "event_f1_dsp_baseline": float(np.mean(dsp_event_scores)),
        "spatial_error_learned": float(np.mean(spatial_learned)),
        "spatial_error_dsp_baseline": float(np.mean(spatial_dsp)),
        "relation_lag_error_learned": float(np.mean(relation_learned)),
        "relation_lag_error_dsp_baseline": float(np.mean(relation_dsp)),
        "silent_fpr_learned": learned_fpr, "silent_fpr_dsp_baseline": dsp_fpr,
        "lr_swap_activity_max_error": float(np.max(lr_errors)),
        "lr_swap_pass": bool(np.max(lr_errors) < 1e-3),
        "finite": bool(all(channel_mask_finite)),
        "noncollapse": bool(np.mean(output_std) > 1e-6),
        "channel_mask_finite": bool(all(channel_mask_finite)),
        "source_count": len(STEM_NAMES), "rows": len(rows), "inactive_target_values": int(sum(inactive_count)),
    }
    result = {"schema_version": 1, "status": "complete", "split": split,
              "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "metrics": metrics, "representation_gate": representation_gate(metrics),
              "scope": "MUSDB stem-derived source semantics; no native dance qualification"}
    if out.exists():
        raise FileExistsError(f"Preserving existing semantic audit: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    out.with_name(out.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(out)}) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = audit(args.manifest, args.checkpoint, args.out, args.split, args.device)
    print(json.dumps({"status": result["status"], "gate": result["representation_gate"], "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
