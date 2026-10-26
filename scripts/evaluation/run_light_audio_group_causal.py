#!/usr/bin/env python3
"""Run isolated LightLayers audio-group masks on a sealed 94-D bus cache.

This is a cached single-song lighting-network diagnostic. It stops at palette/
event outputs, preserves the bus/checkpoint hashes, and never renders DMX/video.
Two-song qualification is impossible when the second bus cache is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    import torch
    from collections import OrderedDict
    from lightsem.dance.pipeline import accelerator_lock
    from lightsem.model.light_layers.training import AUDIO_SCALARS, load_checkpoint, generated_history
    from scripts.light_layers.prepare_visual_next_demo import FinitePaletteCapture

    bus_path = Path(args.bus).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    transforms = Path(args.transforms).resolve()
    with np.load(bus_path, allow_pickle=False) as z:
        normalized = np.asarray(z["normalized"], dtype=np.float32)
        raw = np.asarray(z["raw"], dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != 94 or not np.isfinite(normalized).all():
        raise ValueError("Expected finite normalized 94-D bus")
    arms = {
        "full": [],
        "zero_mel": list(range(64)),
        "zero_energy": [64 + AUDIO_SCALARS.index(x) for x in (
            "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid", "flux_hf",
            "energy_lf", "energy_lmid", "energy_mid", "energy_hmid", "energy_hf", "onset", "onset_ema")],
        "zero_timbre": [64 + AUDIO_SCALARS.index(x) for x in ("centroid", "flatness", "rolloff", "percussive_ratio")],
        "zero_rhythm": [64 + AUDIO_SCALARS.index(x) for x in (
            "pulse_clarity", "beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s")] + [92, 93],
    }
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = []
    with accelerator_lock(), torch.inference_mode():
        model, metadata = load_checkpoint(checkpoint, device="cuda")
        model.eval()
        for arm, columns in arms.items():
            x = normalized.copy()
            if columns:
                x[:, columns] = 0.0
            capture = FinitePaletteCapture(model)
            output = model.forward_chunked(torch.from_numpy(x[None]).cuda())
            rollout = generated_history(capture, output["context10"], output["chromatic_change_logit"],
                                        initial_ids=None, state_ablation=False, cue_threshold=.5, initialize=True)
            result = {key: value[0].detach().cpu().numpy() for key, value in {**output, **rollout}.items()}
            if not all(np.isfinite(v).all() for v in result.values() if isinstance(v, np.ndarray)):
                raise ValueError(f"nonfinite output in {arm}")
            np.savez_compressed(out / f"{arm}.npz", **result)
            rows.append({
                "arm": arm,
                "columns_zeroed": columns,
                "palette_ids_changed_fraction_vs_full": None,
                "chromatic_logit_rms": float(np.sqrt(np.mean(result["chromatic_change_logit"] ** 2))),
                "color_cue_count": int(np.count_nonzero(result["color_cues"])),
                "frames": int(len(result["generated_palette_ids"])),
                "finite": True,
            })
        full = np.load(out / "full.npz", allow_pickle=False)
        full_ids = full["generated_palette_ids"]
        for row in rows:
            arm = row["arm"]
            if arm != "full":
                with np.load(out / f"{arm}.npz", allow_pickle=False) as z:
                    ids = z["generated_palette_ids"]
                    row["palette_ids_changed_fraction_vs_full"] = float(np.mean(ids != full_ids))
        del full
        del model
        torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "qualification_status": "unqualified_single_song_cached_bus",
        "song": args.label,
        "bus_path": str(bus_path),
        "bus_sha256": sha(bus_path),
        "raw_bus_shape": list(raw.shape),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "transforms": str(transforms),
        "transforms_sha256": sha(transforms),
        "rows": rows,
        "scope": "94-D LightLayers audio-group masks; no body change, mapper, DMX or video",
        "missing_evidence": ["Shpongle 94-D bus cache", "full two-song own-audio recomputation", "decoded DMX audio-group retention"],
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bus", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--transforms", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", default="cure_for_me")
    a = p.parse_args()
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
