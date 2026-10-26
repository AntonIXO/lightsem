#!/usr/bin/env python3
"""Cached LightLayers audio-identity donor counterfactual through decoded DMX.

The model receives one song's authenticated 94-D bus while the selected body,
mapper, P2 policy and LUFS brightness remain those of the target song.  The
script stops at decoded DMX and never opens or regenerates raw audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluation.run_light_audio_group_dmx_causal import (
    _apply_two_group_palette, _metrics, sha,
)


def _bus(path: Path):
    with np.load(path, allow_pickle=False) as z:
        normalized = np.asarray(z["normalized"], dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != 94 or not np.isfinite(normalized).all():
        raise ValueError(f"Invalid 94-D bus: {path}")
    return normalized


def _infer(model, bus: np.ndarray):
    import torch
    from lightsem.model.light_layers.training import generated_history
    from scripts.light_layers.prepare_visual_next_demo import FinitePaletteCapture

    with torch.inference_mode():
        output = model.forward_chunked(torch.from_numpy(bus[None]).cuda())
        capture = FinitePaletteCapture(model)
        rollout = generated_history(capture, output["context10"], output["chromatic_change_logit"],
                                    initial_ids=None, state_ablation=False,
                                    cue_threshold=.5, initialize=True)
        result = {key: value[0].detach().cpu().numpy()
                  for key, value in {**output, **rollout}.items()}
    if not all(np.isfinite(value).all() for value in result.values()
               if isinstance(value, np.ndarray)):
        raise ValueError("Nonfinite LightLayers donor output")
    return result


def run(args):
    import torch
    from lightsem.dance.pipeline import accelerator_lock
    from lightsem.model.light_layers.training import load_checkpoint

    buses = {song: Path(path).resolve() for song, path in args.bus.items()}
    transforms = json.loads(Path(args.transforms).resolve().read_text())
    bodies = {}
    patches = {}
    generations = {}
    for song in args.songs:
        root = Path(args.baseline_root).resolve() / song
        with np.load(root / "lights.npz", allow_pickle=False) as z:
            dmx = np.asarray(z["universes"], dtype=np.uint8)
        patch = json.loads((root / "patch.json").read_text())
        generation = json.loads((root / "GENERATION.json").read_text())
        if generation.get("raw_safety_gate", {}).get("status") != "pass":
            raise ValueError(f"Selected body raw safety is not passing: {song}")
        bodies[song] = dmx[:, 0, :]
        patches[song] = patch
        generations[song] = generation
    model, metadata = None, None
    with accelerator_lock():
        model, metadata = load_checkpoint(Path(args.checkpoint).resolve(), device="cuda")
        model.eval()
        inferred = {song: _infer(model, _bus(path)) for song, path in buses.items()}
        del model
        torch.cuda.empty_cache()

    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    songs = []
    for target in args.songs:
        donor = args.songs[1] if target == args.songs[0] else args.songs[0]
        ids_target = inferred[target]["generated_palette_ids"]
        cues_target = inferred[target]["color_cues"]
        ids_donor = inferred[donor]["generated_palette_ids"]
        cues_donor = inferred[donor]["color_cues"]
        target_frames = len(bodies[target])
        n10 = min(len(ids_target), len(ids_donor), target_frames // 4)
        n40 = n10 * 4
        baseline = bodies[target][:n40]
        patch = patches[target]
        rows = []
        target_dir = out / target
        target_dir.mkdir()
        for arm, ids, cues in (
            ("factual", ids_target[:n10], cues_target[:n10]),
            ("donor_audio", ids_donor[:n10], cues_donor[:n10]),
        ):
            if arm == "factual":
                candidate = baseline.copy()
                delivery = {"scope": "sealed target P2 full arm"}
            else:
                candidate, delivery = _apply_two_group_palette(
                    baseline, ids, cues, patch,
                    np.asarray(transforms["palette_prototypes"], dtype=np.float64),
                )
            metrics = _metrics(baseline[:, None, :], candidate[:, None, :], patch)
            np.savez_compressed(target_dir / f"{arm}.npz", baseline_dmx=baseline[:, None, :],
                                candidate_dmx=candidate[:, None, :], palette_ids10=ids,
                                color_cues10=cues)
            rows.append({"arm": arm, "donor_song": donor if arm == "donor_audio" else target,
                         "frames": n40, "seconds": n40 / 40.0, "delivery": delivery, **metrics})
        with np.load(target_dir / "donor_audio.npz", allow_pickle=False) as z:
            donor_candidate = np.asarray(z["candidate_dmx"], dtype=np.uint8)
        donor_effect = _metrics(bodies[target][:n40, None, :], donor_candidate, patch)
        rows[1]["effect_vs_factual"] = {
            "rgb_rms_8bit": donor_effect["rgb_rms_8bit"],
            "brightness_rms": donor_effect["brightness_rms"],
            "dimmer_rms": donor_effect["dimmer_rms"],
            "angular_rms_deg_moving": donor_effect["angular_rms_deg_moving"],
        }
        songs.append({"target": target, "donor": donor, "frames": n40,
                      "rows": rows, "target_lights_sha256": sha(
                          Path(args.baseline_root).resolve() / target / "lights.npz"),
                      "patch_sha256": sha(Path(args.baseline_root).resolve() / target / "patch.json"),
                      "generation_raw_safety": generations[target]["raw_safety_gate"]})
    report = {"schema_version": 1, "status": "diagnostic_only",
              "qualification_status": "unqualified_cached_audio_donor",
              "songs": songs, "buses": {song: {"path": str(path), "sha256": sha(path)}
                                       for song, path in buses.items()},
              "checkpoint": {"path": str(args.checkpoint), "sha256": sha(args.checkpoint),
                             "metadata": metadata},
              "transforms_sha256": sha(args.transforms),
              "scope": "cached LightLayers donor audio -> fixed target body/mapper/P2 decoded DMX",
              "missing_evidence": ["authenticated raw waveform recomputation", "full-song phrase controls", "learned audio-to-pan/tilt path"]}
    (out / "PORTFOLIO_RECEIPT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shpongle-bus", required=True, type=Path)
    p.add_argument("--cure-bus", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--baseline-root", required=True, type=Path)
    p.add_argument("--transforms", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.songs = ["shpongle", "cure_for_me"]
    args.bus = {"shpongle": args.shpongle_bus, "cure_for_me": args.cure_bus}
    args.checkpoint = args.checkpoint.resolve(); args.transforms = args.transforms.resolve()
    report = run(args)
    print(json.dumps({"status": report["status"], "out": str(Path(args.out).resolve())}, sort_keys=True))


if __name__ == "__main__": main()
