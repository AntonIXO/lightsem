#!/usr/bin/env python3
"""Audit StereoAudioV1 caches and optional conditioner outputs.

This is deliberately an evidence collector.  It does not turn descriptor
activity into a choreography claim and refuses to declare the scientific gate
without the paired comparison metrics supplied by a locked evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from lightsem.audio.stereo_v1 import StereoAudioV1, metamorphic_check, sha256
from lightsem.experiments.audio_stereo.diagnostics import feature_diagnostics, representation_gate


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


def run(manifest_path: Path, out: Path, checkpoint: Path | None = None, split: str = "dev") -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_access", True):
        raise ValueError("Stereo audit requires test_access=false")
    rows = [row for row in manifest.get("entries", []) if row.get("split") in (split, "val" if split == "dev" else split)]
    if not rows:
        raise ValueError(f"No rows for split {split!r}")
    model = None
    if checkpoint is not None:
        import torch
        from lightsem.experiments.audio_stereo.model import StereoConditioner, StereoConditionerConfig
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = StereoConditioner(StereoConditionerConfig(**{
            key: saved["config"][key] for key in ("arm", "input_dim", "event_dim", "spatial_dim", "relation_dim", "width", "fused_dim", "spatial_hop", "relation_hop")
        }))
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
    reports = []
    for row in rows:
        descriptor = StereoAudioV1.load(row["feature_path"])
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        metamorphic = metamorphic_check(audio.T, int(sr))
        diagnostics = feature_diagnostics(descriptor.features_40, descriptor.events_40,
                                          descriptor.spatial_10, descriptor.relation_2)
        row_report = {"id": row["id"], "split": row["split"], "metamorphic": metamorphic,
                      "diagnostics": diagnostics, "audio_sha256": sha256(row["audio_path"]),
                      "feature_sha256": sha256(row["feature_path"])}
        if model is not None:
            import torch
            with torch.inference_mode():
                outputs = model(torch.from_numpy(descriptor.features_40[None]))
            row_report["model_outputs"] = {key: {"shape": list(value.shape),
                                                   "finite": bool(torch.isfinite(value).all()),
                                                   "std": float(value.float().std())}
                                           for key, value in outputs.items()}
        reports.append(row_report)
    aggregate = {
        "finite": all(row["diagnostics"]["finite"] for row in reports),
        "noncollapse": all(row["diagnostics"]["noncollapse"] for row in reports),
        "lr_swap_pass": all(row["metamorphic"]["passes_lr_swap"] for row in reports),
    }
    gate_metrics = {**aggregate}
    gate = representation_gate(gate_metrics)
    result = {"schema_version": 1, "status": "complete", "manifest": str(manifest_path),
              "manifest_sha256": sha256(manifest_path), "split": split,
              "checkpoint": None if checkpoint is None else str(checkpoint),
              "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
              "test_access": False, "rows": reports, "aggregate": aggregate,
              "representation_gate": gate,
              "scope": "descriptor and model audit; no source truth or choreography claim"}
    _write(out, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args(argv)
    result = run(args.manifest, args.out, args.checkpoint, args.split)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]),
                      "gate": result["representation_gate"]["pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
