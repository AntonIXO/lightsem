#!/usr/bin/env python3
"""Encode cached StereoAudioV1 mixtures into source-aware native context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import StereoAudioV1, resample_stream, sha256
from lightsem.experiments.audio_stereo.semantic import SemanticStereoConditioner, SemanticStereoConditionerConfig


def write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing semantic context manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


def encode(manifest_path: Path, checkpoint: Path, out: Path, device: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_access", True):
        raise ValueError("Semantic context encoding requires test_access=false")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SemanticStereoConditioner(SemanticStereoConditionerConfig(**{
        key: saved["config"][key] for key in ("arm", "input_dim", "source_event_dim",
                                                "source_spatial_dim", "source_relation_dim",
                                                "width", "context_dim", "spatial_hop", "relation_hop")
    })).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    stats = {key: torch.from_numpy(np.asarray(value)).to(device) for key, value in saved["stats"].items()}
    rows = []
    for row in manifest["entries"]:
        descriptor = StereoAudioV1.load(row["feature_path"] if "feature_path" in row else row["mixture_feature_path"])
        x = torch.from_numpy(descriptor.features_40[None]).to(device)
        x = (x - stats["input_mean"]) / stats["input_std"]
        with torch.inference_mode():
            outputs = model(x)
        fused40 = outputs["fused"][0].float().cpu().numpy()
        relation2 = outputs["relation"][0].float().cpu().numpy()
        time40 = descriptor.time_40
        time2 = descriptor.time_2[:len(relation2)]
        time30 = np.arange(int(np.ceil(float(time40[-1]) * 30)) + 1, dtype=np.float64) / 30.
        fused30 = resample_stream(fused40, time40, time30)
        destination = out / "contexts" / "semantic_factorized" / (Path(descriptor.metadata.get("id", row["id"])).stem + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("xb") as handle:
            np.savez_compressed(handle, time30=time30, fused30=fused30.astype(np.float32),
                                time2=time2.astype(np.float64), relation2=relation2.astype(np.float32))
        rows.append({"id": row["id"], "split": row.get("split", "train"),
                     "feature_path": row.get("feature_path", row.get("mixture_feature_path")),
                     "feature_sha256": sha256(row.get("feature_path", row.get("mixture_feature_path"))),
                     "context_path": str(destination), "context_sha256": sha256(destination),
                     "frames30": len(time30), "frames2": len(time2)})
    result = {"schema_version": 1, "status": "complete", "test_access": False,
              "source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "model_arm": "semantic_factorized", "context_dim": 64, "rows": rows}
    write(out / "STEREO_CONTEXT_MANIFEST_semantic_factorized.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = encode(args.manifest, args.checkpoint, args.out, args.device)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "arm": result["model_arm"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
