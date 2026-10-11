#!/usr/bin/env python3
"""Package the authenticated dual-attention run as a native-C baseline asset.

The training driver intentionally writes a compact research checkpoint. The
production loader also requires the incumbent native feature names and identity
receipt, so this script adds only those provenance fields and verifies the
dual checkpoint against the selected RTMPose baseline before writing a new
external artifact. It never changes model tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dual-checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    dual = torch.load(args.dual_checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if dual.get("baseline_sha256") != sha(args.base_checkpoint):
        raise ValueError("Dual checkpoint is not bound to the selected native baseline")
    cfg = dual.get("config", {})
    if cfg.get("arm") != "joint" or cfg.get("extra_dim") != 1061 or cfg.get("steps") != 32:
        raise ValueError("Dual checkpoint violates native-C contract")
    if cfg.get("stereo_fusion") != "dual_cross_attention" or cfg.get("stereo_dim") != 128:
        raise ValueError("Expected dual cross-attention 128-D checkpoint")
    if not isinstance(base.get("feature_names"), list) or len(base["feature_names"]) != 1061:
        raise ValueError("Selected baseline has no authenticated feature-name contract")
    identity = {
        "protocol_sha256": sha(args.protocol),
        "arm": "joint",
        "seed": int(dual.get("seed", 0)),
        "initial": {"checkpoint": str(args.base_checkpoint), "sha256": sha(args.base_checkpoint)},
        "stereo_training": {
            "fusion": "dual_cross_attention",
            "stereo_scale": dual.get("stereo_scale"),
            "counterfactual_weight": dual.get("counterfactual_weight"),
            "counterfactual_margin": dual.get("counterfactual_margin"),
            "context_manifest": dual.get("context_manifest"),
            "context_manifest_sha256": dual.get("context_manifest_sha256"),
        },
    }
    packaged = dict(dual)
    packaged.update({
        "identity": identity,
        "feature_names": list(base["feature_names"]),
        "quiet_gradient_fix_in_checkpoint": base.get("quiet_gradient_fix_in_checkpoint", False),
        "production_baseline": True,
        "production_baseline_config": "configs/dance/fixed_c.json",
        "packaged_from": str(args.dual_checkpoint),
        "packaged_from_sha256": sha(args.dual_checkpoint),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, args.out)
    receipt = {
        "status": "complete",
        "checkpoint": str(args.out),
        "checkpoint_sha256": sha(args.out),
        "source_dual_checkpoint": str(args.dual_checkpoint),
        "source_dual_checkpoint_sha256": sha(args.dual_checkpoint),
        "selected_motion_baseline": str(args.base_checkpoint),
        "selected_motion_baseline_sha256": sha(args.base_checkpoint),
        "protocol": str(args.protocol),
        "protocol_sha256": sha(args.protocol),
        "model_tensors_unchanged": True,
        "fusion": "dual_cross_attention",
        "stereo_dim": 128,
        "scope": "production baseline packaging; no new training or promotion claim",
    }
    receipt_path = args.out.with_name("PRODUCTION_PACKAGE.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.with_name(receipt_path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(receipt_path)}) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
