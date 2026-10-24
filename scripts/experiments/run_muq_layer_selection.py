#!/usr/bin/env python3
"""Run the frozen MuQ/Base92 layer-selection probe and write one receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightsem.experiments.foundation_audio_light.layer_selection import (
    compare_feature_families,
    load_probe_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    families, target_time, target, groups, meta = load_probe_npz(str(args.features))
    if not families:
        raise SystemExit("feature file contains no Base92 or MuQ layers")
    report = compare_feature_families(
        families,
        target_time=target_time,
        target=target,
        groups=groups,
        pca_components=args.pca_components,
        folds=args.folds,
    )
    report.update({
        "schema": "lightsem.muq_layer_selection.v1",
        "protocol_id": "muq-layer-selection-20261019-v1",
        "input_features": str(args.features.resolve()),
        "metadata": meta,
        "groups_provided": groups is not None,
        "promotion": "forbidden: layer probe only",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "families": list(report["families"])}, sort_keys=True))


if __name__ == "__main__":
    main()

