#!/usr/bin/env python3
"""Compare pre/post-adaptation feature arrays without overwriting reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightsem.experiments.foundation_audio_light.retention import compare_features


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        keys = [key for key in data.files if key == "muq" or key.startswith("muq_layer_")]
        if not keys:
            raise ValueError(f"{path} contains no MuQ arrays")
        return {key: np.asarray(data[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    reference, candidate = _load(args.reference), _load(args.candidate)
    if set(reference) != set(candidate):
        raise SystemExit("reference and candidate layer keys differ")
    report = {
        "schema": "lightsem.muq_retention.v1",
        "protocol_id": "muq-retention-20261019-v1",
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "layers": {key: compare_features(reference[key], candidate[key]) for key in sorted(reference)},
        "promotion": "diagnostic_only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "layers": sorted(reference)}, sort_keys=True))


if __name__ == "__main__":
    main()

