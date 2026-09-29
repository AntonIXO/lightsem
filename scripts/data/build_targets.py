#!/usr/bin/env python
"""Fit the corpus scalers the conductor needs.

    bash run.sh scripts/data/build_targets.py scalers
"""
from __future__ import annotations

import json
import sys

import numpy as np

from lightsem.conductor.scalers import fit_scalers, save_scalers
from lightsem.motion.io import CACHE_ROOT, load_descriptors


def main(limit: int = 400) -> None:
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    rng = np.random.default_rng(0)
    pick = [idx[i] for i in rng.permutation(len(idx))[:limit]]
    acc: dict[str, list] = {}
    for r in pick:
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        for k, v in d.channels.items():
            acc.setdefault(k, []).append(v[::7])          # thin: frames correlate
    samples = {k: np.concatenate(v) for k, v in acc.items()}
    sc = fit_scalers(samples)
    p = save_scalers(sc)
    print(f"fitted {len(sc)} channel scalers from {len(pick)} clips -> {p}")
    for k in sorted(sc):
        print(f"  {k:18s} median {sc[k]['median']:9.3f}  IQR {sc[k]['iqr']:9.3f}")


if __name__ == "__main__":
    main(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
