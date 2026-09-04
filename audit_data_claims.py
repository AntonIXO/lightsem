"""Audit two handoff claims against the real corpus.

Claim 1: "2817 files, 884 MB" of keypoints. Observed 1408 .pkl files.
Claim 2: "Keypoints contain NaN frames." Observed nan_frac 0.0 everywhere
         when reading `keypoints3d_optim`.

Both matter: if NaN never occurs the interpolation policy is dead code, and
if the file count is half what was recorded something may be missing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

KP_DIR = Path("/root/lightsem/data/kp3d/keypoints3d")


def main() -> None:
    files = sorted(KP_DIR.glob("*.pkl"))
    all_entries = sorted(KP_DIR.iterdir())
    print(f"*.pkl files            : {len(files)}")
    print(f"all dir entries        : {len(all_entries)}")
    exts: dict[str, int] = {}
    for p in all_entries:
        exts[p.suffix] = exts.get(p.suffix, 0) + 1
    print(f"extensions             : {exts}")

    # NaN audit across BOTH stored keys, on a large sample.
    sample = files[::7]  # ~200 files, spread across the corpus
    stats = {
        "keypoints3d": {"n": 0, "with_nan": 0, "frac_sum": 0.0},
        "keypoints3d_optim": {"n": 0, "with_nan": 0, "frac_sum": 0.0},
    }
    keysets: dict[tuple, int] = {}
    for p in sample:
        with open(p, "rb") as f:
            d = pickle.load(f)
        kt = tuple(sorted(d.keys())) if isinstance(d, dict) else ("<raw>",)
        keysets[kt] = keysets.get(kt, 0) + 1
        for key in stats:
            if not isinstance(d, dict) or key not in d:
                continue
            arr = np.asarray(d[key], dtype=np.float64)
            bad = ~np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
            s = stats[key]
            s["n"] += 1
            s["frac_sum"] += float(bad.mean())
            if bad.any():
                s["with_nan"] += 1

    print(f"\nsampled {len(sample)} files")
    print(f"key sets observed      : {keysets}")
    for key, s in stats.items():
        if s["n"] == 0:
            print(f"  {key:20s} absent")
            continue
        print(
            f"  {key:20s} present in {s['n']:4d}  "
            f"files_with_nan={s['with_nan']:4d}  "
            f"mean_nan_frac={s['frac_sum'] / s['n']:.6f}"
        )


if __name__ == "__main__":
    main()
