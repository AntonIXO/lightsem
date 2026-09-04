#!/usr/bin/env python
"""Extract all 699 RPMC-L2 songs from the local HDF5 into the Stage-2 cache.

Same output format as `extract_rpmc.py` (which read over HTTP ranges when the
disk could not hold the corpus), so the August event probe and its baselines
run unchanged on 5x the data. Writes to /mnt because the root disk is full.
"""
from __future__ import annotations

import json
import os
import sys
import time

import h5py
import numpy as np

from extract_rpmc import reduce_light

H5 = "/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5"
OUT = "/mnt/lightsem-data/cache/rpmc_all"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    t0 = time.time()
    f = h5py.File(H5, "r")
    keys = list(f.keys())[:limit]
    print(f"{len(keys)} songs", flush=True)
    ok, fail, manifest = 0, 0, []
    for i, k in enumerate(keys):
        dst = os.path.join(OUT, f"{k[:16]}.npz")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            g = f[k]
            emb = g["music"]["openl3"][:].astype(np.float32)
            y = reduce_light(g["light"]["0"][:])
            Lm = emb.shape[1]
            idx = np.linspace(0, Lm - 1, y.shape[0])
            lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
            w = (idx - lo).astype(np.float32)
            x = (emb[:, lo] * (1 - w) + emb[:, hi] * w).T
            np.savez_compressed(dst, x=x.astype(np.float16), y=y.astype(np.float16))
            manifest.append({"key": k, "frames": int(y.shape[0])})
            ok += 1
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  FAIL {k[:16]}: {type(e).__name__} {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(keys)}  ok={ok} fail={fail}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"ok": ok, "fail": fail, "songs": manifest},
              open("/mnt/lightsem-data/cache/rpmc_all_manifest.json", "w"), indent=1)
    print(f"DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s -> {OUT}")
    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
