"""Extract a compact RPMC-L2 training set over HTTP range reads.

Full dataset is 42.5 GB and this box has 1.8 GB free, so we never download it.
Per song we pull only openl3 (512,L') + the un-rotated light track, reduce the
light histograms to a low-dim scene vector, and store float16.

Light target per frame (6 dims):
  hue_cos, hue_sin  -- circular dominant hue (hue is periodic; raw bin index
                       would make 179->0 look like a huge jump)
  sat_mean, val_mean
  val_p90           -- highlight level, separates "bright wash" from "spotlight"
  hue_conc          -- 1 - normalised entropy of hue hist: colour focus vs wash
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from lightsem.data.rpmc_remote import open_h5

OUT = "data/cache/rpmc_train"
N_SONGS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def reduce_light(L: np.ndarray) -> np.ndarray:
    """(F,3,256) HSV histograms -> (F,6) scene vector."""
    h = L[:, 0, :180].astype(np.float64)
    s = L[:, 1, :].astype(np.float64)
    v = L[:, 2, :].astype(np.float64)
    bins = np.arange(256, dtype=np.float64)

    hs = np.maximum(h.sum(1, keepdims=True), 1.0)
    hp = h / hs
    ang = np.arange(180) * (2 * np.pi / 180.0)
    hue_cos = (hp * np.cos(ang)).sum(1)
    hue_sin = (hp * np.sin(ang)).sum(1)
    ent = -(hp * np.log(hp + 1e-12)).sum(1) / np.log(180.0)
    hue_conc = 1.0 - ent

    sat = (s * bins).sum(1) / np.maximum(s.sum(1), 1.0) / 255.0
    val = (v * bins).sum(1) / np.maximum(v.sum(1), 1.0) / 255.0
    vc = np.cumsum(v, axis=1) / np.maximum(v.sum(1, keepdims=True), 1.0)
    val_p90 = np.argmax(vc >= 0.90, axis=1) / 255.0

    return np.stack([hue_cos, hue_sin, sat, val, val_p90, hue_conc], 1).astype(np.float32)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    f, raw = open_h5(cache_dir=None)  # no block cache: we read big contiguous slabs
    keys = list(f.keys())
    print(f"{len(keys)} songs available; extracting {N_SONGS}", flush=True)

    manifest, ok, fail = [], 0, 0
    for i, k in enumerate(keys[:N_SONGS]):
        dst = os.path.join(OUT, f"{k[:16]}.npz")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            g = f[k]
            emb = g["music"]["openl3"][:].astype(np.float32)      # (512, Lm)
            lig = g["light"]["0"][:]                              # (F,3,256)
            y = reduce_light(lig)                                 # (F,6)

            # Resample audio embedding onto the light frame grid.
            Lm = emb.shape[1]
            idx = np.linspace(0, Lm - 1, y.shape[0])
            lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
            w = (idx - lo).astype(np.float32)
            x = (emb[:, lo] * (1 - w) + emb[:, hi] * w).T          # (F,512)

            np.savez_compressed(dst, x=x.astype(np.float16),
                                y=y.astype(np.float16))
            ok += 1
            manifest.append({"key": k, "frames": int(y.shape[0])})
            if ok % 5 == 0:
                mb = raw.bytes_fetched / 1e6
                el = time.time() - t0
                print(f"  {ok}/{N_SONGS}  {mb:.0f} MB net  {el:.0f}s  "
                      f"({mb/max(el,1):.1f} MB/s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {k[:16]}: {type(e).__name__} {e}", flush=True)

    with open("data/cache/rpmc_train_manifest.json", "w") as fh:
        json.dump({"ok": ok, "fail": fail, "songs": manifest}, fh, indent=2)
    du = sum(os.path.getsize(os.path.join(OUT, p)) for p in os.listdir(OUT))
    print(f"\nDONE ok={ok} fail={fail}  local={du/1e6:.0f} MB  "
          f"net={raw.bytes_fetched/1e6:.0f} MB  {time.time()-t0:.0f}s")
    f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
