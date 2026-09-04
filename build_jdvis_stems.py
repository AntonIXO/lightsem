#!/usr/bin/env python
"""H-G neural arm: the JD visual operator-head cache with the per-stem 10 Hz features appended,
so train_operator_head.py runs unchanged on it (x = 52-d bus features + 195-d stem features).
  bash run.sh build_jdvis_stems.py
"""
import numpy as np, json
from pathlib import Path
C = Path("/mnt/lightsem-data/cache"); out = C / "justdance_visual_stems"; out.mkdir(exist_ok=True)
n = 0
for p in sorted((C / "justdance_visual").glob("*.npz")):
    sp = C / "justdance_stem_feats" / p.name
    if not sp.exists():
        continue
    v = np.load(p, allow_pickle=False); s = np.load(sp, allow_pickle=False)
    T = len(v["x"]); x10 = s["x10"][:T]
    if len(x10) < T:
        x10 = np.concatenate([x10, np.repeat(x10[-1:], T - len(x10), 0)])
    np.savez_compressed(out / p.name, x=np.concatenate([v["x"], x10], 1).astype(np.float32), y=v["y"],
                        names=np.concatenate([v["names"], s["names10"]]), bpm=v["bpm"], meta=v["meta"])
    n += 1
print(f"{n} songs -> {out}")
