#!/usr/bin/env bash
cd /root/lightsem
export PATH="/root/lightsem/.venv/bin:$PATH"
exec python - <<'PY'
import pickle, glob, os, numpy as np
pk = sorted(glob.glob("data/kp3d/keypoints3d/*.pkl"))
wv = sorted(glob.glob("data/wav/wav/*.wav"))
print("pkl:", len(pk), "wav:", len(wv))
print("wav names:", [os.path.basename(w) for w in wv[:8]])
with open(pk[0],"rb") as f: d = pickle.load(f)
print("pkl0:", os.path.basename(pk[0]), type(d))
if isinstance(d, dict):
    for k,v in d.items():
        print("   ", k, type(v), getattr(v,"shape",None), getattr(v,"dtype",None))
PY
