"""Re-apply the FIXED verdict rule to the saved horizon sweep.

The sweep costs ~45 min. The bug was only in the interpretation, so replay the
stored numbers through the corrected logic and confirm it now says REACTIVE
where the old z-only rule said ANTICIPATORY.
"""
import json

import numpy as np

d = json.load(open("data/cache/horizon_sweep.json"))
R = d["rows"]
ts = d["horizons_s"]
keys = [f"{t:.1f}s" for t in ts]
zs = [R[k]["z"] for k in keys]
ms = [R[k]["marginal"] for k in keys]

print(f"{'horizon':>8s} {'marginal':>10s} {'z':>7s}")
for t, m, z in zip(ts, ms, zs):
    print(f"{t:>7.1f}s {m:>+10.4f} {z:>+7.2f}")

print("\n--- OLD RULE (z only) ---")
far_z = [z for t, z in zip(ts, zs) if t >= 4.0]
print("far z:", ["%+.2f" % v for v in far_z],
      "-> ANTICIPATORY" if max(far_z) > 2.0 else "-> not")

print("\n--- NEW RULE (material effect size AND z) ---")
MIN_LIFT = 0.02
# Classify by the LARGEST horizon that still carries a material effect, rather
# than by two hand-drawn bands. The earlier near<=0.5 / far>=4.0 split left
# 1.0s and 2.0s in NEITHER band: both pass the gate, but a sweep whose only
# passing horizons were 1-2s would have been mislabelled "fragile".
passing = [t for t, m, z in zip(ts, ms, zs) if m > MIN_LIFT and z > 2.0]
far_ok = [t for t in passing if t >= 4.0]
near_ok = [t for t in passing if t <= 0.5]
if far_ok:
    verdict = "anticipatory"          # material lift seconds ahead -> rig can lead
elif passing:
    verdict = "reactive"              # material lift only at short range
else:
    verdict = "fragile"               # nothing survives the effect-size floor
print(f"all horizons passing lift>{MIN_LIFT} and z>2: {passing}")
print(f"  of which far (>=4.0s):  {far_ok}")
print(f"  of which near (<=0.5s): {near_ok}")
print(f"  mid band (1.0-2.0s):    {[t for t in passing if 0.5 < t < 4.0]}")
print(f"peak marginal lift at {ts[int(np.argmax(ms))]:.1f}s")
print(f"VERDICT: {verdict.upper()}")

assert verdict == "reactive", f"expected reactive, got {verdict}"
print("\nOK: corrected rule reverses the original ANTICIPATORY claim.")
d["verdict"] = verdict
d["peak_marginal_horizon_s"] = float(ts[int(np.argmax(ms))])
json.dump(d, open("data/cache/horizon_sweep.json", "w"), indent=2)
print("updated data/cache/horizon_sweep.json with corrected verdict")
