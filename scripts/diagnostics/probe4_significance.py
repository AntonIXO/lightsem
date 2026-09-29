#!/usr/bin/env python
"""
PROBE 4: is the segmentation-first SAME>DIFF gap statistically real?

probe3b found, at BEAT-SYNCHRONOUS resolution:
    SAME choreo r = +0.131 +-0.320 (n=136)
    DIFF choreo r = -0.007 +-0.277 (n=400)
and structural boundary F1 0.462 vs null 0.352, ceiling 0.500.

probe2, with FIXED WINDOWS, found same/diff indistinguishable (0.366 vs 0.354).
So the difference must be tested properly, not eyeballed, because:
  - pairs are NOT independent (same dancer/track appears in many pairs)
  - a plain t-test would badly overstate significance

Tests:
  S1. Cluster-aware permutation: shuffle the SAME/DIFF label WITHIN each music
      track, recompute the mean gap, 10000 times. Preserves per-track structure.
  S2. Bootstrap over TRACKS (the independent unit), not pairs -> CI on the gap.
  S3. Boundary-F1 fraction-of-ceiling with a track-level bootstrap CI.
  S4. Effect size (Cohen's d) and the practical question: what does r=0.13 buy
      you for a light cue compared to the audio envelope baseline?
"""
import json
from collections import defaultdict

import numpy as np

RES = "/root/lightsem/runs/probes/probe4_input.json"


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 1e-12 else np.nan


def main():
    with open(RES) as f:
        d = json.load(f)

    # ---------- S1: cluster-aware permutation, restricted within track --------
    by_track = defaultdict(lambda: {"same": [], "diff": []})
    for rec in d["pairs"]:
        by_track[rec["music"]][rec["kind"]].append(rec["r"])

    same_all = np.array([r for t in by_track.values() for r in t["same"]])
    diff_all = np.array([r for t in by_track.values() for r in t["diff"]])
    obs = same_all.mean() - diff_all.mean()

    rng = np.random.default_rng(0)
    n_perm = 10000
    null = np.empty(n_perm)
    for i in range(n_perm):
        s_acc, d_acc = [], []
        for t in by_track.values():
            pool = np.array(t["same"] + t["diff"])
            ns = len(t["same"])
            if ns == 0 or len(pool) == ns:
                continue
            idx = rng.permutation(len(pool))
            s_acc.append(pool[idx[:ns]].mean())
            d_acc.append(pool[idx[ns:]].mean())
        null[i] = np.mean(s_acc) - np.mean(d_acc)
    p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)

    print("=" * 74)
    print("S1  CLUSTER-AWARE PERMUTATION (labels shuffled within each track)")
    print("=" * 74)
    print(f"  observed gap (SAME - DIFF) : {obs:+.4f}")
    print(f"  permutation null           : {null.mean():+.4f} +-{null.std():.4f}")
    print(f"  two-sided p                : {p_perm:.4f}   "
          f"{'SIGNIFICANT' if p_perm < 0.05 else 'NOT significant'}")

    # ---------- S2: bootstrap over TRACKS -------------------------------------
    tracks = list(by_track)
    boots = []
    for _ in range(10000):
        pick = rng.choice(len(tracks), len(tracks), replace=True)
        s, dd = [], []
        for k in pick:
            t = by_track[tracks[k]]
            if t["same"]:
                s += t["same"]
            if t["diff"]:
                dd += t["diff"]
        if s and dd:
            boots.append(np.mean(s) - np.mean(dd))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print()
    print("=" * 74)
    print(f"S2  BOOTSTRAP over TRACKS (n={len(tracks)} independent units)")
    print("=" * 74)
    print(f"  gap 95% CI : [{lo:+.4f}, {hi:+.4f}]   "
          f"{'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")
    print(f"  Cohen's d  : {cohens_d(same_all, diff_all):+.3f}")

    # ---------- S3: boundary F1 vs ceiling ------------------------------------
    bf = np.array(d["boundary_f1"])
    bn = np.array(d["boundary_null"])
    bc = np.array(d["boundary_ceiling"])
    bt_track = np.array(d["boundary_track"])
    print()
    print("=" * 74)
    print("S3  STRUCTURAL BOUNDARY: fraction of the inter-dancer ceiling")
    print("=" * 74)
    frac = (bf.mean() - bn.mean()) / (bc.mean() - bn.mean())
    print(f"  pose->audio boundary F1 : {bf.mean():.3f}")
    print(f"  random null             : {bn.mean():.3f}")
    print(f"  inter-dancer ceiling    : {bc.mean():.3f}")
    print(f"  fraction of ceiling     : {frac:.1%}")
    fb = []
    ut = sorted(set(bt_track))
    for _ in range(10000):
        pick = rng.choice(len(ut), len(ut), replace=True)
        sel = np.concatenate([np.where(bt_track == ut[k])[0] for k in pick])
        fb.append(bf[sel].mean() - bn[sel].mean())
    fb = np.array(fb)
    lo2, hi2 = np.percentile(fb, [2.5, 97.5])
    print(f"  lift over null 95% CI   : [{lo2:+.3f}, {hi2:+.3f}]   "
          f"{'excludes 0' if lo2 > 0 else 'INCLUDES 0'}")

    # ---------- S4: what does it buy in practice ------------------------------
    print()
    print("=" * 74)
    print("S4  PRACTICAL READ")
    print("=" * 74)
    r_same = same_all.mean()
    print(f"  beat-sync inter-dancer r = {r_same:+.3f} -> shared variance "
          f"R^2 = {r_same ** 2:.4f} ({r_same ** 2:.2%})")
    print(f"  i.e. even a PERFECT audio->pose model explains at most "
          f"{r_same ** 2:.2%} of another dancer's beat-level energy.")
    print(f"  audio envelope baseline r(KE, RMS) = {d['r_rms']:+.3f} -> "
          f"R^2 = {d['r_rms'] ** 2:.4f}")
    print(f"  audio->KE ridge on held-out track  R^2 = {d['ridge_r2']:+.4f} "
          "(negative = worse than predicting the mean)")


if __name__ == "__main__":
    main()
