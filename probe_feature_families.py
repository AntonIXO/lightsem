"""WHICH audio property predicts lighting events -- and is openl3 worth it?

Runs on every song fetched by fetch_interp_feats.py that has both the
interpretable descriptors and the existing 6-dim light targets (140 as of the
definitive run; an earlier 30-song pass was UNDERPOWERED and its all-null
result was a power problem, not an answer).

The question this settles for the project:

  If ENERGY alone carries the whole event signal, then the industry's L2
  threshold triggers are already at the ceiling on this data, and "L3 semantic
  understanding" has no headroom to claim. That is a project-defining negative.

  If SPECTRAL / HARMONIC add over energy, that names the specific features a
  semantic model must encode, and justifies building one.

  And openl3 (512-d, opaque, learned) is included as a family so we can see
  whether the learned embedding beats hand-picked descriptors at all. If it
  does not, the whole "learned encoder" premise loses its motivation.

Every family gets identical treatment: same causal multi-scale pooling + change
features, same rhythm baseline, same per-song CV, same circular-shift
permutation null. Reported as z against each family's OWN null, because
different feature dimensionalities have different overfitting biases and the
null absorbs that.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import numpy as np
from scipy.ndimage import uniform_filter1d

from lightsem.model.data import DECIM
from probe_event_target import (
    HORIZON,
    HZ,
    PAST,
    average_precision,
    event_train,
    hazard_feats,
    target,
)

K_FOLDS = 5
N_PERM = 25
RNG = np.random.default_rng(0)


def change_feats(a: np.ndarray) -> np.ndarray:
    """(T,D) -> causal multi-scale pooling + novelty. Same recipe as audio_feats."""
    z = (a - a.mean(0)) / (a.std(0) + 1e-6)
    pools = [uniform_filter1d(z, size=k, axis=0, origin=(k - 1) // 2, mode="nearest")
             for k in PAST]
    nov = [pools[0] - pools[i] for i in range(1, len(PAST))]
    return np.concatenate(pools + nov, 1).astype(np.float32)


def fit_ridge(X, y, lam=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Z.T @ (y - y.mean())), y.mean()


def apply_ridge(p, X):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


def load():
    d = np.load("data/cache/interp_feats.npz")
    keys = sorted({k.split("__")[0] for k in d.files})
    songs = []
    for k in keys:
        z = np.load(f"data/cache/rpmc_train/{k}.npz")
        y30, x30 = z["y"].astype(np.float32), z["x"].astype(np.float32)
        n = (len(y30) // DECIM) * DECIM
        y = y30[:n].reshape(-1, DECIM, y30.shape[1]).mean(1)
        x = x30[:n].reshape(-1, DECIM, x30.shape[1])[:, 0, :]
        fams = {f.split("__")[1]: d[f].T for f in d.files if f.startswith(k + "__")}
        T = min([len(y), len(x)] + [len(v) for v in fams.values()])
        if T < 200:
            continue
        fams = {a: v[:T] for a, v in fams.items()}
        fams["openl3"] = x[:T]
        songs.append((k, y[:T], fams))
    return songs


def cv_lift(songs, ev_trains, fam: str | None) -> float:
    built = []
    for (k, y, fams), ev in zip(songs, ev_trains):
        H = hazard_feats(ev)
        A = change_feats(fams[fam]) if fam else None
        built.append((H, A, target(ev)))
    folds = [[] for _ in range(K_FOLDS)]
    for i in range(len(songs)):
        folds[i % K_FOLDS].append(i)

    def X_of(i):
        H, A, _ = built[i]
        return H if A is None else np.concatenate([H, A], 1)

    lifts = []
    for f in range(K_FOLDS):
        te = folds[f]
        tr = [i for i in range(len(songs)) if i not in set(te)]
        Xtr = np.concatenate([X_of(i) for i in tr])
        Ytr = np.concatenate([built[i][2] for i in tr])
        p = fit_ridge(Xtr, Ytr)
        for i in te:
            Y = built[i][2]
            if Y.sum() < 3:
                continue
            lifts.append(average_precision(apply_ridge(p, X_of(i)), Y) / Y.mean())
    return float(np.mean(lifts))


def circ(ev, rng):
    k = int(rng.integers(len(ev) // 10, len(ev) - len(ev) // 10))
    return np.roll(ev, k)


def main():
    songs = load()
    print(f"songs with both descriptors and targets: {len(songs)}")
    fams = sorted(songs[0][2].keys())
    print(f"families: {fams}")
    print(f"dims: " + "  ".join(f"{a}:{songs[0][2][a].shape[1]}" for a in fams))
    ev = [event_train(y)[0] for _, y, _ in songs]
    print(f"events {sum(int(e.sum()) for e in ev)}  horizon {HORIZON/HZ:.1f}s")

    base = cv_lift(songs, ev, None)
    print(f"\nrhythm-only baseline (no audio): {base:.4f}x")

    out = {"n_songs": len(songs), "n_events": int(sum(int(e.sum()) for e in ev)),
           "horizon_s": HORIZON / HZ, "n_perm": N_PERM,
           "written_at": datetime.now().isoformat(timespec="seconds"),
           "rhythm_baseline": base, "families": {}}
    print(f"\n{'family':>10s} {'dim':>5s} {'lift':>8s} {'marg':>8s} "
          f"{'null':>16s} {'z':>7s} {'p':>6s}")
    for fam in fams:
        obs = cv_lift(songs, ev, fam)
        nulls = []
        for _ in range(N_PERM):
            perm = [circ(e, RNG) for e in ev]
            nulls.append(cv_lift(songs, perm, fam))
        nu = np.array(nulls)
        z = (obs - nu.mean()) / (nu.std() + 1e-9)
        p = float((nu >= obs).mean())
        out["families"][fam] = {"lift": obs, "marginal": obs - base,
                                "null_mean": float(nu.mean()),
                                "null_sd": float(nu.std()),
                                "z": float(z), "p": p,
                                "dim": int(songs[0][2][fam].shape[1])}
        print(f"{fam:>10s} {songs[0][2][fam].shape[1]:>5d} {obs:>8.4f} "
              f"{obs-base:>+8.4f} {nu.mean():>8.4f}+-{nu.std():<6.4f} "
              f"{z:>+7.2f} {p:>6.3f}")

    print("\n=== verdict ===")
    sig = {k: v for k, v in out["families"].items() if v["z"] > 2.0}
    if not sig:
        print("  No family beats its own permutation null at z>2.")
        print(f"  -> not supported on these {len(songs)} songs. NOTE: with "
              f"{out['n_events']} events this may be a POWER problem rather")
        print("     than an answer -- check null sd before concluding.")
    else:
        best = max(sig.items(), key=lambda t: t[1]["z"])
        print(f"  significant: {sorted(sig, key=lambda k: -sig[k]['z'])}")
        print(f"  strongest: {best[0]} (z={best[1]['z']:+.2f})")
        e = out["families"].get("energy", {}).get("z", float("-nan"))
        print(f"  energy alone z={e:+.2f}")
        nonenergy = [k for k in sig if k not in ("energy", "openl3")]
        if nonenergy:
            print(f"  BEYOND energy: {nonenergy} -> semantic features add;")
            print("     an L3 model has headroom over threshold triggers.")
        else:
            print("  only energy/openl3 significant -> consistent with L2")
            print("     threshold triggers already being near the ceiling.")

    # Write to a run-specific path AND the canonical one. Previously both the
    # 30-song and 140-song runs wrote only `family_attribution.json`, so a
    # late-finishing small run could silently clobber the definitive large one.
    stamped = f"data/cache/family_attribution_n{len(songs)}.json"
    json.dump(out, open(stamped, "w"), indent=2)
    json.dump(out, open("data/cache/family_attribution.json", "w"), indent=2)
    print(f"\nwrote {stamped} and data/cache/family_attribution.json")
    print(f"  n_songs={len(songs)} n_events={out['n_events']} "
          f"written_at={out['written_at']}")


if __name__ == "__main__":
    main()
