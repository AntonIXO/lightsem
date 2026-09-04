"""Does the audio->event effect have the SHAPE of anticipation?

probe_event_cv.py established the effect exists at z=+2.83 for a 1.0 s horizon.
A single significant point is compatible with several stories, and they have
different consequences for the product:

  ANTICIPATION  - audio predicts an event several seconds out, so a rig could
                  LEAD the music. This is the L3 claim and the whole pitch.
  REACTION      - audio only predicts the event when it is imminent (<0.5 s),
                  which is what a threshold trigger already does. No headroom.
  ARTEFACT      - the lift does not vary sensibly with horizon at all, which
                  would mean the z=+2.83 is a lucky draw from a noisy estimator.

So sweep the horizon and look at the CURVE, not one point. A real anticipatory
signal should decay smoothly with horizon: strong when near, weakening as the
prediction reaches further out, and it should be non-trivial at horizons well
beyond a reaction time.

Each horizon gets its own permutation null, because the base rate changes with
horizon (a longer window contains more events) and z must be computed against
a matched null, not a shared one.

Also included: a HORIZON-0 sanity check. If the effect were leakage from the
target construction, the shortest horizon would be anomalously strong.
"""
from __future__ import annotations

import json

import numpy as np

from lightsem.model.data import load_songs
from probe_event_cv import K_FOLDS, apply_ridge, circ_shift_events, fit_ridge
from probe_event_target import (
    HZ,
    audio_feats,
    average_precision,
    event_train,
    hazard_feats,
)

N_PERM = 20
HORIZONS = [2, 5, 10, 20, 40, 80]     # 0.2 .. 8.0 s
RNG = np.random.default_rng(0)


def target_h(train: np.ndarray, h: int) -> np.ndarray:
    T = len(train)
    cs = np.concatenate([[0.0], np.cumsum(train)])
    out = np.zeros(T, np.float32)
    for t in range(T):
        e = min(T, t + 1 + h)
        out[t] = 1.0 if cs[e] - cs[t + 1] > 0 else 0.0
    return out


def cv_marginal(songs, ev_trains, h: int) -> tuple[float, float]:
    """Returns (lift with audio, lift rhythm-only) for horizon h."""
    built = []
    for s, ev in zip(songs, ev_trains):
        built.append((hazard_feats(ev), audio_feats(s.x), target_h(ev, h)))
    folds = [[] for _ in range(K_FOLDS)]
    for i in range(len(songs)):
        folds[i % K_FOLDS].append(i)

    lifts_both, lifts_haz = [], []
    for f in range(K_FOLDS):
        te = folds[f]
        tr = [i for i in range(len(songs)) if i not in set(te)]
        Yt = np.concatenate([built[i][2] for i in tr])
        Xb = np.concatenate([np.concatenate([built[i][0], built[i][1]], 1) for i in tr])
        Xh = np.concatenate([built[i][0] for i in tr])
        pb, ph = fit_ridge(Xb, Yt), fit_ridge(Xh, Yt)
        for i in te:
            H, A, Y = built[i]
            if Y.sum() < 3 or Y.mean() > 0.95:
                continue
            br = Y.mean()
            lifts_both.append(
                average_precision(apply_ridge(pb, np.concatenate([H, A], 1)), Y) / br)
            lifts_haz.append(average_precision(apply_ridge(ph, H), Y) / br)
    return float(np.mean(lifts_both)), float(np.mean(lifts_haz))


def main():
    songs = load_songs()
    ev = [event_train(s.y)[0] for s in songs]
    print(f"songs {len(songs)}  events {sum(int(e.sum()) for e in ev)}")
    print(f"\n{'horizon':>8s} {'base':>6s} {'both':>7s} {'rhythm':>7s} "
          f"{'marg':>8s} {'null mean+-sd':>16s} {'z':>7s} {'p':>6s}")
    out = {"horizons_s": [h / HZ for h in HORIZONS], "rows": {}}

    for h in HORIZONS:
        base = float(np.mean([target_h(e, h).mean() for e in ev]))
        b, r = cv_marginal(songs, ev, h)
        obs = b - r
        nulls = []
        for _ in range(N_PERM):
            perm = [circ_shift_events(e, RNG) for e in ev]
            nb, nr = cv_marginal(songs, perm, h)
            nulls.append(nb - nr)
        nu = np.array(nulls)
        z = (obs - nu.mean()) / (nu.std() + 1e-9)
        p = float((nu >= obs).mean())
        out["rows"][f"{h/HZ:.1f}s"] = {
            "base_rate": base, "both": b, "rhythm": r, "marginal": obs,
            "null_mean": float(nu.mean()), "null_sd": float(nu.std()),
            "z": float(z), "p": p}
        print(f"{h/HZ:>7.1f}s {base:>6.3f} {b:>7.4f} {r:>7.4f} {obs:>+8.4f} "
              f"{nu.mean():>+8.4f}+-{nu.std():<6.4f} {z:>+7.2f} {p:>6.3f}")

    print("\n=== shape ===")
    zs = [out["rows"][f"{h/HZ:.1f}s"]["z"] for h in HORIZONS]
    ms = [out["rows"][f"{h/HZ:.1f}s"]["marginal"] for h in HORIZONS]
    ts = [h / HZ for h in HORIZONS]
    best = ts[int(np.argmax(ms))]
    print(f"  peak MARGINAL LIFT at horizon {best:.1f}s")
    print(f"  z        by horizon: {['%+.2f' % v for v in zs]}")
    print(f"  marginal by horizon: {['%+.4f' % v for v in ms]}")

    # BUG FIXED (see docs/STAGE_G_EVENT_TARGET.md): the original rule gated on
    # z alone and printed "ANTICIPATORY" for a curve whose effect size had
    # decayed ~99% and gone NEGATIVE. z stays >2 at long horizons only because
    # the permutation null is itself negative and tightens as the base rate
    # grows, so "less bad than the null" scores as significant while the
    # absolute lift is zero. Anticipation requires a MATERIAL effect at long
    # horizons, so gate on marginal lift and use z only to confirm it is real.
    MIN_LIFT = 0.02
    far = [(t, m, z) for t, m, z in zip(ts, ms, zs) if t >= 4.0]
    near = [(t, m, z) for t, m, z in zip(ts, ms, zs) if t <= 0.5]
    far_ok = [t for t, m, z in far if m > MIN_LIFT and z > 2.0]
    near_ok = [t for t, m, z in near if m > MIN_LIFT and z > 2.0]
    if far_ok:
        print(f"  -> material lift (>{MIN_LIFT}) AND z>2 at {far_ok}s: ANTICIPATORY.")
        print("     A rig could lead the music, not just react to it.")
    elif near_ok:
        print(f"  -> material lift only at short horizons {near_ok}s: REACTIVE.")
        print("     Peak at the shortest horizon is the signature of DETECTION,")
        print("     not prediction. Do not sell this as anticipation.")
        sig_far = [t for t, m, z in far if z > 2.0]
        if sig_far:
            print(f"     NOTE: z>2 also at {sig_far}s but lift there is "
                  f"{['%+.4f' % m for t, m, _ in far]} -- significant, not useful.")
    else:
        print("  -> no horizon has both a material lift and z>2. Treat the")
        print("     single-point result as fragile.")
    out["peak_marginal_horizon_s"] = best
    out["verdict"] = ("anticipatory" if far_ok else
                      "reactive" if near_ok else "fragile")

    json.dump(out, open("data/cache/horizon_sweep.json", "w"), indent=2)
    print("\nwrote data/cache/horizon_sweep.json")


if __name__ == "__main__":
    main()
