"""Does audio predict lighting EVENTS (scene changes), not levels?

Everything so far regressed audio -> absolute lighting state and failed:
persistence explains R2_within +0.708 with no audio at all, and audio's
marginal contribution is -0.005. But that comparison is rigged in persistence's
favour BY CONSTRUCTION -- a slowly-varying target is trivially predicted by its
own past, and the 71% is mostly "the lights are still doing what they were
doing". Persistence is useless at exactly the moments a lighting desk acts.

So reformulate the target as the operator's ACTION: when does the scene change?
This is the one formulation the level-regression result does not rule out, and
it is also what a real desk fires (a cue, not a voltage).

DESIGN, and where each piece can go wrong:

1. EVENTS. Per-song z-scored y, novelty = || mean(y[t:t+k]) - mean(y[t-k:t]) ||,
   peak-picked with a minimum separation. Sanity-checked before use: event rate
   per minute and inter-event interval distribution must look like a human
   operator, not like noise crossings. If events are noise, everything after is
   meaningless, so this is reported first.

2. THE HONEST COMPETITOR is NOT persistence -- it cannot predict changes. It is
   the RHYTHM of past changes: time since last event, recent event rate, and
   phase against the song's own median inter-event interval. A lighting desk
   that just fires every N seconds is a real, deployable baseline and it needs
   no audio. Audio must beat THAT.

3. AUDIO FEATURES are causal and built to detect CHANGE, not level: multi-scale
   past means, and the difference between recent and older means (an audio
   novelty curve). Pairing change-with-change is the whole hypothesis.

4. METRIC is average precision per song, averaged over held-out SONGS. The
   floor is the base rate (a constant predictor). Reported as lift over base
   rate, because AP is not comparable across songs with different event rates.

5. CONTROL: shift the audio score against the event train within the same song.
   Real coupling must peak at lag 0. This is the control that killed the
   level-regression claim, so it is applied here before believing anything.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import uniform_filter1d

from lightsem.model.data import load_songs, split_songs

HZ = 10.0
NOV_K = 10          # 1.0 s half-window for the novelty kernel
MIN_SEP = 20        # 2.0 s minimum separation between events
PCTL = 90.0         # peak must exceed this percentile of the song's novelty
HORIZON = 10        # predict an event within the next 1.0 s
PAST = [5, 10, 20, 40]   # audio pooling scales (0.5 .. 4.0 s)


# ---------------------------------------------------------------- events

def _win_mean(z: np.ndarray, k: int, side: str) -> np.ndarray:
    """Mean of the k frames strictly before ('pre') or after ('post') each t.

    Done with cumsum rather than uniform_filter1d because scipy caps `origin`
    at (size-1)//2, so a fully-shifted window is not expressible there.
    """
    T = len(z)
    pad = np.concatenate([np.repeat(z[:1], k, 0), z, np.repeat(z[-1:], k, 0)])
    cs = np.concatenate([np.zeros((1, z.shape[1]), z.dtype), np.cumsum(pad, 0)])
    t = np.arange(T) + k                     # index into pad
    if side == "pre":
        lo, hi = t - k, t                    # [t-k, t)
    else:
        lo, hi = t + 1, t + 1 + k            # (t, t+k]
    return (cs[hi] - cs[lo]) / k


def novelty(y: np.ndarray) -> np.ndarray:
    """Novelty: how different is the near future from the near past."""
    z = (y - y.mean(0)) / (y.std(0) + 1e-6)
    z = uniform_filter1d(z, size=3, axis=0, mode="nearest")
    return np.linalg.norm(_win_mean(z, NOV_K, "post") - _win_mean(z, NOV_K, "pre"),
                          axis=1)


def pick_events(nov: np.ndarray) -> np.ndarray:
    thr = np.percentile(nov, PCTL)
    idx = []
    order = np.argsort(-nov)
    taken = np.zeros(len(nov), bool)
    for i in order:
        if nov[i] < thr:
            break
        if taken[max(0, i - MIN_SEP): i + MIN_SEP + 1].any():
            continue
        idx.append(i)
        taken[i] = True
    return np.array(sorted(idx), dtype=np.int64)


def event_train(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ev = pick_events(novelty(y))
    t = np.zeros(len(y), np.float32)
    t[ev] = 1.0
    return t, ev


# ---------------------------------------------------------------- features

def hazard_feats(train: np.ndarray) -> np.ndarray:
    """Causal rhythm-of-past-changes features. No audio."""
    T = len(train)
    tsl = np.zeros(T, np.float32)     # time since last event (frames)
    c = 1e3
    for t in range(T):
        tsl[t] = c
        c = 0.0 if train[t] > 0 else c + 1.0
    tsl = np.minimum(tsl, 600.0)
    rate = uniform_filter1d(train, size=300, origin=149, mode="nearest")  # last 30 s
    iei = np.diff(np.flatnonzero(train))
    med = float(np.median(iei)) if len(iei) else 100.0
    med = max(med, 5.0)
    ph = 2 * np.pi * tsl / med
    return np.stack([tsl / 100.0, np.log1p(tsl), rate * 100.0,
                     np.cos(ph), np.sin(ph), np.ones(T, np.float32)], 1)


def audio_feats(x: np.ndarray) -> np.ndarray:
    """Causal multi-scale pooling + novelty (recent mean minus older mean)."""
    xc = (x - x.mean(0)) / (x.std(0) + 1e-6)
    pools = []
    for k in PAST:
        pools.append(uniform_filter1d(xc, size=k, axis=0, origin=(k - 1) // 2,
                                      mode="nearest"))
    nov = [pools[0] - pools[i] for i in range(1, len(PAST))]   # change detectors
    return np.concatenate(pools + nov, 1).astype(np.float32)


def target(train: np.ndarray) -> np.ndarray:
    """1 if an event falls in (t, t+HORIZON]."""
    T = len(train)
    out = np.zeros(T, np.float32)
    cs = np.concatenate([[0.0], np.cumsum(train)])
    for t in range(T):
        e = min(T, t + 1 + HORIZON)
        out[t] = 1.0 if cs[e] - cs[t + 1] > 0 else 0.0
    return out


# ---------------------------------------------------------------- metric

def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    o = np.argsort(-score)
    lab = label[o]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    n = lab.sum()
    return float((prec * lab).sum() / n) if n > 0 else float("nan")


def fit_ridge(X, y, lam=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Z.T @ (y - y.mean())), y.mean()


def apply_ridge(p, X):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


# ---------------------------------------------------------------- main

def main():
    songs = load_songs()
    tr, te = split_songs(songs)[0], split_songs(songs)[2]
    print(f"train songs {len(tr)}  test songs {len(te)}")
    out = {}

    print("\n=== 0. are the 'events' real? (sanity before anything else) ===")
    rates, ieis, peakiness = [], [], []
    for s in songs:
        nov = novelty(s.y)
        _, ev = event_train(s.y)
        mins = len(s.y) / HZ / 60.0
        rates.append(len(ev) / max(mins, 1e-6))
        if len(ev) > 1:
            ieis.extend((np.diff(ev) / HZ).tolist())
        peakiness.append(float(nov.max() / (np.median(nov) + 1e-9)))
    rates, ieis = np.array(rates), np.array(ieis)
    print(f"  events/min   median {np.median(rates):.1f}  "
          f"p10 {np.percentile(rates,10):.1f}  p90 {np.percentile(rates,90):.1f}")
    print(f"  inter-event  median {np.median(ieis):.1f} s  "
          f"p10 {np.percentile(ieis,10):.1f}  p90 {np.percentile(ieis,90):.1f}")
    print(f"  novelty peakiness (max/median) median {np.median(peakiness):.1f}")
    out["events_per_min_median"] = float(np.median(rates))
    out["iei_median_s"] = float(np.median(ieis))
    out["novelty_peakiness_median"] = float(np.median(peakiness))

    # ---- build matrices
    def build(sg):
        H, A, Y, sid = [], [], [], []
        for i, s in enumerate(sg):
            tr_, _ = event_train(s.y)
            H.append(hazard_feats(tr_))
            A.append(audio_feats(s.x))
            Y.append(target(tr_))
            sid.append(np.full(len(s.x), i))
        return (np.concatenate(H), np.concatenate(A),
                np.concatenate(Y), np.concatenate(sid))

    Htr, Atr, Ytr, _ = build(tr)
    Hte, Ate, Yte, Ste = build(te)
    base = float(Yte.mean())
    print(f"\n  base rate of 'event within next {HORIZON/HZ:.1f} s' = {base:.3f}")
    out["base_rate"] = base

    def per_song_ap(score):
        aps, lifts = [], []
        for i in np.unique(Ste):
            m = Ste == i
            if Yte[m].sum() < 3:
                continue
            ap = average_precision(score[m], Yte[m])
            br = Yte[m].mean()
            aps.append(ap)
            lifts.append(ap / br)
        return float(np.mean(aps)), float(np.mean(lifts))

    print("\n=== 1. what predicts the NEXT lighting event? (held-out songs) ===")
    print(f"  {'model':>26s}  {'AP':>7s}  {'lift over base':>15s}")
    res = {}
    for tag, ftr, fte in (
        ("audio only", Atr, Ate),
        ("rhythm only (hazard)", Htr, Hte),
        ("hazard + audio", np.concatenate([Htr, Atr], 1),
         np.concatenate([Hte, Ate], 1)),
    ):
        p = fit_ridge(ftr, Ytr)
        ap, lift = per_song_ap(apply_ridge(p, fte))
        res[tag] = {"ap": ap, "lift": lift}
        print(f"  {tag:>26s}  {ap:7.4f}  {lift:15.3f}x")
    rnd = np.random.default_rng(0).standard_normal(len(Yte))
    ap, lift = per_song_ap(rnd)
    res["random"] = {"ap": ap, "lift": lift}
    print(f"  {'random (floor)':>26s}  {ap:7.4f}  {lift:15.3f}x")

    d = res["hazard + audio"]["lift"] - res["rhythm only (hazard)"]["lift"]
    print(f"\n  audio's marginal lift over rhythm: {d:+.4f}x")
    out["models"] = res
    out["audio_marginal_lift"] = d

    print("\n=== 2. lag-shift control on the audio score ===")
    p = fit_ridge(Atr, Ytr)
    curve = []
    lags = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
    for L in lags:
        k = int(L * HZ)
        aps = []
        for i in np.unique(Ste):
            m = Ste == i
            sc, yy = apply_ridge(p, Ate[m]), Yte[m]
            if len(yy) < abs(k) + 60:
                continue
            a, b = (sc[: len(sc) - k], yy[k:]) if k >= 0 else (sc[-k:], yy[: len(yy) + k])
            if b.sum() < 3:
                continue
            aps.append(average_precision(a, b) / b.mean())
        curve.append(float(np.mean(aps)))
        print(f"  lag {L:+4d} s   lift = {curve[-1]:.4f}x")
    peak = lags[int(np.argmax(curve))]
    print(f"\n  peak at lag {peak:+d} s "
          f"({'GOOD - real coupling' if abs(peak) <= 2 else 'BAD - not coupling'})")
    out["shift_curve"] = {"lags_s": lags, "lift": curve, "peak_lag_s": peak}

    json.dump(out, open("data/cache/event_probe.json", "w"), indent=2)
    print("\nwrote data/cache/event_probe.json")


if __name__ == "__main__":
    main()
