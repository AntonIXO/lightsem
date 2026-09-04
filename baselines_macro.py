"""Baselines the trained model must beat, computed BEFORE training.

The metric that matters is R2_within: variance explained AFTER removing each
song's own mean. Global R2 is inflated because ~40-65% of target variance is
between-song (venue/palette identity), which a lighting model does not need
to predict -- it is given the show.

Baselines, weakest to strongest:
  1. global mean       -- predicts the dataset mean. R2_within <= 0 by construction.
  2. per-song mean     -- ORACLE: uses the true mean of the held-out song.
                          This is the honest zero point. R2_within == 0 exactly.
  3. ridge, 1 frame    -- linear map from a single openl3 frame. No temporal
                          context, so it isolates "how much is instantaneous".
  4. ridge, mean-pooled window -- linear + 6.4 s of context, no learned dynamics.

If the transformer cannot beat #4 it has learned nothing that a linear model
with a moving average does not already have.
"""
from __future__ import annotations

import json

import numpy as np

from lightsem.model.data import TARGET_NAMES, load_songs, split_songs

W = 64  # frames @10 Hz = 6.4 s, same window the model gets


def r2_within(songs, predict) -> np.ndarray:
    """R2 computed on per-song-centred targets. predict(song)->(T,6) or None."""
    num = np.zeros(len(TARGET_NAMES))
    den = np.zeros(len(TARGET_NAMES))
    for s in songs:
        y = s.y
        yc = y - y.mean(0, keepdims=True)      # remove song identity
        p = predict(s)
        pc = p - p.mean(0, keepdims=True)      # baseline gets the same courtesy
        num += ((yc - pc) ** 2).sum(0)
        den += (yc ** 2).sum(0)
    return 1.0 - num / np.maximum(den, 1e-9)


def fit_ridge(Xtr, Ytr, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1], dtype=np.float64)
    B = Z.T @ (Ytr - Ytr.mean(0))
    Wm = np.linalg.solve(A, B)
    return mu, sd, Wm, Ytr.mean(0)


def pool(x, w):
    """Causal mean over the trailing w frames."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1]), np.float64), x]), 0)
    idx = np.arange(len(x))
    lo = np.maximum(idx + 1 - w, 0)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)[:, None]


def main():
    songs = load_songs()
    tr, va, te = split_songs(songs)
    print(f"songs train={len(tr)} val={len(va)} test={len(te)}")
    print(f"frames train={sum(len(s.x) for s in tr):,} test={sum(len(s.x) for s in te):,}")

    res = {}

    # 1. global mean
    gmean = np.concatenate([s.y for s in tr]).mean(0)
    res["global_mean"] = r2_within(te, lambda s: np.repeat(gmean[None], len(s.y), 0))

    # 2. per-song mean ORACLE
    res["per_song_mean_oracle"] = r2_within(
        te, lambda s: np.repeat(s.y.mean(0, keepdims=True), len(s.y), 0))

    # 3. ridge on a single frame
    Xtr = np.concatenate([s.x for s in tr]).astype(np.float64)
    Ytr = np.concatenate([s.y for s in tr]).astype(np.float64)
    mu, sd, Wm, ym = fit_ridge(Xtr, Ytr)
    res["ridge_1frame"] = r2_within(te, lambda s: ((s.x - mu) / sd) @ Wm + ym)

    # 4. ridge on causal mean-pooled window
    Xp = np.concatenate([pool(s.x.astype(np.float64), W) for s in tr])
    mu2, sd2, Wm2, ym2 = fit_ridge(Xp, Ytr)
    res["ridge_pooled64"] = r2_within(
        te, lambda s: ((pool(s.x.astype(np.float64), W) - mu2) / sd2) @ Wm2 + ym2)

    # 5+6. FAIR ridge: fit on per-song-CENTRED x and y.
    # Baselines 3/4 are fit on absolute targets, which are dominated by
    # between-song palette variance; asking them for within-song deviations
    # is off-distribution and they score far below zero. To claim the
    # transformer beat "a linear model" we must beat a linear model trained
    # on the same objective we evaluate.
    def centred(sg, w=None):
        xs = [(s.x.astype(np.float64) if w is None else pool(s.x.astype(np.float64), w))
              for s in sg]
        xs = [a - a.mean(0, keepdims=True) for a in xs]
        ys = [s.y.astype(np.float64) - s.y.astype(np.float64).mean(0, keepdims=True)
              for s in sg]
        return np.concatenate(xs), np.concatenate(ys)

    for tag, w in (("ridge_centred_1frame", None), ("ridge_centred_pooled64", W)):
        Xc, Yc = centred(tr, w)
        muc, sdc, Wc, ymc = fit_ridge(Xc, Yc)

        def pred(s, w=w, muc=muc, sdc=sdc, Wc=Wc, ymc=ymc):
            a = s.x.astype(np.float64) if w is None else pool(s.x.astype(np.float64), w)
            a = a - a.mean(0, keepdims=True)
            return ((a - muc) / sdc) @ Wc + ymc

        res[tag] = r2_within(te, pred)

    print(f"\n{'baseline':24s} " + " ".join(f"{n:>9s}" for n in TARGET_NAMES) + "     MEAN")
    print("-" * 96)
    for k, v in res.items():
        print(f"{k:24s} " + " ".join(f"{x:9.4f}" for x in v) + f"  {v.mean():7.4f}")

    best = max(res["ridge_centred_1frame"].mean(), res["ridge_centred_pooled64"].mean())
    print("\nInterpretation:")
    print("  per_song_mean_oracle is the ZERO POINT (uses held-out truth).")
    print(f"  BAR TO BEAT (best fair linear): mean R2_within = {best:.4f}")

    json.dump({k: v.tolist() for k, v in res.items()},
              open("data/cache/baselines.json", "w"), indent=2)
    print("\nwrote data/cache/baselines.json")


if __name__ == "__main__":
    main()
