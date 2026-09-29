"""How much audio->light signal EXISTS, independent of my model?

The 3-epoch shakedown sat near zero and every linear baseline scored below the
per-song-mean oracle. Before blaming the architecture, measure the ceiling.

Three questions, each with a control:

1. SELF-PREDICTABILITY. How much of y_t is predictable from y's own recent
   past (per-song centred)? This is the autoregressive ceiling. If lighting is
   smooth and slow, a large chunk of R2_within is available WITHOUT audio, and
   any audio model must be compared against it -- otherwise we would credit
   audio for what is really persistence.

2. AUDIO ON TOP OF PERSISTENCE. Does audio add anything once the AR term is in?
   This is the only number that matters for the claim "the model reads music".

3. TIME-SHIFT CONTROL. Re-run the audio-vs-light correlation with y shifted by
   +-30 s within the SAME song. Real coupling must peak near zero lag. A flat
   curve means the apparent correlation is slow drift (both signals wander),
   not moment-to-moment coupling.

All fits are ridge on per-song-centred data, evaluated on held-out SONGS.
"""
from __future__ import annotations

import json

import numpy as np

from lightsem.model.data import TARGET_NAMES, load_songs, split_songs


def fit_ridge(X, Y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Z.T @ Y)


def r2_within(songs, predict):
    num = np.zeros(len(TARGET_NAMES))
    den = np.zeros(len(TARGET_NAMES))
    for s in songs:
        yc = s.y - s.y.mean(0, keepdims=True)
        p, mask = predict(s)
        yc = yc[mask]
        p = p - p.mean(0, keepdims=True)
        yc = yc - yc.mean(0, keepdims=True)
        num += ((yc - p) ** 2).sum(0)
        den += (yc ** 2).sum(0)
    return 1.0 - num / np.maximum(den, 1e-9)


def lagged(y, lags):
    """Stack y at the given POSITIVE lags (past). Returns feats, valid mask."""
    T = len(y)
    m = np.zeros(T, bool)
    m[max(lags):] = True
    cols = [np.roll(y, L, axis=0) for L in lags]
    return np.concatenate(cols, 1), m


def main():
    songs = load_songs()
    tr, te = split_songs(songs)[0], split_songs(songs)[2]
    LAGS = [1, 2, 5, 10, 20, 40]      # 0.1 .. 4.0 s back
    out = {}

    def build(sg, use_audio, use_ar):
        Xs, Ys = [], []
        for s in sg:
            yc = s.y - s.y.mean(0, keepdims=True)
            xc = s.x - s.x.mean(0, keepdims=True)
            parts, mask = [], np.ones(len(yc), bool)
            if use_ar:
                a, m = lagged(yc, LAGS)
                parts.append(a)
                mask &= m
            if use_audio:
                parts.append(xc)
            Xs.append(np.concatenate(parts, 1)[mask])
            Ys.append(yc[mask])
        return np.concatenate(Xs), np.concatenate(Ys)

    def make_pred(mu, sd, W, use_audio, use_ar):
        def pred(s):
            yc = s.y - s.y.mean(0, keepdims=True)
            xc = s.x - s.x.mean(0, keepdims=True)
            parts, mask = [], np.ones(len(yc), bool)
            if use_ar:
                a, m = lagged(yc, LAGS)
                parts.append(a)
                mask &= m
            if use_audio:
                parts.append(xc)
            F = np.concatenate(parts, 1)[mask]
            return ((F - mu) / sd) @ W, mask
        return pred

    print("=== 1+2. what predicts the lighting? (ridge, held-out songs) ===")
    for tag, ua, ar in (("audio only", True, False),
                        ("persistence only (AR)", False, True),
                        ("AR + audio", True, True)):
        X, Y = build(tr, ua, ar)
        mu, sd, W = fit_ridge(X, Y)
        r2 = r2_within(te, make_pred(mu, sd, W, ua, ar))
        out[tag] = r2.tolist()
        print(f"  {tag:22s} " + " ".join(f"{v:7.4f}" for v in r2)
              + f"   MEAN {r2.mean():+.4f}")

    delta = np.array(out["AR + audio"]).mean() - np.array(out["persistence only (AR)"]).mean()
    print(f"\n  audio's marginal contribution over persistence: {delta:+.4f}")

    print("\n=== 3. time-shift control (does coupling peak at lag 0?) ===")
    rng = np.random.default_rng(0)
    lags_s = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
    curve = []
    for L in lags_s:
        k = int(L * 10)
        rs = []
        for s in te:
            if len(s.x) < abs(k) + 100:
                continue
            xc = s.x - s.x.mean(0, keepdims=True)
            yc = s.y - s.y.mean(0, keepdims=True)
            if k >= 0:
                a, b = xc[: len(xc) - k], yc[k:]
            else:
                a, b = xc[-k:], yc[: len(yc) + k]
            a = a / (a.std(0) + 1e-8)
            b = b / (b.std(0) + 1e-8)
            r = (a.T @ b) / len(a)          # (512,6)
            rs.append(np.abs(r).mean())
        curve.append(float(np.mean(rs)))
        print(f"  lag {L:+4d} s   mean|r| = {curve[-1]:.5f}")
    peak = lags_s[int(np.argmax(curve))]
    print(f"\n  peak at lag {peak:+d} s "
          f"({'GOOD - real coupling' if abs(peak) <= 2 else 'BAD - drift, not coupling'})")
    flat = (max(curve) - min(curve)) / max(curve)
    print(f"  relative modulation across lags: {flat:.3f} "
          f"({'flat -> drift' if flat < 0.15 else 'peaked -> real'})")

    out["shift_curve"] = {"lags_s": lags_s, "mean_abs_r": curve, "peak_lag_s": peak}
    json.dump(out, open("data/cache/ceiling_probe.json", "w"), indent=2)
    print("\nwrote data/cache/ceiling_probe.json")


if __name__ == "__main__":
    main()
