"""Does the PIVOT chain motion -> sound -> light have any headroom?

The proposal: no dataset has (motion, light) pairs, but AIST has (motion, sound)
and RPMC-L2 has (sound, light). Compose them through sound as a pivot modality.
This is a legitimate technique (bridge/pivot cross-modal transfer). The question
is whether it can carry signal HERE.

Governing constraint (data processing inequality): if the chain is
motion -> sound -> light, then motion is conditionally independent of light
given sound. So ANY motion->light predictor built this way factors through
sound, and its ceiling is set by what the MOTION-RECOVERABLE PART OF SOUND can
say about light. Making the motion->sound model better cannot exceed that.

Measured input to this probe (scripts/diagnostics/probe1_pose_vs_audio.py, AIST, 263 sequences):
  audio -> kinetic energy R2 (GroupKFold by music) = 0.0019 at 0.1 s
  but r = 0.65 at 8 s smoothing
So motion and sound share information ONLY as a slow envelope. Fine-scale is
dancer-idiosyncratic (inter-dancer ceiling r = 0.024).

Therefore this probe gives link 1 an ORACLE: assume motion recovers the audio
envelope PERFECTLY at timescale w, and recovers nothing faster. Then ask what
that band alone predicts about light, over persistence. That is an upper bound
on the whole chain.

Generous on purpose:
  - smoothing is CENTRED (uses the future) -- a real causal system gets less
  - the oracle is perfect within band -- a real motion->sound model gets less
If the upper bound is zero, the chain is dead regardless of model quality.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import uniform_filter1d

from lightsem.model.data import TARGET_NAMES, load_songs, split_songs
from scripts.diagnostics.probe_signal_ceiling import fit_ridge, lagged, r2_within

FPS = 10.0
LAGS = [1, 2, 5, 10, 20, 40]          # 0.1 .. 4.0 s of the light's own past
# 0.0 = no smoothing (full audio). 8.0 s is where AIST motion<->audio r hits 0.65.
WINDOWS_S = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]


def band(x, w_s, part):
    """Split audio features into slow (motion-recoverable) / fast components."""
    if w_s <= 0:
        return x if part == "slow" else np.zeros_like(x)
    k = max(1, int(round(w_s * FPS)))
    slow = uniform_filter1d(x, size=k, axis=0, mode="nearest")
    return slow if part == "slow" else x - slow


def build(sg, w_s, part, use_audio=True):
    Xs, Ys = [], []
    for s in sg:
        yc = s.y - s.y.mean(0, keepdims=True)
        a, mask = lagged(yc, LAGS)
        parts = [a]
        if use_audio:
            xc = s.x - s.x.mean(0, keepdims=True)
            parts.append(band(xc, w_s, part))
        Xs.append(np.concatenate(parts, 1)[mask])
        Ys.append(yc[mask])
    return np.concatenate(Xs), np.concatenate(Ys)


def make_pred(mu, sd, W, w_s, part, use_audio=True):
    def pred(s):
        yc = s.y - s.y.mean(0, keepdims=True)
        a, mask = lagged(yc, LAGS)
        parts = [a]
        if use_audio:
            xc = s.x - s.x.mean(0, keepdims=True)
            parts.append(band(xc, w_s, part))
        F = np.concatenate(parts, 1)[mask]
        return ((F - mu) / sd) @ W, mask
    return pred


def score(tr, te, w_s, part, use_audio=True):
    X, Y = build(tr, w_s, part, use_audio)
    mu, sd, W = fit_ridge(X, Y)
    return r2_within(te, make_pred(mu, sd, W, w_s, part, use_audio))


def main():
    songs = load_songs()
    sp = split_songs(songs)
    tr, te = sp[0], sp[2]
    print(f"train songs {len(tr)}  test songs {len(te)}")
    out = {"targets": TARGET_NAMES}

    print("\n=== 0. reference: persistence alone (the bar to beat) ===")
    ar = score(tr, te, 0.0, "slow", use_audio=False)
    out["persistence_only"] = ar.tolist()
    print(f"  AR only                       MEAN {ar.mean():+.4f}")

    print("\n=== 1. how much audio variance survives the motion bottleneck? ===")
    x0 = np.concatenate([s.x - s.x.mean(0, keepdims=True) for s in te])
    tot = x0.var(0).sum()
    frac = {}
    for w in WINDOWS_S[1:]:
        sl = np.concatenate(
            [band(s.x - s.x.mean(0, keepdims=True), w, "slow") for s in te])
        frac[w] = float(sl.var(0).sum() / tot)
        print(f"  smoothing {w:5.1f} s -> slow band keeps {frac[w]*100:5.1f}% "
              f"of audio variance")
    out["slow_band_variance_fraction"] = frac

    print("\n=== 2. marginal contribution over persistence, per band ===")
    print(f"  {'band':>28s}  {'MEAN R2_within':>15s}  {'marginal':>9s}")
    rows = {}
    for w in WINDOWS_S:
        for part in (("slow",) if w == 0 else ("slow", "fast")):
            r2 = score(tr, te, w, part)
            d = r2.mean() - ar.mean()
            tag = "full audio (no split)" if w == 0 else f"{part} band, w={w:.0f}s"
            rows[tag] = {"r2": r2.tolist(), "marginal": float(d)}
            print(f"  {tag:>28s}  {r2.mean():+15.4f}  {d:+9.4f}")
    out["bands"] = rows

    print("\n=== 3. verdict on the pivot chain ===")
    mot = rows.get("slow band, w=8s", {}).get("marginal", float("nan"))
    best = max(v["marginal"] for v in rows.values())
    bestk = [k for k, v in rows.items() if v["marginal"] == best][0]
    print(f"  motion-recoverable band (8 s, oracle link 1): {mot:+.4f}")
    print(f"  best band of ANY timescale:                   {best:+.4f}  ({bestk})")
    out["motion_band_marginal"] = mot
    out["best_band_marginal"] = best
    out["best_band"] = bestk
    if best <= 0.005:
        print("  -> no band of audio adds over persistence. Chain ceiling ~ 0.")
    else:
        print("  -> some band carries signal; chain is worth pursuing there.")

    json.dump(out, open("data/cache/pivot_probe.json", "w"), indent=2)
    print("\nwrote data/cache/pivot_probe.json")


if __name__ == "__main__":
    main()
