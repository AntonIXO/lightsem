#!/usr/bin/env python
"""Stage 2B — the operator head on the FULL RPMC-L2 (runbook rb-ls2-02).

The August result (AP lift +0.09, z +2.83) came from 140 of the 699 songs,
because the corpus did not fit on the root disk. All 699 are local now, so the
same probe runs on 5x the data with the same code path, the same event
definition and the same honest competitor: the rhythm of past changes.

    bash run.sh scripts/legacy_stages/run_stage2.py [n_songs]

Gate (from the plan): event AP lift >= +0.09 over the rhythm baseline with
z > 3, and no anticipation claim unless the lift at +4 s exceeds +0.05 with a
CI excluding zero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from lightsem.model.data import load_songs, split_songs
from scripts.diagnostics.probe_event_target import (HORIZON, HZ, audio_feats, average_precision,
                                apply_ridge, event_train, fit_ridge, hazard_feats,
                                novelty, target)

ROOT = "/mnt/lightsem-data/cache/rpmc_all"
OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage2")


def feats(song, which: str):
    """Per-song feature block. 'audio' is 3584-dimensional (512 openl3 x 4 causal
    pools + 3 novelty differences), which is why nothing here ever holds the whole
    training set in memory: 478 songs of it is 34 GB and the box has 31."""
    tr_, _ = event_train(song.y)
    if which == "rhythm":
        F = hazard_feats(tr_)
    elif which == "audio":
        F = audio_feats(song.x)
    else:
        F = np.concatenate([hazard_feats(tr_), audio_feats(song.x)], 1)
    return F.astype(np.float32), target(tr_)


def fit_streaming(songs, which: str, lam: float = 30.0):
    """Ridge by accumulating the normal equations one song at a time."""
    XtX = Xty = None
    for s in songs:
        F, y = feats(s, which)
        F1 = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1).astype(np.float64)
        if XtX is None:
            d = F1.shape[1]
            XtX = np.zeros((d, d)); Xty = np.zeros(d)
        XtX += F1.T @ F1
        Xty += F1.T @ y.astype(np.float64)
    d = XtX.shape[0]
    return np.linalg.solve(XtX + lam * np.eye(d), Xty)


def score_song(song, which: str, w):
    F, y = feats(song, which)
    F1 = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1).astype(np.float64)
    return F1 @ w, y


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    songs = load_songs(root=ROOT)[:n]
    tr, va, te = split_songs(songs)
    print(f"{len(songs)} songs: train {len(tr)} val {len(va)} test {len(te)}")
    rng = np.random.default_rng(0)

    rates, ieis = [], []
    for s in songs:
        _, ev = event_train(s.y)
        rates.append(len(ev) / max(len(s.y) / HZ / 60.0, 1e-6))
        if len(ev) > 1:
            ieis.extend((np.diff(ev) / HZ).tolist())
    print(f"events/min median {np.median(rates):.1f}; inter-event median "
          f"{np.median(ieis):.1f} s")

    base = float(np.mean([target(event_train(s.y)[0]).mean() for s in te]))
    print(f"base rate of an event within {HORIZON/HZ:.1f} s = {base:.3f}")

    res, scores, weights = {}, {}, {}
    for tag, which in (("audio_only", "audio"), ("rhythm_only", "rhythm"),
                       ("rhythm_plus_audio", "both")):
        w = fit_streaming(tr, which)
        weights[which] = w
        lifts, keep = [], []
        for j, s in enumerate(te):
            sc, y = score_song(s, which, w)
            if y.sum() < 3:
                continue
            lifts.append(average_precision(sc, y) / y.mean())
            keep.append(j)
        scores[tag] = np.array(lifts)
        res[tag] = {"lift_mean": float(scores[tag].mean()),
                    "lift_sd": float(scores[tag].std()), "n_songs": int(len(lifts))}
        print(f"  {tag:>18s}  lift {scores[tag].mean():.4f}x  (n={len(lifts)})",
              flush=True)
    keep_idx = keep

    d = scores["rhythm_plus_audio"] - scores["rhythm_only"]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
    z = float(d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12))
    res["audio_marginal"] = {
        "lift": float(d.mean()), "z": z,
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "n_songs": int(len(d))}
    print(f"\naudio's marginal lift over rhythm: {d.mean():+.4f}x  z={z:.2f}  "
          f"CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")

    print("\nlag-shift control (real coupling must peak at 0):")
    w = weights["audio"]
    lags = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
    per_song_scores = []
    for j in keep_idx:
        per_song_scores.append(score_song(te[j], "audio", w))
    curve = []
    for L in lags:
        k = int(L * HZ)
        aps = []
        for sc, yy in per_song_scores:
            if len(yy) < abs(k) + 60:
                continue
            a, b = (sc[: len(sc) - k], yy[k:]) if k >= 0 else (sc[-k:], yy[: len(yy) + k])
            if b.sum() < 3:
                continue
            aps.append(average_precision(a, b) / b.mean())
        curve.append(float(np.mean(aps)))
    peak = lags[int(np.argmax(curve))]
    print("  " + "  ".join(f"{L:+d}s:{c:.3f}" for L, c in zip(lags, curve)))
    print(f"  peak at {peak:+d} s")
    res["shift_curve"] = {"lags_s": lags, "lift": curve, "peak_lag_s": peak}

    gate = (res["audio_marginal"]["lift"] >= 0.09 and z > 3 and abs(peak) <= 2)
    res["gates"] = {"G_event_ap_lift": {
        "passed": bool(gate),
        "evidence": f"audio marginal lift {res['audio_marginal']['lift']:+.4f}x "
                    f"(needed >= +0.09), z {z:.2f} (needed > 3), shift-control peak "
                    f"at {peak:+d} s (needed |peak| <= 2 s), on {len(songs)} songs"}}
    res["n_songs"] = len(songs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_head.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    main()
