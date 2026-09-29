"""Event target, done properly: CV over all songs + calibrated permutation null.

scripts/diagnostics/probe_event_target.py gave audio a marginal lift of +0.41x over the rhythm
baseline -- the first positive result in this project. Two things in that run
say DO NOT believe it yet:

  1. The random floor came out at 1.141x, not 1.000x. Average precision is
     upward-biased at small n, so "lift over base rate" is not calibrated and
     a raw lift number cannot be read as skill.
  2. The lag-shift control bounced between 1.09x and 2.14x and peaked at +5 s.
     Either the coupling is fake, or 18 test songs is too few to see it.

Both are variance problems, so fix the variance before interpreting anything:

  - K-FOLD CV over all 140 songs (by song hash) so every song is held out once.
    ~8x more evaluation data than a single 18-song test split.
  - PERMUTATION NULL: recompute the whole statistic with the event train
    CIRCULARLY SHIFTED by a random offset within each song. This preserves the
    event count and the inter-event rhythm exactly, and destroys only the
    alignment to audio. That is the correct null -- an uncalibrated AP bias
    appears identically in the null and cancels.
  - Report z = (observed - null_mean) / null_sd, plus the empirical p-value.

CONFOUND THAT MUST BE NAMED: the light target comes from stage VIDEO
histograms, so a CAMERA CUT produces a novelty spike that is not a lighting
operator action. Concert videos are edited on the music. Any audio->event
coupling found here could be audio->editing, not audio->lighting. This probe
cannot separate them; it is recorded as a limit on what a positive result means.
"""
from __future__ import annotations

import json

import numpy as np

from lightsem.model.data import load_songs
from scripts.diagnostics.probe_event_target import (
    HORIZON,
    HZ,
    audio_feats,
    average_precision,
    event_train,
    hazard_feats,
    target,
)

K_FOLDS = 5
N_PERM = 40
RNG = np.random.default_rng(0)


def fit_ridge(X, y, lam=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return mu, sd, np.linalg.solve(A, Z.T @ (y - y.mean())), y.mean()


def apply_ridge(p, X):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


def circ_shift_events(ev_train: np.ndarray, rng) -> np.ndarray:
    """Rotate the event train within the song: same count, same rhythm, wrong phase."""
    k = int(rng.integers(len(ev_train) // 10, len(ev_train) - len(ev_train) // 10))
    return np.roll(ev_train, k)


def build_song(s, ev_train):
    return (hazard_feats(ev_train), audio_feats(s.x), target(ev_train))


def cv_lift(songs, ev_trains, feature_set: str) -> float:
    """Mean per-song AP/base-rate over K folds. feature_set in {audio,hazard,both}."""
    folds = [[] for _ in range(K_FOLDS)]
    for i in range(len(songs)):
        folds[i % K_FOLDS].append(i)

    built = [build_song(s, e) for s, e in zip(songs, ev_trains)]
    lifts = []
    for f in range(K_FOLDS):
        teidx = set(folds[f])
        tr = [i for i in range(len(songs)) if i not in teidx]
        te = folds[f]

        def mat(idx):
            H = np.concatenate([built[i][0] for i in idx])
            A = np.concatenate([built[i][1] for i in idx])
            Y = np.concatenate([built[i][2] for i in idx])
            if feature_set == "audio":
                X = A
            elif feature_set == "hazard":
                X = H
            else:
                X = np.concatenate([H, A], 1)
            return X, Y

        Xtr, Ytr = mat(tr)
        p = fit_ridge(Xtr, Ytr)
        for i in te:
            H, A, Y = built[i]
            X = A if feature_set == "audio" else H if feature_set == "hazard" \
                else np.concatenate([H, A], 1)
            if Y.sum() < 3:
                continue
            lifts.append(average_precision(apply_ridge(p, X), Y) / Y.mean())
    return float(np.mean(lifts))


def main():
    songs = load_songs()
    print(f"songs {len(songs)}  ({K_FOLDS}-fold CV, every song held out once)")
    ev_trains = [event_train(s.y)[0] for s in songs]
    n_ev = sum(int(e.sum()) for e in ev_trains)
    print(f"events {n_ev}  horizon {HORIZON/HZ:.1f}s")
    out = {"n_songs": len(songs), "n_events": n_ev}

    print("\n=== observed (CV over all songs) ===")
    obs = {}
    for fs in ("audio", "hazard", "both"):
        obs[fs] = cv_lift(songs, ev_trains, fs)
        print(f"  {fs:>8s}  lift {obs[fs]:.4f}x")
    obs_marginal = obs["both"] - obs["hazard"]
    print(f"  audio marginal over rhythm: {obs_marginal:+.4f}x")
    out["observed"] = obs
    out["observed_marginal"] = obs_marginal

    print(f"\n=== permutation null ({N_PERM} reps, circular-shifted events) ===")
    print("  preserves event count + rhythm, destroys audio alignment")
    null_audio, null_marg = [], []
    for r in range(N_PERM):
        perm = [circ_shift_events(e, RNG) for e in ev_trains]
        a = cv_lift(songs, perm, "audio")
        h = cv_lift(songs, perm, "hazard")
        b = cv_lift(songs, perm, "both")
        null_audio.append(a)
        null_marg.append(b - h)
        if (r + 1) % 10 == 0:
            print(f"  {r+1}/{N_PERM}  audio {np.mean(null_audio):.4f} "
                  f"+- {np.std(null_audio):.4f}   "
                  f"marginal {np.mean(null_marg):+.4f} +- {np.std(null_marg):.4f}")
    na, nm = np.array(null_audio), np.array(null_marg)
    out["null_audio"] = {"mean": float(na.mean()), "sd": float(na.std()),
                         "vals": na.tolist()}
    out["null_marginal"] = {"mean": float(nm.mean()), "sd": float(nm.std()),
                            "vals": nm.tolist()}

    print("\n=== verdict ===")
    for tag, o, null in (("audio-only lift", obs["audio"], na),
                         ("audio marginal lift", obs_marginal, nm)):
        z = (o - null.mean()) / (null.std() + 1e-9)
        p = float((null >= o).mean())
        print(f"  {tag:>22s}: obs {o:+.4f}  null {null.mean():+.4f} "
              f"+- {null.std():.4f}   z {z:+.2f}   p {p:.3f}")
        out[tag.replace(" ", "_")] = {"obs": o, "z": float(z), "p": p}

    z_marg = (obs_marginal - nm.mean()) / (nm.std() + 1e-9)
    print()
    if z_marg > 2.0:
        print(f"  -> audio beats its own null by z={z_marg:+.2f}. REAL coupling")
        print("     CAVEAT: could still be audio->camera-editing, not audio->lighting.")
    else:
        print(f"  -> z={z_marg:+.2f} is inside the null. The earlier +0.41x lift was")
        print("     small-sample noise, not signal.")

    json.dump(out, open("data/cache/event_probe_cv.json", "w"), indent=2)
    print("\nwrote data/cache/event_probe_cv.json")


if __name__ == "__main__":
    main()
