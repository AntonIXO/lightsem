#!/usr/bin/env python
"""Replicate Burger et al. 2013 stimulus-level correlations on AIST++.

Burger, Thompson, Luck, Saarikallio & Toiviainen 2013 (N = 60 dancers, 30 pop
clips) report stimulus-level correlations between audio features and movement:
pulse clarity -> quantity of motion r = 0.62, centre-of-mass speed 0.67, foot
speed 0.55; low-frequency flux -> head speed 0.73; high-frequency flux -> hand
speed 0.65. Those are correlations across STIMULI of per-stimulus averages, so
this reproduces exactly that design: one row per track, movement averaged over
every dancer and every frame, audio averaged over the track.

If they replicate here, the shared skeleton exists on this corpus and any
failure to predict it is a modelling failure. If they do not, AIST++ is the
wrong corpus for block A -- it is choreographed, loudness-normalised, tempo-
stratified studio dance, not free movement to varied music -- and block A needs
free-dancing data (the Oslo club-dancing mocap) before it can be trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors

PAIRS = (
    ("pulse_clarity", "qom", 0.62), ("pulse_clarity", "com_speed", 0.67),
    ("pulse_clarity", "foot_speed", 0.55), ("pulse_clarity", "hip_yaw_speed", 0.58),
    ("flux_lf", "head_speed", 0.73), ("flux_hf", "hand_speed", 0.65),
    ("flux_hf", "head_speed", 0.64), ("flux_hf", "hand_spread", 0.57),
    ("percussive_ratio", "head_speed", 0.67), ("rms", "qom", None),
    ("energy_lf", "qom", None), ("energy_lf", "vert_bounce_amp", None),
    ("tempo", "qom", 0.0), ("tempo", "vert_period_s", None),
)
AUDIO_F = ("pulse_clarity", "flux_lf", "flux_hf", "percussive_ratio", "rms",
           "energy_lf", "onset", "centroid", "flatness")


def boot_ci(x, y, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    vals = []
    for _ in range(n):
        i = rng.integers(0, n_obs, n_obs)
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        vals.append(np.corrcoef(x[i], y[i])[0, 1])
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    by_music: dict[str, list] = {}
    for r in idx:
        by_music.setdefault(r["music_id"], []).append(r)

    rows = []
    for mid, rs in sorted(by_music.items()):
        descs = [load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz") for r in rs]
        bus = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        rec = {"music_id": mid, "n_clips": len(descs), "tempo": bus.tempo,
               "genre": rs[0]["genre"]}
        for f in AUDIO_F:
            rec[f] = float(np.mean(bus.channels[f]))
        for ch in descs[0].channels:
            rec[ch] = float(np.mean([np.mean(d.channels[ch]) for d in descs]))
        rows.append(rec)
    print(f"{len(rows)} tracks, {sum(r['n_clips'] for r in rows)} clips")

    print(f"\n{'audio':17s} {'movement':17s} {'r(AIST++)':>10s} {'95% CI':>18s} "
          f"{'Burger 2013':>12s}")
    out = []
    for af, mf, ref in PAIRS:
        x = np.array([r[af] for r in rows])
        y = np.array([r[mf] for r in rows])
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(x, y)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        out.append({"audio": af, "movement": mf, "r": r, "ci": [lo, hi],
                    "burger": ref, "significant": sig == "yes"})
        print(f"{af:17s} {mf:17s} {r:+10.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{('%+.2f' % ref) if ref is not None else '     -':>12s}")

    # partial out tempo, which AIST varies by design (80-135 BPM in 10 BPM steps)
    print("\nsame, with tempo partialled out (AIST varies tempo by design):")
    tempo = np.array([r["tempo"] for r in rows])
    def resid(v):
        A = np.c_[tempo, np.ones(len(tempo))]
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    for af, mf, ref in PAIRS:
        if af == "tempo":
            continue
        x = resid(np.array([r[af] for r in rows]))
        y = resid(np.array([r[mf] for r in rows]))
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(x, y)
        print(f"{af:17s} {mf:17s} {r:+10.3f} [{lo:+.3f},{hi:+.3f}]")

    # how much does the corpus's audio actually vary?
    print("\naudio feature variation across the 60 tracks (CV = sd/mean):")
    for f in AUDIO_F + ("tempo",):
        v = np.array([r[f] for r in rows])
        print(f"  {f:17s} mean {v.mean():10.3f}  sd {v.std():9.3f}  CV {v.std()/abs(v.mean()+1e-9):.3f}")

    p = Path("/root/lightsem/experiments/legacy/stages/outputs/stage0b"); p.mkdir(parents=True, exist_ok=True)
    (p / "burger_replication.json").write_text(json.dumps(
        {"pairs": out, "n_tracks": len(rows), "per_track": rows}, indent=1))




def within_genre():
    """Same correlations with GENRE partialled out.

    AIST pairs each dance genre with its own music, so a between-track
    correlation is partly a between-genre comparison: break dance music next to
    break dance choreography. Removing genre (and tempo) leaves the question the
    plan actually needs answered — within one style, do louder/punchier tracks
    produce more movement?
    """
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    by_music: dict[str, list] = {}
    for r in idx:
        by_music.setdefault(r["music_id"], []).append(r)
    rows = []
    for mid, rs in sorted(by_music.items()):
        descs = [load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz") for r in rs]
        bus = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        rec = {"music_id": mid, "genre": rs[0]["genre"], "tempo": bus.tempo}
        for f in AUDIO_F:
            rec[f] = float(np.mean(bus.channels[f]))
        for ch in descs[0].channels:
            rec[ch] = float(np.mean([np.mean(d.channels[ch]) for d in descs]))
        rows.append(rec)
    genres = sorted({r["genre"] for r in rows})
    G = np.array([[1.0 * (r["genre"] == g) for g in genres] + [r["tempo"], 1.0]
                  for r in rows])

    def resid(v):
        return v - G @ np.linalg.lstsq(G, v, rcond=None)[0]

    print(f"\n=== genre AND tempo partialled out (n={len(rows)} tracks, "
          f"{len(genres)} genres) ===")
    print(f"{'audio':17s} {'movement':17s} {'r':>8s} {'95% CI':>18s} {'Burger':>8s}")
    res = []
    for af, mf, ref in PAIRS:
        if af == "tempo":
            continue
        x, y = resid(np.array([r[af] for r in rows])), resid(np.array([r[mf] for r in rows]))
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = boot_ci(x, y)
        res.append({"audio": af, "movement": mf, "r": r, "ci": [lo, hi], "burger": ref})
        print(f"{af:17s} {mf:17s} {r:+8.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{('%+.2f' % ref) if ref is not None else '   -':>8s}")
    p = Path("/root/lightsem/experiments/legacy/stages/outputs/stage0b")
    (p / "burger_within_genre.json").write_text(json.dumps(res, indent=1))
    # how much movement variance is genre?
    print("\nshare of between-track movement variance explained by genre alone:")
    Gg = np.array([[1.0 * (r["genre"] == g) for g in genres] for r in rows])
    for mf in ("qom", "com_speed", "head_speed", "hand_speed", "foot_speed",
               "vert_bounce_amp", "footfall_rate", "spread"):
        v = np.array([r[mf] for r in rows])
        pred = Gg @ np.linalg.lstsq(Gg, v, rcond=None)[0]
        r2 = 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()
        print(f"  {mf:18s} R2(genre) = {r2:.3f}")


if __name__ == "__main__":
    main()
    within_genre()
