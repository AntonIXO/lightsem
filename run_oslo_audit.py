#!/usr/bin/env python
"""Stage 0b, part two — the predictability audit on FREE dancing to EDM.

AIST++ answered "does audio determine movement" with "no, genre does", and the
reason was its design: choreographed dance to genre-matched music. This runs the
same questions on the Oslo club-dancing corpus, where ~29 people moved freely to
the same DJ mix, which is both the condition the music-induced-movement
literature was measured under and the domain the product ships into.

Four measurements:
  1. reliability — do dancers hearing the same music move together at all
  2. structure   — does movement rise at drops and collapse at breakdowns
  3. audio -> movement — held-out-time ridge on causal acoustic features
  4. locking     — is vertical bounce periodic at the track's tactus
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lightsem.audio.online import ema
from lightsem.data.oslo import (GROUPS, descriptors, load_audio_features,
                                load_segments, load_session, load_tracks)

OUT = Path("/root/lightsem/outputs/stage0b")
FPS = 40.0
TAUS = (0.5, 2.0, 8.0)
AUD = ("rms_dbfs", "spectral_centroid_hz", "spectral_bandwidth_hz", "rolloff85_hz",
       "spectral_flatness", "zcr", "bass_frac_below150hz", "onset_strength",
       "pulse_clarity_w8", "tempo_bpm_w8")


def stack_speed(desc: dict, gs: list[str]) -> np.ndarray:
    """Every dancer on one mix as columns, on the groups' common clock.

    The two groups bound to mix 1 are separate Qualisys takes and need not have
    the same frame count, so the concatenation has to be truncated first.
    """
    n = min(len(desc[g]["speed"]) for g in gs)
    return np.concatenate([desc[g]["speed"][:n] for g in gs], axis=1)


def nanmean_cols(x):
    with np.errstate(invalid="ignore"):
        return np.nanmean(x, axis=1)


def main():
    rng = np.random.default_rng(0)
    sessions = {g: load_session(g) for g in GROUPS}
    desc = {g: descriptors(s) for g, s in sessions.items()}
    tracks = load_tracks()
    segs = load_segments()
    res: dict = {}

    # ---------------- 1. reliability of the shared component ----------------
    print("=== 1. do dancers hearing the same music move together? ===")
    rel = {}
    for mix in (1, 2):
        gs = [g for g, (_, m) in GROUPS.items() if m == mix]
        cols, ids = [], []
        for g in gs:
            sp = desc[g]["speed"]
            n = min(len(sp), min(len(desc[x]["speed"]) for x in gs))
            for p in range(sp.shape[1]):
                cols.append(sp[:n, p])
                ids.append(f"{g}{p}")
        M = np.array(cols)                    # (P, T)
        for tau in TAUS:
            S = np.array([ema(np.nan_to_num(m, nan=np.nanmean(m)), FPS, tau) for m in M])
            order = rng.permutation(len(S))
            a, b = S[order[::2]].mean(0), S[order[1::2]].mean(0)
            r = float(np.corrcoef(a, b)[0, 1])
            # pairwise inter-dancer correlation, the honest per-person number
            pw = [float(np.corrcoef(S[i], S[j])[0, 1])
                  for i in range(len(S)) for j in range(i + 1, len(S))]
            rel[f"mix{mix}_tau{tau}"] = {"split_half_r": r, "n_dancers": len(S),
                                         "mean_pairwise_r": float(np.mean(pw))}
            print(f"  mix {mix}, tau {tau:4.1f}s, {len(S):2d} dancers: "
                  f"split-half r = {r:.3f}   mean pairwise r = {np.mean(pw):+.3f}")
    res["reliability"] = rel

    # ---------------- 2. structure response ---------------------------------
    print("\n=== 2. does the crowd move more at the drop? ===")
    struct = []
    for t in tracks:
        if not t["drop_s"]:
            continue
        mix = int(t["dj_mix"])
        drop, bd, bu = float(t["drop_s"]), float(t["breakdown_s"]), float(t["buildup_s"])
        gs = [g for g, (_, m) in GROUPS.items() if m == mix]
        per_dancer = []
        for g in gs:
            sp = desc[g]["speed"]
            for p in range(sp.shape[1]):
                v = sp[:, p]

                def win(a, b):
                    i0, i1 = int(a * FPS), int(b * FPS)
                    seg = v[max(i0, 0):min(i1, len(v))]
                    return float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan

                per_dancer.append({
                    "breakdown": win(bd, bu), "buildup": win(bu, drop),
                    "drop": win(drop, drop + 20), "core": win(30, 60),
                })
        arr = {k: np.array([d[k] for d in per_dancer]) for k in per_dancer[0]}
        ok = np.isfinite(arr["drop"]) & np.isfinite(arr["breakdown"])
        d_vs_b = arr["drop"][ok] - arr["breakdown"][ok]
        boot = np.array([d_vs_b[rng.integers(0, ok.sum(), ok.sum())].mean()
                         for _ in range(4000)])
        cohen = float(d_vs_b.mean() / (d_vs_b.std() + 1e-9))
        rec = {"track": t["title"], "mix": mix, "n_dancers": int(ok.sum()),
               "drop_mm_s": float(arr["drop"][ok].mean()),
               "breakdown_mm_s": float(arr["breakdown"][ok].mean()),
               "buildup_mm_s": float(np.nanmean(arr["buildup"])),
               "drop_minus_breakdown": float(d_vs_b.mean()),
               "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               "cohens_d": cohen,
               "ratio": float(arr["drop"][ok].mean() / max(arr["breakdown"][ok].mean(), 1e-9))}
        struct.append(rec)
        print(f"  {t['title']:12s} mix{mix}  n={rec['n_dancers']:2d}  breakdown "
              f"{rec['breakdown_mm_s']:6.1f}  build {rec['buildup_mm_s']:6.1f}  drop "
              f"{rec['drop_mm_s']:6.1f} mm/s   drop-breakdown {rec['drop_minus_breakdown']:+6.1f} "
              f"CI [{rec['ci95'][0]:+.1f},{rec['ci95'][1]:+.1f}]  d={cohen:+.2f}  "
              f"ratio {rec['ratio']:.2f}x")
    res["structure"] = struct

    # ---------------- 3. audio -> shared movement ---------------------------
    print("\n=== 3. can causal audio predict the shared movement? (held-out time) ===")
    fit = {}
    for mix in (1, 2):
        af = load_audio_features(mix)
        gs = [g for g, (_, m) in GROUPS.items() if m == mix]
        sp = stack_speed(desc, gs)
        n = min(len(sp), len(af["t_s"]))
        shared = nanmean_cols(sp[:n])
        shared = np.nan_to_num(shared, nan=float(np.nanmean(shared)))
        cols = []
        for c in AUD:
            v = np.nan_to_num(af[c][:n], nan=0.0)
            for tau in TAUS:
                cols.append(ema(v, FPS, tau))
        X = np.stack(cols, 1)
        for tau in TAUS:
            y = ema(shared, FPS, tau)
            n_blk = 8
            edges = np.linspace(0, n, n_blk + 1).astype(int)
            r2s = []
            for b in range(n_blk):
                te = np.zeros(n, bool); te[edges[b]:edges[b + 1]] = True
                tr = ~te
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xtr = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
                Xte = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
                ym = y[tr].mean()
                best, bestr2 = None, -np.inf
                inner = np.zeros(tr.sum(), bool); inner[::5] = True
                for a in (1e1, 1e2, 1e3, 1e4, 1e5):
                    W = np.linalg.solve(Xtr[~inner].T @ Xtr[~inner] + a * np.eye(Xtr.shape[1]),
                                        Xtr[~inner].T @ (y[tr][~inner] - ym))
                    p = Xtr[inner] @ W + ym
                    r2i = 1 - ((y[tr][inner] - p) ** 2).sum() / max(
                        ((y[tr][inner] - y[tr][inner].mean()) ** 2).sum(), 1e-9)
                    if r2i > bestr2:
                        bestr2, best = r2i, a
                W = np.linalg.solve(Xtr.T @ Xtr + best * np.eye(Xtr.shape[1]),
                                    Xtr.T @ (y[tr] - ym))
                pred = Xte @ W + ym
                r2s.append(1 - ((y[te] - pred) ** 2).sum() / max(
                    ((y[te] - y[te].mean()) ** 2).sum(), 1e-9))
            r2s = np.array(r2s)
            bs = np.array([r2s[rng.integers(0, len(r2s), len(r2s))].mean() for _ in range(4000)])
            fit[f"mix{mix}_tau{tau}"] = {
                "r2": float(r2s.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "per_block": [float(x) for x in r2s]}
            print(f"  mix {mix}, tau {tau:4.1f}s: held-out-time R2 = {r2s.mean():+.3f}  "
                  f"CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    res["audio_to_movement"] = fit

    # cross-group: train on one group, test on the OTHER group hearing the same mix
    print("\n  cross-dancer check (fit on group A, predict group C, same mix):")
    af = load_audio_features(1)
    spA, spC = desc["A"]["speed"], desc["C"]["speed"]
    n = min(len(spA), len(spC), len(af["t_s"]))
    yA = np.nan_to_num(nanmean_cols(spA[:n]), nan=0.0)
    yC = np.nan_to_num(nanmean_cols(spC[:n]), nan=0.0)
    cols = []
    for c in AUD:
        v = np.nan_to_num(af[c][:n], nan=0.0)
        for tau in TAUS:
            cols.append(ema(v, FPS, tau))
    X = np.stack(cols, 1)
    xg = []
    for tau in TAUS:
        a, b = ema(yA, FPS, tau), ema(yC, FPS, tau)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xn = np.c_[(X - mu) / sd, np.ones(n)]
        W = np.linalg.solve(Xn.T @ Xn + 1e3 * np.eye(Xn.shape[1]), Xn.T @ (a - a.mean()))
        pred = Xn @ W + a.mean()
        r2 = 1 - ((b - pred) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        r_ac = float(np.corrcoef(a, b)[0, 1])
        xg.append({"tau": tau, "r2_cross_group": float(r2), "r_group_A_vs_C": r_ac})
        print(f"    tau {tau:4.1f}s: R2(A-fit -> C) = {r2:+.3f}   (A vs C correlation "
              f"{r_ac:+.3f} is the ceiling)")
    res["cross_group"] = xg

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oslo_audit.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT/'oslo_audit.json'}")




def per_track_fit():
    """Held-out-TRACK prediction of the shared movement curve, with the causal
    gain control a live system would actually have.

    The first attempt regressed absolute movement on absolute loudness across
    contiguous time blocks and scored negative R2 everywhere. That is a
    modelling failure rather than a finding: each track in a DJ mix sits at its
    own mix level, so a linear model asked to extrapolate the LEVEL across
    tracks it has never heard fails on the offset before it gets to the shape.
    A real lighting desk does not work in absolute units either — it has an
    automatic gain. So both sides are normalised by a causal trailing 30 s mean
    and standard deviation, and the question becomes the one that matters:
    within a track, does audio predict the SHAPE of the crowd's movement?
    """
    rng = np.random.default_rng(0)
    sessions = {g: load_session(g) for g in GROUPS}
    desc = {g: descriptors(s) for g, s in sessions.items()}
    tracks = load_tracks()
    out = []
    print("\n=== 3b. held-out TRACK, causally gain-normalised ===")
    for mix in (1, 2):
        af = load_audio_features(mix)
        gs = [g for g, (_, m) in GROUPS.items() if m == mix]
        sp = stack_speed(desc, gs)
        n = min(len(sp), len(af["t_s"]))
        shared = np.nan_to_num(nanmean_cols(sp[:n]), nan=0.0)
        tks = [t for t in tracks if int(t["dj_mix"]) == mix]
        tid = np.full(n, -1)
        for i, t in enumerate(tks):
            a, b = int(float(t["start_s"]) * FPS), int(float(t["end_s"]) * FPS)
            tid[a:min(b, n)] = i

        def cnorm(v, win_s=30.0):
            """Causal z-score against a trailing window — a live AGC."""
            m = ema(v, FPS, win_s)
            s = np.sqrt(np.maximum(ema((v - m) ** 2, FPS, win_s), 1e-9))
            return (v - m) / s

        cols = []
        for c in AUD:
            v = np.nan_to_num(af[c][:n], nan=0.0)
            vn = cnorm(v)
            for tau in TAUS:
                cols.append(ema(vn, FPS, tau))
        X = np.stack(cols, 1)
        for tau in TAUS:
            y = cnorm(ema(shared, FPS, tau))
            r2s, rs = [], []
            for i in range(len(tks)):
                te = tid == i
                tr = (tid >= 0) & (~te)
                if te.sum() < 400 or tr.sum() < 400:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xtr = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
                Xte = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
                ym = y[tr].mean()
                W = np.linalg.solve(Xtr.T @ Xtr + 1e3 * np.eye(Xtr.shape[1]),
                                    Xtr.T @ (y[tr] - ym))
                pred = Xte @ W + ym
                r2s.append(1 - ((y[te] - pred) ** 2).sum() /
                           max(((y[te] - y[te].mean()) ** 2).sum(), 1e-9))
                rs.append(float(np.corrcoef(y[te], pred)[0, 1]))
            r2s, rs = np.array(r2s), np.array(rs)
            bs = np.array([r2s[rng.integers(0, len(r2s), len(r2s))].mean()
                           for _ in range(4000)])
            out.append({"mix": mix, "tau": tau, "r2": float(r2s.mean()),
                        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "r": float(rs.mean()), "n_tracks": int(len(r2s))})
            print(f"  mix {mix}, tau {tau:4.1f}s: held-out-track R2 = {r2s.mean():+.3f} "
                  f"CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  "
                  f"r = {rs.mean():+.3f}  ({len(r2s)} tracks)")

    p = OUT / "oslo_audit.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["held_out_track"] = out
    p.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
    per_track_fit()
