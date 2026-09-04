#!/usr/bin/env python
"""Stage 8.2 -- does audio predict the coach's movement on held-out SONGS?

The Oslo recipe (run_oslo_audit.per_track_fit), applied per coach: both sides
causally gain-normalised against a trailing 30 s window (a live AGC), audio
scalars of the causal bus at three EMA timescales, ridge fitted on every other
song and scored on the held-out song, so the number is within-song SHAPE, not
between-song LEVEL -- the between-song part is exactly what genre explained on
AIST++ and it is reported separately as the level-only baseline.

Also: inter-coach agreement on multi-coach routines (the split-half of Oslo's
crowd, with two to four dancers instead of nineteen), and the vertical-bounce
period against the beat grid across the corpus (the 8.3 sync check).

  bash run.sh run_justdance_audit.py [--limit N] [--tag jd]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import ema  # noqa: E402
from lightsem.motion.io import load_bus, load_descriptors  # noqa: E402

CACHE = Path("/mnt/lightsem-data/cache")
MOTION = CACHE / "justdance_motion"
INDEX = CACHE / "justdance_motion_index.json"
BUS = CACHE / "justdance_audio"
OUT = Path("/root/lightsem/outputs/stage8")
FPS = 40.0
TAUS = (0.5, 2.0, 8.0)
AUD = ("rms_db", "flux", "flux_lf", "flux_hf", "onset", "centroid", "flatness", "rolloff",
       "percussive_ratio", "pulse_clarity", "energy_lf", "energy_hf", "beat_accent")
CH = ("qom", "vert_bounce_amp", "head_speed", "com_speed", "spread", "hand_speed")


def cnorm(v, win_s=30.0):
    m = ema(v, FPS, win_s)
    s = np.sqrt(np.maximum(ema((v - m) ** 2, FPS, win_s), 1e-9))
    return (v - m) / s


def load_rows(limit):
    rows = json.loads(INDEX.read_text())
    rows = [r for r in rows if r.get("coverage", 0) >= 0.8]
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="jd")
    ap.add_argument("--min-songs", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.limit)
    print(f"{len(rows)} coach entries", flush=True)
    t0 = time.time()
    # ---- assemble per-entry feature / target matrices
    X_all, Y_all, song_all, ent_all = [], [], [], []
    per_song_coaches: dict[str, list[int]] = {}
    bus_cache: dict[str, object] = {}
    for ei, r in enumerate(rows):
        d = load_descriptors(MOTION / f"{r['name']}.npz")
        yt = r["music_id"]
        if yt not in bus_cache:
            bus_cache[yt] = load_bus(BUS / f"{yt}.npz")
        bus = bus_cache[yt]
        T = min(len(d.times), len(bus.times))
        if T < 40 * FPS:
            continue
        cols = []
        for c in AUD:
            v = np.nan_to_num(bus.channels[c][:T].astype(np.float64), nan=0.0)
            vn = cnorm(v)
            for tau in TAUS:
                cols.append(ema(vn, FPS, tau))
        X = np.stack(cols, 1)
        Y = {}
        for ch in CH:
            v = np.nan_to_num(d.channels[ch][:T].astype(np.float64), nan=0.0)
            Y[ch] = {tau: cnorm(ema(v, FPS, tau)) for tau in TAUS}
        X_all.append(X); Y_all.append(Y); song_all.append(r["song_key"]); ent_all.append(r["name"])
        per_song_coaches.setdefault(yt, []).append(len(X_all) - 1)
    songs = sorted(set(song_all))
    print(f"{len(X_all)} entries, {len(songs)} songs, {time.time()-t0:.0f}s to load", flush=True)
    if len(songs) < a.min_songs:
        print("too few songs for a held-out fit"); return
    rng = np.random.default_rng(0)
    # ---- held-out-song ridge, per channel and tau
    song_idx = {s: i for i, s in enumerate(songs)}
    sid = np.array([song_idx[s] for s in song_all])
    n_ent = len(X_all)
    lens = np.array([len(x) for x in X_all])
    Xc = np.concatenate(X_all)
    ent_of_frame = np.repeat(np.arange(n_ent), lens)
    song_of_frame = sid[ent_of_frame]
    res = {"n_entries": n_ent, "n_songs": len(songs), "taus": TAUS, "channels": CH, "held_out_song": {}}
    # 5-fold by song, every song held out once
    folds = np.array_split(rng.permutation(len(songs)), 5)
    for ch in CH:
        for tau in TAUS:
            y = np.concatenate([Y[ch][tau] for Y in Y_all])
            r2_song, r_song = {}, {}
            for f in folds:
                te = np.isin(song_of_frame, f); tr = ~te
                mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-9
                Xtr = np.c_[(Xc[tr] - mu) / sd, np.ones(tr.sum())]
                ym = y[tr].mean()
                W = np.linalg.solve(Xtr.T @ Xtr + 1e3 * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ym))
                for s_i in f:
                    m = song_of_frame == s_i
                    if m.sum() < 400:
                        continue
                    Xte = np.c_[(Xc[m] - mu) / sd, np.ones(m.sum())]
                    pred = Xte @ W + ym
                    r2_song[int(s_i)] = 1 - ((y[m] - pred) ** 2).sum() / max(((y[m] - y[m].mean()) ** 2).sum(), 1e-9)
                    r_song[int(s_i)] = float(np.corrcoef(y[m], pred)[0, 1])
            r2s = np.array(list(r2_song.values())); rs = np.array(list(r_song.values()))
            bs = np.array([r2s[rng.integers(0, len(r2s), len(r2s))].mean() for _ in range(4000)])
            res["held_out_song"][f"{ch}@{tau}"] = {
                "r2": float(r2s.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "r2_median": float(np.median(r2s)), "r": float(rs.mean()), "n_songs": int(len(r2s))}
            print(f"  {ch:16s} tau {tau:4.1f}s  held-out-song R2 {r2s.mean():+.3f} "
                  f"CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  r {rs.mean():+.3f}  "
                  f"({len(r2s)} songs)", flush=True)
    # ---- inter-coach agreement on multi-coach routines (shared vs personal)
    multi = {yt: idx for yt, idx in per_song_coaches.items() if len(idx) >= 2}
    agree = {}
    for tau in TAUS:
        for ch in ("qom", "vert_bounce_amp", "head_speed"):
            rs = []
            for yt, idx in multi.items():
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        a_, b_ = Y_all[idx[i]][ch][tau], Y_all[idx[j]][ch][tau]
                        n = min(len(a_), len(b_))
                        rs.append(float(np.corrcoef(a_[:n], b_[:n])[0, 1]))
            if rs:
                agree[f"{ch}@{tau}"] = {"mean_r": float(np.mean(rs)), "median_r": float(np.median(rs)),
                                        "n_pairs": len(rs)}
    res["inter_coach"] = {"n_multi_songs": len(multi), "pairs": agree}
    if multi:
        for k, v in agree.items():
            print(f"  inter-coach {k:22s} mean r {v['mean_r']:+.3f} median {v['median_r']:+.3f} ({v['n_pairs']} pairs)")
    # ---- 8.3 sync: vertical bounce period against the beat grid
    ratios = []
    for r in rows[:n_ent]:
        pass
    vp = []
    for ei, r in enumerate(rows):
        try:
            d = load_descriptors(MOTION / f"{r['name']}.npz")
        except Exception:
            continue
        v = d.channels["vert_period_s"]; v = v[np.isfinite(v) & (v > 0.2)]
        if v.size and r.get("tempo"):
            vp.append(float(np.median(v) / (60.0 / r["tempo"])))
    if vp:
        vp = np.array(vp)
        res["bounce_period_over_beat"] = {"median": float(np.median(vp)),
                                          "frac_near_1_or_2": float(((np.abs(vp - 1) < 0.15) | (np.abs(vp - 2) < 0.3)).mean()),
                                          "n": int(len(vp))}
        print(f"  bounce period / beat: median {np.median(vp):.2f}, near 1x or 2x in "
              f"{res['bounce_period_over_beat']['frac_near_1_or_2']:.0%} of {len(vp)} entries")
    # ---- section contrast: coach movement in the loudest third of a song's
    # sections against the quietest third (the Just Dance analogue of Oslo's
    # drop/breakdown 1.68-2.53x), from the structure labels' segment energies
    STRUCT = CACHE / "justdance_structure"
    ratios = {ch: [] for ch in ("qom", "vert_bounce_amp", "head_speed")}
    for ei, r in enumerate(rows):
        sp = STRUCT / f"{r['music_id']}.npz"
        if not sp.exists():
            continue
        try:
            d = load_descriptors(MOTION / f"{r['name']}.npz")
            segs = json.loads(str(np.load(sp, allow_pickle=False)["meta"]))["segments"]
        except Exception:
            continue
        off = float(r.get("start_s", 0.0)); n = len(d.times)
        en = np.array([sg["energy"] for sg in segs])
        if len(en) < 3:
            continue
        lo_t, hi_t = np.percentile(en, [33, 67])
        for ch in ratios:
            v = np.nan_to_num(d.channels[ch][:n], nan=0.0)
            hi, lo = [], []
            for sg in segs:
                a_, b_ = int(round((sg["start_s"] - off) * FPS)), int(round((sg["end_s"] - off) * FPS))
                a_, b_ = max(a_, 0), min(b_, n)
                if b_ - a_ < FPS:
                    continue
                (hi if sg["energy"] >= hi_t else lo if sg["energy"] <= lo_t else []).append(float(v[a_:b_].mean()))
            if hi and lo and np.mean(lo) > 0:
                ratios[ch].append(float(np.mean(hi) / np.mean(lo)))
    res["section_contrast"] = {}
    for ch, rr in ratios.items():
        if len(rr) >= 5:
            rr = np.array(rr)
            bs = np.array([rr[rng.integers(0, len(rr), len(rr))].mean() for _ in range(4000)])
            res["section_contrast"][ch] = {"ratio_mean": float(rr.mean()), "ratio_median": float(np.median(rr)),
                                           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                           "n_entries": int(len(rr))}
            print(f"  section contrast {ch:16s} loud/quiet thirds: mean {rr.mean():.2f}x median {np.median(rr):.2f}x "
                  f"CI [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] ({len(rr)} entries)")
    (OUT / f"audit_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / f"audit_{a.tag}.json")


if __name__ == "__main__":
    main()
