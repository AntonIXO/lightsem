#!/usr/bin/env python
"""H-G: do source-separated stems add lighting-relevant information the mix bus collapses?
(PLAN.md 9.7, gates G9.7-stems / G9.7-split / G9.7-causal)

Everything is a song-split ridge against the SAME bus the operator head and exp-016 used, with
the family under test appended, so a gain can only come from the source split:

  events   Just Dance visual scene-change events (exp-016 recipe, 10 Hz): per-song AP of the
           bus (audio_feats pooling, exactly exp-016's ridge) vs bus + {drums, bass, other,
           vocals, compact 4-stem, full 4-stem, vocal activity, bar phase}, with a SHUFFLED-DONOR
           control (another song's stem features appended: any gain from width alone shows here)
           and the time-shift null (target shifted +-2/4/8 s) for every family.
  L = 0    the same on the songs that also have causal (trailing-window) stems: whole-track vs
           causal stems on identical songs and folds.
  coach    the head/hands split (run_justdance_audit recipe, 40 fps, 2 s envelopes): held-out-
           song R2 of hand / head / foot speed, footfall rate, bounce, QoM from bus scalars vs
           bus + per-stem scalars; a SOURCE-SPECIFIC split means drums move feet/head more than
           hands and vocals/other move hands more than feet.

  bash run.sh run_stage9_stems.py
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import ema  # noqa: E402
from lightsem.motion.io import load_bus, load_descriptors  # noqa: E402
from probe_event_target import audio_feats, average_precision, event_train, hazard_feats, target  # noqa: E402

CACHE = Path("/mnt/lightsem-data/cache")
OUT = Path("/root/lightsem/outputs/stage9")
HZ, FPS = 10.0, 40.0
SHIFTS = (-8.0, -4.0, -2.0, 2.0, 4.0, 8.0)
STEMS = ("drums", "bass", "other", "vocals")
COMPACT = ("rms", "onset", "flux", "flux_lf", "flux_hf")
AUD = ("rms_db", "flux", "flux_lf", "flux_hf", "onset", "centroid", "flatness", "rolloff",
       "percussive_ratio", "pulse_clarity", "energy_lf", "energy_hf", "beat_accent")
TAUS = (0.5, 2.0, 8.0)
CH = ("hand_speed", "head_speed", "foot_speed", "footfall_rate", "vert_bounce_amp", "qom")


def cnorm(v, fps, win_s=30.0):
    m = ema(v, fps, win_s)
    s = np.sqrt(np.maximum(ema((v - m) ** 2, fps, win_s), 1e-9))
    return (v - m) / s


def shift_target(t, sh):
    out = np.zeros_like(t)
    if sh >= 0:
        out[:len(t) - sh] = t[sh:] if sh else t
    else:
        out[-sh:] = t[:len(t) + sh]
    return out


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    return mu, sd, np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean())), y.mean()


def ridge_apply(p, X):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


def folds_of(keys, k=5, seed=0):
    songs = sorted(set(keys)); rng = np.random.default_rng(seed)
    f = {s: i % k for i, s in enumerate(rng.permutation(songs))}
    return np.array([f[key] for key in keys])


def fit_to_len(a: np.ndarray, T: int) -> np.ndarray:
    if len(a) >= T:
        return a[:T]
    reps = int(np.ceil((T - len(a)) / len(a)))
    return np.concatenate([a, np.tile(a, (reps, 1) if a.ndim == 2 else reps)[: T - len(a)]])


def sel(names, pred):
    return [i for i, n in enumerate(names) if pred(n)]


def event_block(songs, keys, fold, families, lam, tag, shifts=True, with_haz=False):
    """families: name -> per-song feature builder (song dict -> (T, d))."""
    res = {}
    aps = {f: [] for f in families}; ap_sh = {f: {s: [] for s in SHIFTS} for f in families}
    for f in range(5):
        tr = [i for i in range(len(songs)) if fold[i] != f]; te = [i for i in range(len(songs)) if fold[i] == f]
        for fam, build in families.items():
            X = np.concatenate([build(songs[i]) for i in tr]); Y = np.concatenate([songs[i]["t"] for i in tr])
            p = ridge_fit(X, Y, lam)
            for i in te:
                t = songs[i]["t"]
                if t.sum() < 3:
                    continue
                s = ridge_apply(p, build(songs[i]))
                aps[fam].append(average_precision(s, t))
                if shifts:
                    for sh in SHIFTS:
                        ts = shift_target(t, int(round(sh * HZ)))
                        if ts.sum() >= 3:
                            ap_sh[fam][sh].append(average_precision(s, ts))
    rate = float(np.mean([s["t"].mean() for s in songs]))
    rng = np.random.default_rng(0)
    base = np.array(aps["bus"])
    for fam in families:
        v = np.array(aps[fam]); d = v - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        shm = float(np.mean([np.mean(ap_sh[fam][s]) for s in SHIFTS])) if shifts else float("nan")
        res[fam] = {"ap": float(v.mean()), "lift": float((v / rate).mean()), "delta_ap": float(d.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "ap_shift_mean": shm, "ap_minus_shift": float(v.mean() - shm), "n": int(len(v))}
        print(f"  [{tag}] {fam:22s} AP {v.mean():.4f} lift {(v/rate).mean():.2f}  dAP vs bus {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]"
              + (f"  shift-null {shm:.4f} (audio-shift {v.mean()-shm:+.3f})" if shifts else ""), flush=True)
    res["_rate"] = rate
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--lam-coach", type=float, default=1e3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    # ------------------------------------------------------------------ load
    songs = []
    for p in sorted((CACHE / "justdance_stem_feats").glob("*.npz")):
        yt = p.stem
        vp = CACHE / "justdance_visual" / f"{yt}.npz"; bp = CACHE / "justdance_audio" / f"{yt}.npz"
        if not (vp.exists() and bp.exists()):
            continue
        v = np.load(vp, allow_pickle=False); sf = np.load(p, allow_pickle=False)
        x, y = v["x"].astype(np.float32), v["y"].astype(np.float32)
        if len(x) < 320 or not np.isfinite(x).all():
            continue
        tr, _ = event_train(y)
        bus = load_bus(bp)
        T = len(x); t10 = np.arange(T) / HZ
        bar = np.interp(t10, bus.times, bus.channels["bar_phase"])
        n10 = [str(n) for n in sf["names10"]]
        x10 = sf["x10"].astype(np.float32)[:T]
        if len(x10) < T:
            x10 = np.concatenate([x10, np.repeat(x10[-1:], T - len(x10), 0)])
        cp = CACHE / "justdance_stem_feats_causal" / f"{yt}.npz"
        xc = None
        if cp.exists():
            xc = np.load(cp, allow_pickle=False)["x10"].astype(np.float32)[:T]
            if len(xc) < T:
                xc = np.concatenate([xc, np.repeat(xc[-1:], T - len(xc), 0)])
        songs.append({"key": yt, "xa": audio_feats(x), "x10": x10, "xc": xc, "n10": n10, "haz": hazard_feats(tr).astype(np.float32),
                      "t": target(tr).astype(np.float32),
                      "bar": np.stack([np.sin(2 * np.pi * bar), np.cos(2 * np.pi * bar), (1 - bar) ** 2.5], 1).astype(np.float32)})
    if a.limit:
        songs = songs[:a.limit]
    keys = [s["key"] for s in songs]; fold = folds_of(keys)
    n10 = songs[0]["n10"]
    idx = {st: sel(n10, lambda n, st=st: n.startswith(st + "_")) for st in STEMS}
    idx["compact"] = sel(n10, lambda n: any(n.startswith(f"{st}_{c}_ema") for st in STEMS for c in COMPACT))
    idx["full"] = sel(n10, lambda n: not n.startswith("vocal_act"))
    idx["vocal_act"] = sel(n10, lambda n: n.startswith("vocal_act"))
    rng = np.random.default_rng(1); perm = rng.permutation(len(songs))
    donor = {songs[i]["key"]: songs[perm[i]] if perm[i] != i else songs[(perm[i] + 1) % len(songs)] for i in range(len(songs))}
    print(f"{len(songs)} songs, {sum(s['xc'] is not None for s in songs)} with causal stems, {time.time()-t0:.0f}s to load", flush=True)
    res = {"n_songs": len(songs), "lam": a.lam, "families": {k: len(v) for k, v in idx.items()}}

    def fam(cols, extra=None):
        def build(s):
            parts = [s["xa"]]
            if cols is not None:
                parts.append(s["x10"][:, cols])
            if extra == "bar":
                parts.append(s["bar"])
            if extra == "donor":
                parts.append(fit_to_len(donor[s["key"]]["x10"][:, idx["compact"]], len(s["xa"])))
            if extra == "haz":
                parts.append(s["haz"])
            return np.concatenate(parts, 1)
        return build
    families = {"bus": fam(None), "bus+drums": fam(idx["drums"]), "bus+bass": fam(idx["bass"]), "bus+other": fam(idx["other"]),
                "bus+vocals": fam(idx["vocals"]), "bus+compact4": fam(idx["compact"]), "bus+full4": fam(idx["full"]),
                "bus+vocal_act": fam(idx["vocal_act"]), "bus+bar_phase": fam(None, "bar"), "bus+donor_compact4": fam(None, "donor"),
                "stems_compact4_only": lambda s: s["x10"][:, idx["compact"]]}
    print("\n=== events (JD visual scene changes), audio-only ===", flush=True)
    res["events"] = event_block(songs, keys, fold, families, a.lam, "audio")
    fam_h = {"bus": fam(None, "haz"), "bus+compact4": lambda s: np.concatenate([s["xa"], s["x10"][:, idx["compact"]], s["haz"]], 1),
             "bus+full4": lambda s: np.concatenate([s["xa"], s["x10"][:, idx["full"]], s["haz"]], 1),
             "bus+donor_compact4": lambda s: np.concatenate([fam(None, "donor")(s), s["haz"]], 1)}
    print("\n=== events, with the rhythm hazard (exp-016's headline form) ===", flush=True)
    res["events_with_hazard"] = event_block(songs, keys, fold, fam_h, a.lam, "hazard", shifts=False)

    # ------------------------------------------------------------------ L = 0 counterpart
    sub = [s for s in songs if s["xc"] is not None]
    if len(sub) >= 20:
        fsub = folds_of([s["key"] for s in sub])
        fam_c = {"bus": fam(None), "bus+compact4_whole": fam(idx["compact"]),
                 "bus+compact4_causal": lambda s: np.concatenate([s["xa"], s["xc"][:, idx["compact"]]], 1),
                 "bus+full4_whole": fam(idx["full"]),
                 "bus+full4_causal": lambda s: np.concatenate([s["xa"], s["xc"][:, idx["full"]]], 1)}
        print(f"\n=== L = 0: whole-track vs causal (trailing-window) stems on {len(sub)} songs ===", flush=True)
        res["causal_subset"] = event_block(sub, [s["key"] for s in sub], fsub, fam_c, a.lam, "L=0", shifts=False)
        res["causal_subset"]["n_songs"] = len(sub)

    # ------------------------------------------------------------------ coach split at 40 fps
    print("\n=== coach split (held-out-song R2 at 2 s) ===", flush=True)
    rows = json.loads((CACHE / "justdance_motion_index.json").read_text())
    rows = [r for r in rows if r.get("coverage", 0) >= 0.8 and (CACHE / "justdance_stem_feats" / f"{r['music_id']}.npz").exists()]
    ents, bus_cache, sf_cache = [], {}, {}
    for r in rows:
        yt = r["music_id"]
        if yt not in bus_cache:
            bus_cache[yt] = load_bus(CACHE / "justdance_audio" / f"{yt}.npz")
            sf_cache[yt] = np.load(CACHE / "justdance_stem_feats" / f"{yt}.npz", allow_pickle=False)
        d = load_descriptors(CACHE / "justdance_motion" / f"{r['name']}.npz")
        bus = bus_cache[yt]; sf = sf_cache[yt]
        T = min(len(d.times), len(bus.times), len(sf["x40"]))
        if T < 40 * FPS:
            continue
        cols = []
        for c in AUD:
            vn = cnorm(np.nan_to_num(bus.channels[c][:T].astype(np.float64)), FPS)
            cols += [ema(vn, FPS, tau) for tau in TAUS]
        Xb = np.stack(cols, 1).astype(np.float32)
        n40 = [str(n) for n in sf["names40"]]
        st_cols = {}
        for st in STEMS:
            cc = []
            for c in ("rms", "onset", "flux_lf", "flux_hf"):
                j = n40.index(f"{st}_{c}")
                vn = cnorm(np.nan_to_num(sf["x40"][:T, j].astype(np.float64)), FPS)
                cc += [ema(vn, FPS, tau) for tau in TAUS]
            st_cols[st] = np.stack(cc, 1).astype(np.float32)
        Y = {ch: cnorm(ema(np.nan_to_num(d.channels[ch][:T].astype(np.float64)), FPS, 2.0), FPS).astype(np.float32) for ch in CH}
        ents.append({"key": yt, "Xb": Xb, "st": st_cols, "Y": Y})
    ekeys = [e["key"] for e in ents]; efold = folds_of(ekeys)
    rng = np.random.default_rng(2); perm = rng.permutation(len(ents))
    edonor = {i: ents[perm[i]] if perm[i] != i else ents[(perm[i] + 1) % len(ents)] for i in range(len(ents))}
    cfam = {"bus": lambda i: ents[i]["Xb"],
            **{f"bus+{st}": (lambda i, st=st: np.concatenate([ents[i]["Xb"], ents[i]["st"][st]], 1)) for st in STEMS},
            "bus+all4": lambda i: np.concatenate([ents[i]["Xb"]] + [ents[i]["st"][st] for st in STEMS], 1),
            "bus+donor4": lambda i: np.concatenate([ents[i]["Xb"]] + [fit_to_len(edonor[i]["st"][st], len(ents[i]["Xb"])) for st in STEMS], 1)}
    coach = {}
    for ch in CH:
        coach[ch] = {}
        r2 = {f: [] for f in cfam}
        for f in range(5):
            tr = [i for i in range(len(ents)) if efold[i] != f]; te = [i for i in range(len(ents)) if efold[i] == f]
            Y = np.concatenate([ents[i]["Y"][ch] for i in tr])
            for name, build in cfam.items():
                p = ridge_fit(np.concatenate([build(i) for i in tr]), Y, a.lam_coach)
                for i in te:
                    y = ents[i]["Y"][ch]; pred = ridge_apply(p, build(i))
                    r2[name].append(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
        base = np.array(r2["bus"])
        for name in cfam:
            v = np.array(r2[name]); d = v - base
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
            coach[ch][name] = {"r2": float(v.mean()), "delta_r2": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print(f"  {ch:16s} bus R2 {coach[ch]['bus']['r2']:+.3f}  " + "  ".join(f"{n[4:]} {coach[ch][n]['delta_r2']:+.3f}" for n in cfam if n != "bus"), flush=True)
    res["coach_split"] = {"n_entries": len(ents), "n_songs": len(set(ekeys)), "per_channel": coach}
    # source-specific split statistic
    dr = lambda ch, n: coach[ch][n]["delta_r2"]  # noqa: E731
    split = {"drums_feet_head_minus_hands": float(np.mean([dr("foot_speed", "bus+drums"), dr("footfall_rate", "bus+drums"), dr("head_speed", "bus+drums")]) - dr("hand_speed", "bus+drums")),
             "vocals_other_hands_minus_feet": float(np.mean([dr("hand_speed", "bus+vocals"), dr("hand_speed", "bus+other")]) - np.mean([dr("foot_speed", "bus+vocals"), dr("foot_speed", "bus+other")])),
             "hand_speed_gain_vocals_other": float(np.mean([dr("hand_speed", "bus+vocals"), dr("hand_speed", "bus+other")])),
             "hand_speed_gain_all4": dr("hand_speed", "bus+all4"), "hand_speed_gain_donor": dr("hand_speed", "bus+donor4")}
    res["coach_split"]["split_stats"] = split
    print("  split stats", json.dumps({k: round(v, 4) for k, v in split.items()}), flush=True)

    # ------------------------------------------------------------------ gates
    ev = res["events"]
    best = max(("bus+drums", "bus+bass", "bus+other", "bus+vocals", "bus+compact4", "bus+full4", "bus+vocal_act"), key=lambda k: ev[k]["delta_ap"])
    donor_gain = ev["bus+donor_compact4"]["delta_ap"]
    g_stems = {"best_family": best, "delta_ap": ev[best]["delta_ap"], "ci95": ev[best]["ci95"], "donor_delta_ap": donor_gain,
               "shift_flat": bool(abs(ev[best]["ap_shift_mean"] - ev["bus"]["ap_shift_mean"]) < 0.01),
               "passed": bool(ev[best]["delta_ap"] >= 0.02 and ev[best]["ci95"][0] > 0 and ev[best]["delta_ap"] - donor_gain >= 0.02)}
    g_split = {"stats": split, "passed": bool(split["drums_feet_head_minus_hands"] > 0.01 and split["vocals_other_hands_minus_feet"] > 0.01
                                             and split["hand_speed_gain_vocals_other"] >= 0.03 and split["hand_speed_gain_vocals_other"] > split["hand_speed_gain_donor"] + 0.01)}
    g_causal = {"reported": "causal_subset" in res, "passed": "causal_subset" in res}
    if "causal_subset" in res:
        cs = res["causal_subset"]
        g_causal.update({"bus": cs["bus"]["ap"], "whole": cs["bus+compact4_whole"]["ap"], "causal": cs["bus+compact4_causal"]["ap"]})
    res["gate"] = {"G9.7-stems": g_stems, "G9.7-split": g_split, "G9.7-causal": g_causal}
    res["bar_phase"] = {"delta_ap": ev["bus+bar_phase"]["delta_ap"], "ci95": ev["bus+bar_phase"]["ci95"], "kill_below": 0.02}
    print("\nGATE", json.dumps(res["gate"], indent=1), flush=True)
    res["elapsed_s"] = time.time() - t0
    (OUT / "hg_stems.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT/'hg_stems.json'} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
