#!/usr/bin/env python
"""Stage 0b — predictability audit (runbook rb-ls2-00b).

Which motion descriptors does audio actually determine? Every channel is
scored at four causal timescales against two references:

* **dancer identity** — predict the per-dancer mean of the target. A channel
  that audio cannot beat is a fact about the person, not about the music, and
  putting it in a regression loss teaches the model to output the corpus's
  average dancer.
* **genre mean** — audio can identify genre, so a channel explained entirely
  by genre is only weakly "musical". Reported separately, never subtracted.

Everything is causal: features are trailing exponential moving averages, and
targets are smoothed the same way, so a number here is achievable by a system
that cannot see the future. Splits are GroupKFold by music id, so a held-out
score is a score on tracks never seen.

    bash run.sh run_stage0b.py build     # cache the frame matrix
    bash run.sh run_stage0b.py fit       # ridge + baselines + CIs
    bash run.sh run_stage0b.py locking   # period locking vs a shuffled-tempo null
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from lightsem.audio.online import ema
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors

OUT = Path("/root/lightsem/outputs/stage0b")
DATA = CACHE_ROOT / "audit_dataset.npz"

TAUS = (0.1, 0.5, 2.0, 8.0)

AUDIO_BASE = (
    "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid",
    "flux_hf", "energy_lf", "energy_lmid", "energy_mid", "energy_hmid",
    "energy_hf", "onset", "centroid", "flatness", "rolloff", "percussive_ratio",
    "pulse_clarity",
)
AUDIO_RAW = ("beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase",
             "beat_period_s")

#: targets. beat_period_s is excluded (it IS audio); metric_level is kept but
#: flagged, since it is defined as a ratio against the audio's beat period.
TARGETS = (
    "qom", "com_speed", "head_speed", "foot_speed", "hand_speed", "hip_yaw_speed",
    "twist", "vert_root", "vert_bounce", "vert_bounce_amp", "vert_period_s",
    "vert_period_conf", "metric_level", "weight", "time_eff", "jerk", "flow_eff",
    "space_eff", "spread", "hand_spread", "lateral_pos", "lateral_vel",
    "travel_speed", "travel_sin", "travel_cos", "symmetry", "laterality",
    "footfall_rate", "section_energy",
)
DEFINITIONAL = {"metric_level"}
GATE_CHANNELS = ("qom", "vert_bounce_amp", "head_speed", "metric_level")


def build(limit: int | None = None) -> None:
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    if limit:
        idx = idx[:limit]
    bus_cache: dict[str, object] = {}
    Xs, Ys, meta = [], [], []
    t0 = time.time()
    for i, r in enumerate(idx):
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        mid = r["music_id"]
        if mid not in bus_cache:
            bus_cache[mid] = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        bus = bus_cache[mid]
        T = min(len(d.times), len(bus.times))
        if T < 80:
            continue
        fps = d.fps
        cols = []
        for name in AUDIO_BASE:
            v = bus.channels[name][:T]
            for tau in TAUS:
                cols.append(ema(v, fps, tau))
        for name in AUDIO_RAW:
            cols.append(bus.channels[name][:T])
        X = np.stack(cols, axis=1).astype(np.float32)
        Y = np.stack([np.stack([ema(d.channels[t][:T], fps, tau) for tau in TAUS], 1)
                      for t in TARGETS], axis=2).astype(np.float32)   # (T, n_tau, n_ch)
        Xs.append(X)
        Ys.append(Y)
        meta.append((r["music_id"], r["dancer"], r["genre"], r["choreo_id"], T))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(idx)}] {time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    groups = np.concatenate([[m[0]] * m[4] for m in meta])
    dancers = np.concatenate([[m[1]] * m[4] for m in meta])
    genres = np.concatenate([[m[2]] * m[4] for m in meta])
    feat_names = [f"{n}_ema{t}" for n in AUDIO_BASE for t in TAUS] + list(AUDIO_RAW)
    np.savez(DATA, X=X, Y=Y, groups=groups, dancers=dancers, genres=genres,
             feat_names=np.array(feat_names), targets=np.array(TARGETS),
             taus=np.array(TAUS))
    print(f"X {X.shape} Y {Y.shape} tracks {len(set(groups))} dancers {len(set(dancers))} "
          f"in {time.time()-t0:.0f}s -> {DATA}")


def _ridge_multi(Xtr, Ytr, alphas):
    """Solve ridge for several alphas at once. Returns {alpha: W}."""
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Ytr
    out = {}
    n_f = Xtr.shape[1]
    for a in alphas:
        out[a] = np.linalg.solve(XtX + a * np.eye(n_f), XtY)
    return out


def _r2(y, pred):
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def fit(n_folds: int = 5) -> dict:
    z = np.load(DATA, allow_pickle=False)
    X, Y = z["X"].astype(np.float64), z["Y"].astype(np.float64)
    groups, dancers, genres = z["groups"], z["dancers"], z["genres"]
    targets, taus = list(z["targets"]), list(z["taus"])
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(0)
    fold_of = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[g] for g in groups])
    alphas = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]

    res = {t: {str(tau): {} for tau in taus} for t in targets}
    for ti, tau in enumerate(taus):
        Yt = Y[:, ti, :]
        r2_audio = np.zeros((n_folds, len(targets)))
        r2_dancer = np.zeros((n_folds, len(targets)))
        r2_genre = np.zeros((n_folds, len(targets)))
        for f in range(n_folds):
            tr, te = fold != f, fold == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
            Xte = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
            ymu = Yt[tr].mean(0)
            Ytr = Yt[tr] - ymu
            # inner split by track to pick alpha per channel
            inner_tracks = np.array(sorted(set(groups[tr].tolist())))
            hold = set(inner_tracks[::4].tolist())
            im = np.array([g in hold for g in groups[tr]])
            Ws = _ridge_multi(Xtr[~im], Ytr[~im], alphas)
            best = {}
            for ci in range(len(targets)):
                scores = [_r2(Ytr[im][:, ci:ci+1], (Xtr[im] @ Ws[a][:, ci:ci+1]))[0]
                          for a in alphas]
                best[ci] = alphas[int(np.argmax(scores))]
            Wfull = _ridge_multi(Xtr, Ytr, sorted(set(best.values())))
            pred = np.stack([Xte @ Wfull[best[ci]][:, ci] for ci in range(len(targets))], 1) + ymu
            r2_audio[f] = _r2(Yt[te], pred)
            # dancer-identity baseline: per-dancer train mean
            dmean = {}
            for dcode in set(dancers[tr].tolist()):
                m = dancers[tr] == dcode
                dmean[dcode] = Yt[tr][m].mean(0)
            gm = Yt[tr].mean(0)
            pred_d = np.stack([dmean.get(dc, gm) for dc in dancers[te]])
            r2_dancer[f] = _r2(Yt[te], pred_d)
            gmean = {}
            for gc in set(genres[tr].tolist()):
                m = genres[tr] == gc
                gmean[gc] = Yt[tr][m].mean(0)
            pred_g = np.stack([gmean.get(gc, gm) for gc in genres[te]])
            r2_genre[f] = _r2(Yt[te], pred_g)
        for ci, t in enumerate(targets):
            a = r2_audio[:, ci]; dd = r2_dancer[:, ci]; gg = r2_genre[:, ci]
            diff = a - dd
            bs = np.array([diff[rng.integers(0, n_folds, n_folds)].mean() for _ in range(4000)])
            res[t][str(tau)] = {
                "r2_audio": float(a.mean()), "r2_audio_sd": float(a.std()),
                "r2_dancer": float(dd.mean()), "r2_genre": float(gg.mean()),
                "delta_vs_dancer": float(diff.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "beats_identity": bool(np.percentile(bs, 2.5) > 0),
            }
        print(f"tau {tau}s done", flush=True)

    # block assignment
    assign = {}
    for t in targets:
        best_tau, best = None, None
        for tau in taus:
            r = res[t][str(tau)]
            if best is None or r["delta_vs_dancer"] > best["delta_vs_dancer"]:
                best, best_tau = r, tau
        if best["beats_identity"] and best["delta_vs_dancer"] > 0.10 and best["r2_audio"] > 0.0:
            block = "A"
        elif best["r2_audio"] > 0.0 and best["beats_identity"]:
            block = "A-weak"
        else:
            block = "C-or-noise"
        assign[t] = {"block": block, "best_tau": best_tau,
                     "r2_audio": best["r2_audio"], "r2_dancer": best["r2_dancer"],
                     "delta": best["delta_vs_dancer"], "ci95": best["ci95"],
                     "definitional": t in DEFINITIONAL}
    gate = {c: assign[c] for c in GATE_CHANNELS}
    gate_pass = all(assign[c]["delta"] > 0 and assign[c]["ci95"][0] > 0
                    and assign[c]["r2_audio"] > 0.3 for c in GATE_CHANNELS
                    if c not in DEFINITIONAL)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"per_channel": res, "assignment": assign,
               "gate_channels": gate, "gate_passed": bool(gate_pass),
               "n_frames": int(X.shape[0]), "n_tracks": int(len(uniq)),
               "n_folds": n_folds}
    (OUT / "audit.json").write_text(json.dumps(payload, indent=1))
    print(f"\n{'channel':18s} {'block':11s} {'tau':>5s} {'R2audio':>8s} {'R2dancer':>9s} "
          f"{'delta':>8s}  CI")
    for t in targets:
        a = assign[t]
        flag = " (definitional)" if a["definitional"] else ""
        print(f"{t:18s} {a['block']:11s} {a['best_tau']:5.1f} {a['r2_audio']:8.3f} "
              f"{a['r2_dancer']:9.3f} {a['delta']:+8.3f}  "
              f"[{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}]{flag}")
    print(f"\ngate (QoM, vert_bounce_amp, head_speed beat identity with R2>0.3): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return payload




# ---------------------------------------------------------------------------
# the shared-component audit
# ---------------------------------------------------------------------------

def build_shared(min_clips: int = 8) -> dict:
    """Per-track, the across-dancer MEAN of every channel, and its reliability.

    Why this is the right target and the per-clip version is not. Audio is
    identical for every dancer of a track, so the best any audio-only model can
    do for one dancer is predict the across-dancer mean; the individual's
    deviation from it is unpredictable by construction. Scoring audio against a
    single dancer therefore charges it for variance no audio model could ever
    explain, and the per-clip audit above does exactly that.

    The split-half reliability of the mean is the CEILING: if averaging half the
    dancers does not predict the other half, there is no shared component to
    find and the channel is noise for our purposes.
    """
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    by_music: dict[str, list] = {}
    for r in idx:
        by_music.setdefault(r["music_id"], []).append(r)
    rng = np.random.default_rng(0)
    rows = []
    for mid, rs in sorted(by_music.items()):
        if len(rs) < min_clips:
            continue
        descs = [load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz") for r in rs]
        bus = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        T = min(min(len(d.times) for d in descs), len(bus.times))
        if T < 200:
            continue
        order = rng.permutation(len(descs))
        halves = (order[::2], order[1::2])
        Ymean = np.zeros((T, len(TAUS), len(TARGETS)))
        Yhalf = np.zeros((2, T, len(TAUS), len(TARGETS)))
        for ci, t in enumerate(TARGETS):
            for ti, tau in enumerate(TAUS):
                stack = np.stack([ema(d.channels[t][:T], d.fps, tau) for d in descs])
                Ymean[:, ti, ci] = stack.mean(0)
                for h in (0, 1):
                    Yhalf[h, :, ti, ci] = stack[halves[h]].mean(0)
        cols = []
        for name in AUDIO_BASE:
            v = bus.channels[name][:T]
            for tau in TAUS:
                cols.append(ema(v, bus.fps, tau))
        for name in AUDIO_RAW:
            cols.append(bus.channels[name][:T])
        rows.append({"mid": mid, "n_clips": len(descs), "T": T,
                     "X": np.stack(cols, 1).astype(np.float32),
                     "Y": Ymean.astype(np.float32), "H": Yhalf.astype(np.float32)})
        print(f"  {mid}: {len(descs)} clips, {T} frames", flush=True)
    return {"rows": rows}


def fit_shared(n_folds: int = 6) -> dict:
    data = build_shared()
    rows = data["rows"]
    X = np.concatenate([r["X"] for r in rows]).astype(np.float64)
    Y = np.concatenate([r["Y"] for r in rows]).astype(np.float64)
    H = np.concatenate([r["H"] for r in rows], axis=1).astype(np.float64)
    groups = np.concatenate([[r["mid"]] * r["T"] for r in rows])
    clip_id = np.concatenate([[i] * r["T"] for i, r in enumerate(rows)])
    rng = np.random.default_rng(0)
    uniq = np.array(sorted(set(groups.tolist())))
    fold_of = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of[g] for g in groups])
    alphas = [10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]

    def centre(y, ids):
        out = y.copy()
        for i in np.unique(ids):
            m = ids == i
            out[m] -= out[m].mean(0)
        return out

    res = {}
    for ti, tau in enumerate(TAUS):
        Yt = Y[:, ti, :]
        Ht = H[:, :, ti, :]
        rel = np.array([np.corrcoef(Ht[0][:, ci], Ht[1][:, ci])[0, 1]
                        for ci in range(len(TARGETS))])
        r2a = np.zeros((n_folds, len(TARGETS)))
        r2w = np.zeros((n_folds, len(TARGETS)))
        for f in range(n_folds):
            tr, te = fold != f, fold == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
            Xte = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
            ymu = Yt[tr].mean(0)
            Ytr = Yt[tr] - ymu
            # Alpha is selected on tracks the inner fit never saw. Selecting it
            # on data the model was fitted to always picks the smallest penalty
            # and produced held-out R2 of -19 before this was fixed.
            inner_tracks = np.array(sorted(set(groups[tr].tolist())))
            hold = set(inner_tracks[::3].tolist())
            inner = np.array([g in hold for g in groups[tr]])
            Winner = _ridge_multi(Xtr[~inner], Ytr[~inner], alphas)
            best = {}
            for ci in range(len(TARGETS)):
                sc = [_r2(Ytr[inner][:, ci:ci+1], Xtr[inner] @ Winner[a][:, ci:ci+1])[0]
                      for a in alphas]
                best[ci] = alphas[int(np.argmax(sc))]
            W = _ridge_multi(Xtr, Ytr, sorted(set(best.values())))
            pred = np.stack([Xte @ W[best[ci]][:, ci] for ci in range(len(TARGETS))], 1) + ymu
            r2a[f] = _r2(Yt[te], pred)
            r2w[f] = _r2(centre(Yt[te], clip_id[te]), centre(pred, clip_id[te]))
        for ci, t in enumerate(TARGETS):
            a, w = r2a[:, ci], r2w[:, ci]
            bs = np.array([a[rng.integers(0, n_folds, n_folds)].mean() for _ in range(4000)])
            bw = np.array([w[rng.integers(0, n_folds, n_folds)].mean() for _ in range(4000)])
            res.setdefault(t, {})[str(tau)] = {
                "reliability": float(rel[ci]),
                "r2_shared": float(a.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "r2_within_track": float(w.mean()),
                "ci95_within": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
            }
        print(f"tau {tau}s done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_shared.json").write_text(json.dumps(
        {"per_channel": res, "n_tracks": len(uniq), "n_folds": n_folds}, indent=1))
    print(f"\n{'channel':18s} {'tau':>5s} {'reliab':>7s} {'R2shared':>9s} {'CI':>18s} "
          f"{'R2within':>9s}")
    for t in TARGETS:
        for tau in TAUS:
            r = res[t][str(tau)]
            if r["r2_shared"] < 0.02 and r["r2_within_track"] < 0.02:
                continue
            print(f"{t:18s} {tau:5.1f} {r['reliability']:7.3f} {r['r2_shared']:9.3f} "
                  f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['r2_within_track']:9.3f}")
    return res


def locking(tol: float = 0.10, conf_min: float = 0.30,
            n_null: int = 200, seed: int = 0) -> dict:
    """Does the dancer's vertical bounce hold a metrical multiple of the beat?

    The bounce period is compared with the track's beat period on a log2 axis,
    so a dancer bouncing every beat, every other beat or twice a beat all count
    as locked and only a period that sits between levels counts as free.

    The null is the same clip against somebody else's tempo, and it has to be
    built with some care: AIST++ uses six design tempi, so a naive shuffle hands
    a clip its own tempo about a sixth of the time, and an octave of it more
    often than that -- and an octave is invisible to a metrical-level test by
    construction. Candidate tempi within `tol` of an integer number of octaves
    from the true one are therefore excluded from the null, which is what stops
    this from being a test that cannot fail.
    """
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    rows = []
    for r in idx:
        tempo = float(r.get("tempo") or 0.0)
        if tempo <= 0:
            continue
        try:
            z = np.load(CACHE_ROOT / "motion" / f"{r['name']}.npz", allow_pickle=False)
            per = z["ch__vert_period_s"].astype(np.float64)
            conf = z["ch__vert_period_conf"].astype(np.float64)
        except Exception:                              # noqa: BLE001
            continue
        ok = np.isfinite(per) & (per > 0) & (conf >= conf_min)
        if ok.sum() < 40:
            continue
        rows.append({"name": r["name"], "genre": r.get("genre"),
                     "tempo": tempo, "log2_per": np.log2(per[ok])})
    if not rows:
        raise SystemExit("no usable clips; build the motion cache first")

    def locked_frac(log2_per, tempo):
        """Fraction of confident frames within `tol` of an exact metrical level."""
        d = log2_per - np.log2(60.0 / tempo)
        return float((np.abs(d - np.round(d)) < tol).mean())

    real = np.array([locked_frac(r["log2_per"], r["tempo"]) for r in rows])
    tempi = np.unique([r["tempo"] for r in rows])
    rng = np.random.default_rng(seed)
    null = np.zeros(n_null)
    for k in range(n_null):
        vals = []
        for r in rows:
            alt = tempi[np.abs(np.log2(tempi / r["tempo"])
                               - np.round(np.log2(tempi / r["tempo"]))) > tol]
            if not alt.size:
                continue
            vals.append(locked_frac(r["log2_per"], float(rng.choice(alt))))
        null[k] = float(np.mean(vals))
    z = (real.mean() - null.mean()) / max(null.std(), 1e-9)

    # Which level, when it locks at all.
    levels = {}
    for r in rows:
        d = r["log2_per"] - np.log2(60.0 / r["tempo"])
        hit = np.abs(d - np.round(d)) < tol
        for lv, c in zip(*np.unique(np.round(d[hit]).astype(int), return_counts=True)):
            levels[int(lv)] = levels.get(int(lv), 0) + int(c)
    tot = max(sum(levels.values()), 1)

    res = {"n_clips": len(rows), "tol_log2": tol, "conf_min": conf_min,
           "locked_frac": float(real.mean()),
           "locked_frac_sd": float(real.std()),
           "null_mean": float(null.mean()), "null_sd": float(null.std()),
           "z": float(z), "n_distinct_tempi": int(tempi.size),
           "chance_within_tol": float(2 * tol),
           "levels": {str(k): levels[k] / tot for k in sorted(levels)}}
    print(f"clips {res['n_clips']}  distinct tempi {res['n_distinct_tempi']}")
    print(f"locked {res['locked_frac']:.3f} +- {res['locked_frac_sd']:.3f}   "
          f"wrong-tempo null {res['null_mean']:.3f} +- {res['null_sd']:.3f}   "
          f"z {res['z']:+.2f}")
    print(f"a period drawn at random lands inside the tolerance {2*tol:.2f} of the time")
    print("  level  share of locked frames")
    for k in sorted(levels):
        print(f"  {k:+3d}    {levels[k]/tot:.3f}   "
              f"(bounce every {2.0**k:g} beat{'s' if 2.0**k != 1 else ''})")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "locking.json").write_text(json.dumps(res, indent=1))
    print(f"-> {OUT/'locking.json'}")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "fit":
        fit()
    elif cmd == "shared":
        fit_shared()
    elif cmd == "locking":
        locking()
    else:
        raise SystemExit(f"unknown command {cmd} "
                         "(build | fit | shared | locking)")
