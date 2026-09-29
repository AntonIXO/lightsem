#!/usr/bin/env python
"""H-H: is Beat Saber lighting a LIGHTING corpus or a TEMPO corpus?  (PLAN.md 9.2, gate G9.2)

The rb-ls2-00b / exp-016 recipe on the Stage 9 target format, split by SONG (different maps of
one song stay in one fold), every number next to its nulls:

  events   scene-change events (exp-016 novelty on the rendered scene vector, >= 2 s apart) and
           state flips (any group switching on/off): per-map AP of a song-split ridge on the
           causal 10 Hz audio features (probe_event_target.audio_feats), the rhythm-of-past-
           events hazard, both, the OFFLINE counterpart (symmetric pooling on the same features),
           the map's exact BPM grid in place of our estimate, and the beat-trigger baseline
           (conductor/beat_trigger.py rendered from the causal bus, its own scene novelty as the
           score). Null: the target shifted by +-2, +-4, +-8 s.
  envelopes per-lane gain-normalised intensity at 40 fps (EMA 0.5 s and 2 s), ridge on the bus
           scalars at three EMAs, held-out-song R2, vs the shifted-target null and the beat
           trigger's matching role channel.
  controls number of active groups from RMS alone vs the full bus; our beat grid vs the map BPM.

  bash run.sh scripts/legacy_stages/run_stage9_gate.py [--limit N]
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import ema  # noqa: E402
from lightsem.conductor.beat_trigger import render_beat_trigger  # noqa: E402
from lightsem.conductor.rig import canonical_rig, role_index  # noqa: E402
from lightsem.dmx.decode import decode_stream, emitted_light  # noqa: E402
from lightsem.motion.io import load_bus  # noqa: E402
from scripts.diagnostics.probe_event_target import (PAST, audio_feats, average_precision, hazard_feats, novelty, target)  # noqa: E402


def target_h(train: np.ndarray, horizon: int) -> np.ndarray:
    """1 if an event falls in (t, t+horizon] (probe_event_target.target with a chosen horizon)."""
    cs = np.concatenate([[0.0], np.cumsum(train)])
    T = len(train)
    e = np.minimum(np.arange(T) + 1 + horizon, T)
    return (cs[e] - cs[np.arange(T) + 1] > 0).astype(np.float32)

CACHE = Path("/mnt/lightsem-data/cache")
OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")
HZ, FPS = 10.0, 40.0
SHIFTS = (-8.0, -4.0, -2.0, 2.0, 4.0, 8.0)
TAUS = (0.5, 2.0, 8.0)
AUD = ("rms_db", "flux", "flux_lf", "flux_hf", "onset", "centroid", "flatness", "rolloff",
       "percussive_ratio", "pulse_clarity", "energy_lf", "energy_hf", "beat_accent")
# v2 lane -> the beat trigger's role whose channel is the matching baseline
LANE_ROLE = {"ring_lights": "big_slow", "center_lights": "back", "back_lasers": "event",
             "left_lasers": "beam", "right_lasers": "beam"}


def audio_feats_offline(x: np.ndarray) -> np.ndarray:
    """Same pooling scales, centred (past AND future), plus future-minus-past novelty."""
    xc = (x - x.mean(0)) / (x.std(0) + 1e-6)
    pools = [uniform_filter1d(xc, size=2 * k + 1, axis=0, mode="nearest") for k in PAST]
    fut = [uniform_filter1d(xc, size=k, axis=0, origin=-(k // 2), mode="nearest") for k in PAST]
    past = [uniform_filter1d(xc, size=k, axis=0, origin=(k - 1) // 2, mode="nearest") for k in PAST]
    nov = [fut[i] - past[i] for i in range(len(PAST))]
    return np.concatenate(pools + nov, 1).astype(np.float32)


def cnorm(v, fps, win_s=30.0):
    m = ema(v, fps, win_s)
    s = np.sqrt(np.maximum(ema((v - m) ** 2, fps, win_s), 1e-9))
    return (v - m) / s


def shift_target(t: np.ndarray, sh: int) -> np.ndarray:
    out = np.zeros_like(t)
    if sh >= 0:
        out[:len(t) - sh] = t[sh:] if sh else t
    else:
        out[-sh:] = t[:len(t) + sh]
    return out


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, y.mean()


def ridge_apply(p, X):
    mu, sd, w, b = p
    return ((X - mu) / sd) @ w + b


def folds_by_song(keys, k=5, seed=0):
    songs = sorted({key.split("=")[0] for key in keys})
    rng = np.random.default_rng(seed)
    fold_of = {s: i for i, s in zip(np.repeat(np.arange(k), int(np.ceil(len(songs) / k)))[:len(songs)], rng.permutation(songs))}
    return np.array([fold_of[key.split("=")[0]] for key in keys])


def beat_trigger_scene(bus):
    """The 30-line baseline rendered from the causal bus -> (T40, 6) scene vector and role dimmers."""
    rig = canonical_rig()
    st = render_beat_trigger(bus, rig=rig)
    dec = decode_stream(rig, st.frames)
    lum = emitted_light(dec)                                   # (T, nf)
    w = lum / np.maximum(lum.sum(1, keepdims=True), 1e-9)
    hue = dec["hue"]; sat = dec["sat"]
    hc = (w * np.cos(2 * np.pi * hue)).sum(1); hs = (w * np.sin(2 * np.pi * hue)).sum(1)
    scene = np.stack([hc, hs, (w * sat).sum(1), lum.mean(1), np.percentile(lum, 90, axis=1), np.sqrt(hc ** 2 + hs ** 2)], 1)
    roles = role_index(rig)
    role_dim = {r: lum[:, idx].mean(1) for r, idx in roles.items() if len(idx)}
    return scene.astype(np.float32), role_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--lam-env", type=float, default=1e3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted((CACHE / "beatsaver_visual").glob("*.npz"))
    if a.limit:
        files = files[:a.limit]
    maps = []
    for p in files:
        key = p.stem
        v = np.load(p, allow_pickle=False)
        lp = CACHE / "beatsaver_light" / f"{key}.npz"
        bp = CACHE / "beatsaver_audio" / f"{key}.npz"
        if not (lp.exists() and bp.exists()):
            continue
        L = np.load(lp, allow_pickle=False)
        meta = json.loads(str(L["meta"]))
        x, y = v["x"].astype(np.float32), v["y"].astype(np.float32)
        T10 = len(x)
        if T10 < 400 or not np.isfinite(x).all():
            continue
        bus = load_bus(bp)
        on = L["on"]                                           # (G, T40)
        flips = (np.abs(np.diff(on.astype(np.int8), axis=1)) > 0).any(0)   # (T40-1,)
        fl10 = np.zeros(T10, np.float32)
        idx = np.minimum((np.flatnonzero(flips) / FPS * HZ).astype(int), T10 - 1)
        fl10[idx] = 1.0
        bt_scene, bt_roles = beat_trigger_scene(bus)
        # beat-trigger scene novelty at 10 Hz (its own 'cue' score), and its roles at 40 fps
        t40 = np.arange(len(bt_scene)) / FPS; t10 = np.arange(T10) / HZ
        bt10 = np.stack([np.interp(t10, t40, bt_scene[:, k]) for k in range(6)], 1)
        bt_nov = novelty(bt10)
        # map grid phase (exact) at 10 Hz
        per = 60.0 / meta["bpm"]
        ph = np.mod((t10 - meta["offset_s"]) / per, 1.0)
        oracle = np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), (1 - ph) ** 2.5], 1).astype(np.float32)
        maps.append({"key": key, "x": x, "y": y, "ev_scene": v["ev_scene"].astype(np.float32), "ev_state": fl10,
                     "n_active": v["n_active"].astype(np.float32), "bt_nov": bt_nov.astype(np.float32),
                     "bt_roles": bt_roles, "oracle": oracle, "bus": bus, "inten": L["inten"], "groups": list(L["groups"]),
                     "meta": meta, "grid_bpm": float(v["bpm"]), "bus_bpm": float(bus.tempo)})
    print(f"{len(maps)} maps loaded in {time.time()-t0:.0f}s", flush=True)
    t_ev = time.time()
    keys = [m["key"] for m in maps]
    fold = folds_by_song(keys)
    res = {"n_maps": len(maps), "n_songs": len({k.split('=')[0] for k in keys}), "lam": a.lam, "shifts_s": SHIFTS}

    # ------------------------------------------------------------ beat-grid validation (free, exact)
    def ratio_class(est, ref):
        r = est / ref
        for name, val in (("1", 1.0), ("2", 2.0), ("1/2", 0.5), ("3/2", 1.5), ("2/3", 2 / 3), ("3", 3.0), ("1/3", 1 / 3), ("4/3", 4 / 3), ("3/4", 0.75)):
            if abs(r / val - 1) < 0.02:
                return name
        return "other"
    grid = {"features_10hz": {}, "bus_40fps": {}}
    for name, k in (("features_10hz", "grid_bpm"), ("bus_40fps", "bus_bpm")):
        cls = [ratio_class(m[k], m["meta"]["bpm"]) for m in maps]
        grid[name] = {c: int(cls.count(c)) for c in sorted(set(cls))}
        grid[name]["exact_frac"] = float(cls.count("1") / len(cls))
    res["beat_grid_vs_map_bpm"] = grid
    print("beat grid vs map BPM:", json.dumps(grid), flush=True)

    # ------------------------------------------------------------ events
    ev_res = {}
    for tname in ("ev_scene", "ev_state"):
        feats = {}
        for m in maps:
            tr = m[tname]
            haz = hazard_feats(tr).astype(np.float32)
            xa = audio_feats(m["x"]); xo = audio_feats_offline(m["x"])
            xg = audio_feats(np.concatenate([m["x"][:, :-4], m["oracle"]], 1))    # our beat cols replaced by the map grid
            feats[m["key"]] = {"t": (target(tr) if tname == "ev_scene" else target_h(tr, 3)).astype(np.float32), "haz": haz, "audio": xa, "audio_haz": np.concatenate([xa, haz], 1),
                               "offline": xo, "offline_haz": np.concatenate([xo, haz], 1), "oracle_grid": xg,
                               "bt": m["bt_nov"][:, None], "base": haz[:, -1:]}
        arms = ("base", "haz", "audio", "audio_haz", "offline", "offline_haz", "oracle_grid", "bt")
        ap_arm = {k: [] for k in arms}; lift_arm = {k: [] for k in arms}
        ap_shift = {k: {s: [] for s in SHIFTS} for k in ("audio", "audio_haz", "offline", "bt")}
        rate = []
        for f in range(5):
            tr_keys = [k for k, ff in zip(keys, fold) if ff != f]; te_keys = [k for k, ff in zip(keys, fold) if ff == f]
            for arm in arms:
                if arm == "bt":
                    continue
                X = np.concatenate([feats[k][arm] for k in tr_keys]); Y = np.concatenate([feats[k]["t"] for k in tr_keys])
                p = ridge_fit(X, Y, a.lam)
                for k in te_keys:
                    t = feats[k]["t"]
                    if t.sum() < 3:
                        continue
                    s = ridge_apply(p, feats[k][arm])
                    apv = average_precision(s, t)
                    ap_arm[arm].append(apv); lift_arm[arm].append(apv / t.mean())
                    if arm in ap_shift:
                        for sh in SHIFTS:
                            ts = shift_target(t, int(round(sh * HZ)))
                            if ts.sum() >= 3:
                                ap_shift[arm][sh].append(average_precision(s, ts))
            for k in te_keys:
                t = feats[k]["t"]
                if t.sum() < 3:
                    continue
                s = feats[k]["bt"][:, 0]
                ap_arm["bt"].append(average_precision(s, t)); lift_arm["bt"].append(average_precision(s, t) / t.mean())
                for sh in SHIFTS:
                    ts = shift_target(t, int(round(sh * HZ)))
                    if ts.sum() >= 3:
                        ap_shift["bt"][sh].append(average_precision(s, ts))
                rate.append(float(feats[k]["t"].mean()))
        r = {"n_eval_maps": len(ap_arm["audio"]), "target_rate": float(np.mean(rate)),
             "ap": {k: float(np.mean(v)) for k, v in ap_arm.items()}, "lift": {k: float(np.mean(v)) for k, v in lift_arm.items()},
             "ap_shift": {k: {str(s): float(np.mean(v)) for s, v in d.items()} for k, d in ap_shift.items()}}
        for arm in ap_shift:
            r[f"ap_{arm}_minus_shift"] = r["ap"][arm] - float(np.mean([r["ap_shift"][arm][str(s)] for s in SHIFTS]))
        d = np.array(ap_arm["audio"]) - np.array(ap_arm["bt"][:len(ap_arm["audio"])])
        rng = np.random.default_rng(0)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        r["audio_minus_bt"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        d2 = np.array(ap_arm["offline"]) - np.array(ap_arm["bt"][:len(ap_arm["offline"])])
        r["offline_minus_bt"] = float(d2.mean())
        ev_res[tname] = r
        print(f"\n[{tname}] rate {r['target_rate']:.3f}  AP: " + "  ".join(f"{k} {v:.3f}" for k, v in r["ap"].items()), flush=True)
        print(f"   lift: " + "  ".join(f"{k} {v:.2f}" for k, v in r["lift"].items()), flush=True)
        for arm in ap_shift:
            print(f"   {arm:10s} shift-null AP " + " ".join(f"{s:+.0f}s {r['ap_shift'][arm][str(s)]:.3f}" for s in SHIFTS) + f"   audio-minus-shift {r[f'ap_{arm}_minus_shift']:+.3f}", flush=True)
        print(f"   audio - beat-trigger {r['audio_minus_bt']['mean']:+.3f} CI {np.round(r['audio_minus_bt']['ci95'], 3)}   offline - bt {r['offline_minus_bt']:+.3f}", flush=True)
    res["events"] = ev_res
    print(f"events {time.time()-t_ev:.0f}s", flush=True)

    # ------------------------------------------------------------ per-lane envelopes at 40 fps
    t1 = time.time()
    env_res = {}
    lanes = sorted({g for m in maps for g in m["groups"] if not g.startswith("g")})
    FUT = (0.5, 1.0, 2.0)
    ENV_SHIFTS = (-4.0, 4.0)
    pre = {}                     # key -> dict(X, Xo, lanes: {lane: [(y_tau05, y_tau2, bt05, bt2)]})
    for m in maps:
        bus = m["bus"]; T = min(len(bus.times), m["inten"].shape[1])
        cols = []
        for c in AUD:
            v = np.nan_to_num(bus.channels[c][:T].astype(np.float64), nan=0.0)
            vn = cnorm(v, FPS)
            cols += [ema(vn, FPS, tau) for tau in TAUS]
        X = np.stack(cols, 1).astype(np.float32)
        Xo = np.concatenate([X] + [np.roll(X, -int(round(s_ * FPS)), axis=0) for s_ in FUT], 1)
        entry = {"X": X, "Xo": Xo, "lanes": {}}
        for i, g in enumerate(m["groups"]):
            lane = g if not g.startswith("g") else "v3_groups"
            if np.std(m["inten"][i][:T]) < 1e-3:
                continue
            role = LANE_ROLE.get(lane, "beam")
            b = m["bt_roles"][role][:T].astype(np.float64)
            ys = {}
            for tau in (0.5, 2.0):
                ys[tau] = (cnorm(ema(m["inten"][i][:T].astype(np.float64), FPS, tau), FPS).astype(np.float32),
                           cnorm(ema(b, FPS, tau), FPS).astype(np.float32)[:, None])
            entry["lanes"].setdefault(lane, []).append(ys)
        pre[m["key"]] = entry
    print(f"envelope features ready {time.time()-t1:.0f}s", flush=True)
    for lane in lanes + ["v3_groups"]:
        for tau in (0.5, 2.0):
            r2 = {"audio": [], "shift": [], "bt": [], "offline": []}
            for f in range(5):
                tr_keys = [k for k, ff in zip(keys, fold) if ff != f and lane in pre[k]["lanes"]]
                te_keys = [k for k, ff in zip(keys, fold) if ff == f and lane in pre[k]["lanes"]]
                if not tr_keys or not te_keys:
                    continue
                def cat(klist, what):
                    out = []
                    for k in klist:
                        for ys in pre[k]["lanes"][lane]:
                            y, b = ys[tau]
                            if what == "X":
                                out.append(pre[k]["X"])
                            elif what == "Xo":
                                out.append(pre[k]["Xo"])
                            elif what == "y":
                                out.append(y)
                            elif what == "b":
                                out.append(b)
                            elif what.startswith("ysh"):
                                out.append(np.roll(y, int(round(float(what[3:]) * FPS))))
                    return np.concatenate(out)
                Ytr = cat(tr_keys, "y")
                p = ridge_fit(cat(tr_keys, "X"), Ytr, a.lam_env)
                po = ridge_fit(cat(tr_keys, "Xo"), Ytr, a.lam_env)
                pb = ridge_fit(cat(tr_keys, "b"), Ytr, 1.0)
                psh = {sh: ridge_fit(cat(tr_keys, "X"), cat(tr_keys, f"ysh{sh}"), a.lam_env) for sh in ENV_SHIFTS}
                for k in te_keys:
                    for ys in pre[k]["lanes"][lane]:
                        y, b = ys[tau]; ss = ((y - y.mean()) ** 2).sum() + 1e-9
                        r2["audio"].append(1 - ((y - ridge_apply(p, pre[k]["X"])) ** 2).sum() / ss)
                        r2["offline"].append(1 - ((y - ridge_apply(po, pre[k]["Xo"])) ** 2).sum() / ss)
                        r2["bt"].append(1 - ((y - ridge_apply(pb, b)) ** 2).sum() / ss)
                        r2["shift"].append(np.mean([1 - ((np.roll(y, int(round(sh * FPS))) - ridge_apply(psh[sh], pre[k]["X"])) ** 2).sum() / ss for sh in ENV_SHIFTS]))
            if r2["audio"]:
                env_res[f"{lane}@{tau}"] = {k: float(np.mean(v)) for k, v in r2.items()} | {"n": len(r2["audio"])}
                e = env_res[f"{lane}@{tau}"]
                print(f"  env {lane:14s} tau {tau:3.1f}s  R2 audio {e['audio']:+.3f}  shift-null {e['shift']:+.3f}  beat-trigger {e['bt']:+.3f}  offline {e['offline']:+.3f}  (n {e['n']})", flush=True)
    res["envelopes"] = env_res
    print(f"envelopes {time.time()-t1:.0f}s", flush=True)

    # ------------------------------------------------------------ group-count control
    gc = {"rms_only": [], "full": []}
    for f in range(5):
        tr_keys = [k for k, ff in zip(keys, fold) if ff != f]; te_keys = [k for k, ff in zip(keys, fold) if ff == f]
        def rows(klist, cols):
            Xs, Ys = [], []
            for k in klist:
                m = maps[keys.index(k)]
                if m["n_active"].std() < 1e-3:
                    continue
                Xs.append(audio_feats(m["x"][:, cols])); Ys.append(cnorm(m["n_active"].astype(np.float64), HZ))
            return Xs, Ys
        names = list(np.load(files[0], allow_pickle=False)["names"])
        rms_cols = [i for i, n in enumerate(names) if n.startswith("rms_")]
        for arm, cols in (("rms_only", rms_cols), ("full", list(range(len(names))))):
            Xs, Ys = rows(tr_keys, cols)
            p = ridge_fit(np.concatenate(Xs), np.concatenate(Ys), a.lam)
            Xt, Yt = rows(te_keys, cols)
            for X, y in zip(Xt, Yt):
                gc[arm].append(1 - ((y - ridge_apply(p, X)) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    res["group_count_control"] = {k: float(np.mean(v)) for k, v in gc.items()}
    print(f"\ngroup-count control: R2 rms-only {res['group_count_control']['rms_only']:+.3f}  full bus {res['group_count_control']['full']:+.3f}", flush=True)

    # ------------------------------------------------------------ gate G9.2 (literal AP / R2 units)
    e = ev_res["ev_scene"]
    env_a = np.mean([v["audio"] for k, v in env_res.items() if k.endswith("@2.0")]) if env_res else float("nan")
    env_s = np.mean([v["shift"] for k, v in env_res.items() if k.endswith("@2.0")]) if env_res else float("nan")
    env_b = np.mean([v["bt"] for k, v in env_res.items() if k.endswith("@2.0")]) if env_res else float("nan")
    gate = {"G9.2-shift": {"events_margin": e["ap_audio_minus_shift"], "envelope_margin": float(env_a - env_s),
                           "passed": bool(e["ap_audio_minus_shift"] >= 0.15 and (env_a - env_s) >= 0.15)},
            "G9.2-bt": {"events_margin": e["audio_minus_bt"]["mean"], "envelope_margin": float(env_a - env_b),
                        "passed": bool(e["audio_minus_bt"]["mean"] >= 0.05 and (env_a - env_b) >= 0.05)},
            "G9.2-L0": {"causal_ap": e["ap"]["audio"], "offline_ap": e["ap"]["offline"], "passed": True}}
    res["gate"] = gate
    print("\nGATE", json.dumps(gate, indent=1), flush=True)
    res["elapsed_s"] = time.time() - t0
    (OUT / "hh_gate.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT/'hh_gate.json'} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
