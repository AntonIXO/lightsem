#!/usr/bin/env python
"""Concert-domain transfer (PLAN 6.8 step 8.5): a trained sound->light model
scored against real people dancing to real EDM.

The Oslo record gives 29 dancers' head-marker motion on the clock of two DJ
mixes, and the mixes were reconstructed from the four source recordings by
the parallel session's placements (experiments/legacy/stages/outputs/oslo_audio/placements.json). Each
mix is rendered, the causal bus extracted, the model run, and its predicted
energy channels compared with the crowd's mean speed inside each track slot:
(a) the raw correlation of the model's qom with the crowd curve, both sides
causally gain-normalised and smoothed at 2 s -- no fitting at all -- and (b)
the held-out-track ridge from the model's six predicted channels, grouped by
title across the two mixes so the same recording never sits on both sides.
The Oslo audit's own +0.30 used ten published audio scalars the same way.

  bash run_torch.sh scripts/legacy_stages/run_oslo_transfer.py --ckpt <path> --tag <name>
"""
from __future__ import annotations

import argparse, dataclasses, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import ema, extract_bus  # noqa: E402
from lightsem.data import oslo_audio as oa  # noqa: E402
from lightsem.data.oslo import GROUPS, descriptors, load_session, load_tracks  # noqa: E402
from lightsem.model.infer import LightModel  # noqa: E402
from lightsem.motion.io import CACHE_ROOT, load_bus, save_bus  # noqa: E402

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage8")
FPS = 40.0
CH = ("qom", "vert_bounce_amp", "head_speed", "com_speed", "spread", "hand_speed")


def cnorm(v, win_s=30.0):
    m = ema(v, FPS, win_s)
    s = np.sqrt(np.maximum(ema((v - m) ** 2, FPS, win_s), 1e-9))
    return (v - m) / s


def mix_wav(mix: int) -> Path:
    p = CACHE_ROOT / "oslo_mix" / f"mix{mix}.wav"
    if not p.exists():
        rows = json.loads(Path("/root/lightsem/experiments/legacy/stages/outputs/oslo_audio/placements.json").read_text())
        names = {f.name for f in dataclasses.fields(oa.Placement)}
        pl = [oa.Placement(**{k: v for k, v in r.items() if k in names}) for r in rows if int(r["mix"]) == mix]
        y = oa.render_mix(mix, pl)
        oa.write_wav(p, y)
        print(f"rendered mix {mix}: {len(y)/oa.SR:.0f} s from {len(pl)} placements", flush=True)
    return p


def mix_bus(mix: int, causal: bool = True):
    p = CACHE_ROOT / "oslo_mix" / f"mix{mix}_bus{'' if causal else '_offline'}.npz"
    if not p.exists():
        save_bus(extract_bus(mix_wav(mix), causal=causal), p)
    return load_bus(p)


def crowd_curve(mix: int) -> np.ndarray:
    gs = [g for g, (_, m) in GROUPS.items() if m == mix]
    desc = {g: descriptors(load_session(g)) for g in gs}
    n = min(len(desc[g]["speed"]) for g in gs)
    sp = np.concatenate([desc[g]["speed"][:n] for g in gs], axis=1)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(sp, axis=1), nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--offline", action="store_true", help="offline bus for an offline checkpoint")
    ap.add_argument("--tau", type=float, default=2.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lm = LightModel(a.ckpt)
    tracks = load_tracks()
    slots = []          # (title, mix, X (T,6) standardised model channels, y crowd, placement margin)
    pl = {(r["title"], int(r["mix"])): r for r in json.loads(Path("/root/lightsem/experiments/legacy/stages/outputs/oslo_audio/placements.json").read_text())}
    for mix in (1, 2):
        bus = mix_bus(mix, causal=not a.offline)
        dm = lm.descriptors(bus, seed=0)
        crowd = crowd_curve(mix)
        n = min(len(dm.times), len(crowd))
        X = np.stack([cnorm(ema(np.nan_to_num(dm.channels[c][:n], nan=0.0), FPS, a.tau)) for c in CH], 1)
        y = cnorm(ema(crowd[:n], FPS, a.tau))
        for t in [t for t in tracks if int(t["dj_mix"]) == mix]:
            s0, s1 = int(float(t["start_s"]) * FPS), min(int(float(t["end_s"]) * FPS), n)
            if s1 - s0 < 400:
                continue
            title = next((k[0] for k in pl if k[1] == mix and k[0].lower() in t.get("title", t.get("track", "")).lower()), t.get("title", "?"))
            slots.append((title, mix, X[s0:s1], y[s0:s1], float(pl.get((title, mix), {}).get("margin", float("nan")))))
    print(f"{len(slots)} track slots", flush=True)
    # (a) raw correlation, no fitting
    raw = []
    for title, mix, X, y, margin in slots:
        r = float(np.corrcoef(X[:, 0], y)[0, 1])
        raw.append({"title": title, "mix": mix, "r_qom_vs_crowd": r, "placement_margin": margin,
                    "r_each": {c: float(np.corrcoef(X[:, i], y)[0, 1]) for i, c in enumerate(CH)}})
        print(f"  {title:12s} mix {mix}  r(model qom, crowd speed) {r:+.3f}   margin {margin:+.3f}", flush=True)
    # (b) held-out-track ridge from the model's channels, grouped by title
    titles = sorted({s[0] for s in slots})
    r2s, rs = [], []
    for held in titles:
        tr = [s for s in slots if s[0] != held]; te = [s for s in slots if s[0] == held]
        Xtr = np.concatenate([s[2] for s in tr]); ytr = np.concatenate([s[3] for s in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        W = np.linalg.solve(A.T @ A + 1e3 * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
        for s in te:
            B = np.c_[(s[2] - mu) / sd, np.ones(len(s[2]))]
            pred = B @ W + ytr.mean()
            r2s.append(1 - ((s[3] - pred) ** 2).sum() / max(((s[3] - s[3].mean()) ** 2).sum(), 1e-9))
            rs.append(float(np.corrcoef(s[3], pred)[0, 1]))
    r2s, rs = np.array(r2s), np.array(rs)
    rng = np.random.default_rng(0)
    bs = np.array([r2s[rng.integers(0, len(r2s), len(r2s))].mean() for _ in range(4000)])
    res = {"ckpt": a.ckpt, "tag": a.tag, "tau": a.tau, "offline_bus": a.offline, "n_slots": len(slots),
           "raw": raw, "raw_mean_r_qom": float(np.mean([x["r_qom_vs_crowd"] for x in raw])),
           "held_out_track_ridge": {"r2": float(r2s.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                    "r": float(rs.mean()), "n_slots": int(len(r2s))},
           "reference_oslo_audit_r2": 0.30}
    print(f"raw mean r(qom) {res['raw_mean_r_qom']:+.3f}; held-out-track ridge from model channels R2 {r2s.mean():+.3f} "
          f"CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] r {rs.mean():+.3f}  (Oslo audit from published audio: +0.30)")
    (OUT / f"oslo_transfer_{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
