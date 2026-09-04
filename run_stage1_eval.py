#!/usr/bin/env python
"""Stage 1 gates at the LIGHT level (runbook rb-ls2-01).

Descriptor R2 is an intermediate. What the plan actually gates on is the light:
does the model's light land on the music, does its feature distribution look
like light derived from a real dancer, and does sampling the style head give
variety rather than one averaged look.

    bash run_torch.sh run_stage1_eval.py --n 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightsem.conductor.beat_trigger import render_beat_trigger
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.eval.bas import bas_report
from lightsem.eval.fid import FeatureScaler, diversity, frechet, split_half_floor
from lightsem.eval.light_features import block_shuffle, delta_features, level_features, light_onsets
from lightsem.model.infer import LightModel
from lightsem.model.light_data import load_tensor, make_splits
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus

OUT = Path("/root/lightsem/outputs/stage1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="val_track")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ckpt", default="/mnt/lightsem-data/ckpt/light_trunk_v1.pt")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--no-calibration", action="store_true",
                    help="ignore the checkpoint's fitted event thresholds and fall "
                         "back to the per-clip 97th percentile, to price the "
                         "calibration on otherwise identical features")
    args = ap.parse_args()

    z, meta = load_tensor()
    sp = make_splits(meta)
    idx = getattr(sp, args.split)
    rng = np.random.default_rng(0)
    idx = idx[rng.permutation(len(idx))[:args.n]]
    names = [meta[i]["name"] for i in idx]

    lm = LightModel(args.ckpt)
    if args.no_calibration:
        lm.event_thr = {}
    print(f"event thresholds: {lm.event_thr or 'per-clip 97th percentile'}")
    cond = ConductorHybrid()
    rig = canonical_rig()
    F_model, F_real, F_bt, F_null = [], [], [], []
    rows, div_within = [], []

    for nm in names:
        d = load_descriptors(CACHE_ROOT / "motion" / f"{nm}.npz")
        mid = nm.split("_")[4]
        bus_full = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        dur = d.duration_s
        bus = slice_bus(bus_full, dur)
        beats = bus_full.beats[(bus_full.beats >= 0) & (bus_full.beats < dur)]
        if beats.size < 6:
            continue
        real = cond.render(d, bus).frames
        bt = render_beat_trigger(bus).frames
        model_frames = []
        for s in range(args.seeds):
            dm = lm.descriptors(bus, seed=s)
            model_frames.append(cond.render(dm, bus).frames)
        F_model.append(delta_features(model_frames[0], rig, 40.0))
        F_real.append(delta_features(real, rig, 40.0))
        F_bt.append(delta_features(bt, rig, 40.0))
        F_null.append(delta_features(block_shuffle(real, 40.0, 1.0, rng), rig, 40.0))
        feats_seeds = [delta_features(f, rig, 40.0) for f in model_frames]
        div_within.append(np.array(feats_seeds))

        ev_m, _ = light_onsets(model_frames[0], rig, 40.0)
        ev_r, _ = light_onsets(real, rig, 40.0)
        ev_b, _ = light_onsets(bt, rig, 40.0)
        rows.append({
            "name": nm,
            "model": bas_report(ev_m, beats, dur, rng=rng),
            "real": bas_report(ev_r, beats, dur, rng=rng),
            "bt": bas_report(ev_b, beats, dur, rng=rng),
            "level_model": level_features(model_frames[0], rig),
            "level_real": level_features(real, rig),
        })

    Xm, Xr, Xb, Xn = map(np.array, (F_model, F_real, F_bt, F_null))
    sc = FeatureScaler(Xr)
    Zm, Zr, Zb, Zn = sc(Xm), sc(Xr), sc(Xb), sc(Xn)
    floor = split_half_floor(Zr, rng=rng)
    fid_model = frechet(Zm, Zr)
    fid_null = frechet(Zr, Zn)
    fid_bt = frechet(Zb, Zr)

    def paired(key):
        a = np.array([r[key]["bas"] - r[key]["null_mean"] for r in rows])
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(4000)])
        return {"delta": float(a.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "excludes_zero": bool(np.percentile(bs, 2.5) > 0)}

    div_seed = float(np.mean([
        np.mean([np.linalg.norm(sc(v[a])[None] - sc(v[b])[None]) for a in range(len(v))
                 for b in range(a + 1, len(v))]) for v in div_within]))
    div_real_set = diversity(Zr, rng=rng)
    div_model_set = diversity(Zm, rng=rng)

    res = {
        "n_clips": len(rows), "split": args.split, "seeds": args.seeds,
        "bas_minus_null": {"model": paired("model"), "real_dance": paired("real"),
                           "beat_trigger": paired("bt")},
        "fid_light": {"model_vs_real": fid_model, "beat_trigger_vs_real": fid_bt,
                      "real_vs_block_shuffled": fid_null, "split_half_floor": floor},
        "diversity": {"across_seeds_same_track": div_seed,
                      "across_clips_model": div_model_set,
                      "across_clips_real": div_real_set},
        "brightness": {
            "model_mean": float(np.mean([r["level_model"]["mean"] for r in rows])),
            "real_mean": float(np.mean([r["level_real"]["mean"] for r in rows]))},
    }
    b = res["bas_minus_null"]
    res["gates"] = {
        "G_bas_model_ge_0.9_real": {
            "passed": bool(b["model"]["delta"] >= 0.9 * b["real_dance"]["delta"]),
            "evidence": f"model {b['model']['delta']:+.4f} CI "
                        f"{[round(x,4) for x in b['model']['ci95']]} versus real-mapped "
                        f"{b['real_dance']['delta']:+.4f}; beat trigger "
                        f"{b['beat_trigger']['delta']:+.4f}"},
        "G_fid_lt_half_null": {
            "passed": bool(fid_model < 0.5 * fid_null),
            "evidence": f"FID(model, real) {fid_model:.2f} versus half the "
                        f"structure-destroying null {0.5*fid_null:.2f}; split-half floor "
                        f"{floor['mean']:.2f}; beat trigger scores {fid_bt:.2f}"},
        "G_div_ge_0.7_real": {
            "passed": bool(div_model_set >= 0.7 * div_real_set),
            "evidence": f"across clips model {div_model_set:.2f} versus real "
                        f"{div_real_set:.2f}; across style samples of ONE track "
                        f"{div_seed:.3f}"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage1_light_eval_{args.tag}.json").write_text(json.dumps(
        {"summary": res, "per_clip": rows}, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
