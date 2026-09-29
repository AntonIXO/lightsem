#!/usr/bin/env python
"""Stage 0a — descriptors -> light targets -> metrics -> renders.

    bash run.sh scripts/legacy_stages/run_stage0a.py metrics [--n 120]
    bash run.sh scripts/legacy_stages/run_stage0a.py render  [--n 5]

Preregistered gates (PLAN.md section 6, runbook rb-ls2-00):
  G1  Light-BAS(real-dance light)  >= 0.9 x metronome
  G2  FID_light(split-half real) x 3 < FID_light(structure-destroying null)
  G4  >= 4/5 real-dance renders read as dancing  (Anton's eyes, not a number)
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

from lightsem.conductor.beat_trigger import render_beat_trigger
from lightsem.conductor.rig import canonical_rig
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.v0 import ConductorV0
from lightsem.eval.bas import bas_report, circular_phase
from lightsem.eval.fid import FeatureScaler, diversity, frechet, split_half_floor
from lightsem.eval.light_features import (
    FEATURE_NAMES, block_shuffle, delta_features, level_features, light_onsets)
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage0a")


def sample_clips(n: int, seed: int = 0) -> list[dict]:
    """Stratified by genre, one clip per (music, choreo) group to avoid dupes."""
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    seen, pool = set(), []
    for r in idx:
        key = (r["music_id"], r["choreo_id"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(r)
    by_genre = collections.defaultdict(list)
    for r in pool:
        by_genre[r["genre"]].append(r)
    rng = np.random.default_rng(seed)
    out = []
    per = max(n // max(len(by_genre), 1), 1)
    for g in sorted(by_genre):
        rows = by_genre[g]
        pick = rng.permutation(len(rows))[:per]
        out += [rows[i] for i in pick]
    return out[:n]


def metrics(n: int = 120) -> dict:
    clips = sample_clips(n)
    cond = ConductorV0()
    hyb = ConductorHybrid()
    rig = canonical_rig()
    rng = np.random.default_rng(0)
    rows, feats_o, feats_b, feats_null, feats_h = [], [], [], [], []
    t0 = time.time()
    for r in clips:
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        bus_full = load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz")
        dur = d.duration_s
        beats = bus_full.beats[(bus_full.beats >= 0) & (bus_full.beats < dur)]
        if beats.size < 6:
            continue
        bus = slice_bus(bus_full, dur)
        st_o = cond.render(d)
        st_h = hyb.render(d, bus)
        st_b = render_beat_trigger(bus)
        ev_o, _ = light_onsets(st_o.frames, rig, 40.0)
        ev_h, _ = light_onsets(st_h.frames, rig, 40.0)
        ev_b, _ = light_onsets(st_b.frames, rig, 40.0)
        rep_o = bas_report(ev_o, beats, dur, rng=rng)
        rep_h = bas_report(ev_h, beats, dur, rng=rng)
        rep_b = bas_report(ev_b, beats, dur, rng=rng)
        ph_o = circular_phase(ev_o, beats)
        # re-phased: remove the choreography's own phase offset, then re-score
        rep_o_rp = bas_report(ev_o - r["beat_offset_s"], beats, dur, rng=rng)
        feats_o.append(delta_features(st_o.frames, rig, 40.0))
        feats_b.append(delta_features(st_b.frames, rig, 40.0))
        feats_h.append(delta_features(st_h.frames, rig, 40.0))
        feats_null.append(delta_features(block_shuffle(st_o.frames, 40.0, 1.0, rng), rig, 40.0))
        rows.append({
            "name": r["name"], "genre": r["genre"], "duration_s": dur,
            "n_light_events_oracle": int(ev_o.size), "n_light_events_bt": int(ev_b.size),
            "bas_oracle": rep_o["bas"], "bas_oracle_rephased": rep_o_rp["bas"],
            "bas_hybrid": rep_h["bas"], "null_hybrid": rep_h["null_mean"],
            "z_hybrid": rep_h["z"], "n_light_events_hybrid": int(ev_h.size),
            "bas_bt": rep_b["bas"], "metronome": rep_o["metronome"],
            "null_oracle": rep_o["null_mean"], "null_bt": rep_b["null_mean"],
            "z_oracle": rep_o["z"], "z_bt": rep_b["z"],
            "light_phase_R": ph_o["R"], "dance_phase_R": r["beat_phase_R"],
            "level_hybrid": level_features(st_h.frames, rig),
            "level_oracle": level_features(st_o.frames, rig),
            "level_bt": level_features(st_b.frames, rig),
        })
    Xo, Xb, Xn, Xh = (np.array(feats_o), np.array(feats_b), np.array(feats_null),
                      np.array(feats_h))
    sc = FeatureScaler(Xo)
    Zo, Zb, Zn, Zh = sc(Xo), sc(Xb), sc(Xn), sc(Xh)
    floor = split_half_floor(Zo, rng=rng)

    def paired(a_key: str, null_key: str) -> dict:
        """Paired bootstrap of (BAS - its own shift null) over clips."""
        a = np.array([r[a_key] for r in rows]) - np.array([r[null_key] for r in rows])
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(4000)])
        return {"delta": float(a.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "excludes_zero": bool(np.percentile(bs, 2.5) > 0)}
    res = {
        "n_clips": len(rows),
        "elapsed_s": round(time.time() - t0, 1),
        "bas": {
            "oracle_mean": float(np.mean([r["bas_oracle"] for r in rows])),
            "oracle_rephased_mean": float(np.mean([r["bas_oracle_rephased"] for r in rows])),
            "beat_trigger_mean": float(np.mean([r["bas_bt"] for r in rows])),
            "metronome_mean": float(np.mean([r["metronome"] for r in rows])),
            "null_oracle_mean": float(np.mean([r["null_oracle"] for r in rows])),
            "null_bt_mean": float(np.mean([r["null_bt"] for r in rows])),
            "oracle_ratio_metronome": float(np.mean([r["bas_oracle"] / r["metronome"] for r in rows])),
            "bt_ratio_metronome": float(np.mean([r["bas_bt"] / r["metronome"] for r in rows])),
            "oracle_median_z": float(np.median([r["z_oracle"] for r in rows])),
            "bt_median_z": float(np.median([r["z_bt"] for r in rows])),
            "oracle_frac_z_gt3": float(np.mean([r["z_oracle"] > 3 for r in rows])),
            "bt_frac_z_gt3": float(np.mean([r["z_bt"] > 3 for r in rows])),
        },
        "bas_vs_own_null_paired": {
            "oracle_motion_only": paired("bas_oracle", "null_oracle"),
            "hybrid": paired("bas_hybrid", "null_hybrid"),
            "beat_trigger": paired("bas_bt", "null_bt"),
        },
        "event_rate": {
            "oracle_per_s": float(np.mean([r["n_light_events_oracle"] / r["duration_s"] for r in rows])),
            "hybrid_per_s": float(np.mean([r["n_light_events_hybrid"] / r["duration_s"] for r in rows])),
            "beat_trigger_per_s": float(np.mean([r["n_light_events_bt"] / r["duration_s"] for r in rows])),
        },
        "fid_light": {
            "split_half_floor": floor,
            "oracle_vs_block_shuffled_null": frechet(Zo, Zn),
            "oracle_vs_beat_trigger": frechet(Zo, Zb),
            "hybrid_vs_oracle": frechet(Zh, Zo),
            "hybrid_vs_beat_trigger": frechet(Zh, Zb),
            "div_hybrid": diversity(Zh, rng=rng),
            "div_oracle": diversity(Zo, rng=rng),
            "div_beat_trigger": diversity(Zb, rng=rng),
        },
        "brightness": {
            "oracle_mean": float(np.mean([r["level_oracle"]["mean"] for r in rows])),
            "oracle_p95": float(np.mean([r["level_oracle"]["p95"] for r in rows])),
            "bt_mean": float(np.mean([r["level_bt"]["mean"] for r in rows])),
            "bt_p95": float(np.mean([r["level_bt"]["p95"] for r in rows])),
            "hybrid_mean": float(np.mean([r["level_hybrid"]["mean"] for r in rows])),
            "hybrid_p95": float(np.mean([r["level_hybrid"]["p95"] for r in rows])),
        },
    }
    res["bas"]["hybrid_mean"] = float(np.mean([r["bas_hybrid"] for r in rows]))
    res["bas"]["hybrid_ratio_metronome"] = float(
        np.mean([r["bas_hybrid"] / r["metronome"] for r in rows]))
    g1 = res["bas"]["oracle_ratio_metronome"] >= 0.9
    g2 = floor["mean"] * 3 < res["fid_light"]["oracle_vs_block_shuffled_null"]
    g1b = res["bas_vs_own_null_paired"]["hybrid"]["excludes_zero"]
    res["gates"] = {
        "G1b_restated_hybrid_beats_shift_null": {"passed": bool(g1b),
            "evidence": f"hybrid BAS - null = "
                        f"{res['bas_vs_own_null_paired']['hybrid']['delta']:+.4f}, "
                        f"CI {res['bas_vs_own_null_paired']['hybrid']['ci95']}"},
        "G1_light_bas_ge_0.9_metronome": {"passed": bool(g1),
            "evidence": f"oracle BAS/metronome = {res['bas']['oracle_ratio_metronome']:.3f}"},
        "G2_split_half_x3_lt_null": {"passed": bool(g2),
            "evidence": f"floor {floor['mean']:.2f} x3 = {floor['mean']*3:.2f} vs null "
                        f"{res['fid_light']['oracle_vs_block_shuffled_null']:.2f}"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({"summary": res, "per_clip": rows},
                                                 indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


def render(n: int = 5) -> None:
    from lightsem.sim.renderer import RenderConfig, StageRenderer
    clips = sample_clips(40, seed=3)
    seen, chosen = set(), []
    for r in clips:
        if r["genre"] in seen or r["duration_s"] < 9.0:
            continue
        seen.add(r["genre"])
        chosen.append(r)
        if len(chosen) >= n:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    cond = ConductorV0()
    hyb = ConductorHybrid()
    rig = canonical_rig()
    ren = StageRenderer(rig, RenderConfig(width=960, height=540, fps=30))
    print("renderer:", ren.renderer_name)
    for r in chosen:
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        bus = load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz")
        dur = min(d.duration_s, 20.0)
        wav = f"/root/lightsem/data/wav/wav/{r['music_id']}.wav"
        sl = slice_bus(bus, dur)
        for label, frames in (("1_dance_oracle", cond.render(d).frames),
                              ("2_hybrid", hyb.render(d, sl).frames),
                              ("3_beat_trigger", render_beat_trigger(sl).frames)):
            out = OUT / f"{r['name']}__{label}.mp4"
            t0 = time.time()
            ren.render_video(frames, 40.0, out, audio_path=wav, duration_s=dur,
                             progress_every=0)
            print(f"  {out.name}  {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "metrics"
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None
    if cmd == "metrics":
        metrics(n or 120)
    elif cmd == "render":
        render(n or 5)
    else:
        raise SystemExit(f"unknown command {cmd}")
