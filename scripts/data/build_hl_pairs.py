#!/usr/bin/env python
"""H-L materials (PLAN.md 9.6): brightness-matched pairwise clips with knob perturbations, ready to rate.

Each pair = one held-out Just Dance song, a 30 s window, the shipped model through the hybrid
conductor twice: A and B differ only in the conductor's knobs (a random perturbation of up to 8 of
them from the defaults, or the defaults themselves), rendered with the SAME audio and matched on
mean brightness. One pair in ten is an over-bright foil (B at 1.6x the gain) to catch raters who
prefer "brighter". Writes experiments/legacy/stages/outputs/stage9/hl_pairs/<pair>__{A,B}.mp4 and pairs.json with the knob
vectors phi_A, phi_B (the Bradley-Terry features).

    bash run_torch.sh scripts/data/build_hl_pairs.py --n 40 --clip-s 30
"""
from __future__ import annotations

import argparse, dataclasses, json, time
from pathlib import Path

import numpy as np

from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.conductor.v0 import ConductorConfig
from lightsem.dmx.encode import pack_universe
from lightsem.dmx.decode import decode_stream
from lightsem.eval.light_features import level_features
from lightsem.model.infer import LightModel
from lightsem.model.jd_data import make_jd_splits
from lightsem.model.light_data import load_tensor
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from lightsem.sim.renderer import RenderConfig, StageRenderer

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9/hl_pairs")
CKPT = "/mnt/lightsem-data/ckpt/light_trunk_jd_none_smooth.pt"
#: the <= 12 knobs of the rule table, as (name, default, log-uniform range factor or absolute range)
KNOBS = {"base_level": (0.12, (0.02, 0.35)), "bounce_depth": (0.68, (0.2, 1.0)), "contrast_gain": (0.85, (0.4, 1.6)),
         "blinder_decay_s": (0.22, (0.08, 0.6)), "strobe_decay_s": (0.12, (0.05, 0.3)), "spatial_sigma": (0.42, (0.2, 0.9)),
         "slew_hue": (0.45, (0.1, 1.5)), "slew_wash_dimmer": (4.0, (1.0, 12.0)), "onset_percentile": (88.0, (75.0, 96.0))}


def sample_knobs(rng, k_perturb: int = 4) -> dict:
    phi = {k: v[0] for k, v in KNOBS.items()}
    for k in rng.choice(list(KNOBS), size=k_perturb, replace=False):
        lo, hi = KNOBS[k][1]
        phi[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return phi


def conductor_of(phi: dict) -> ConductorHybrid:
    cfg = ConductorConfig(base_level=phi["base_level"], bounce_depth=phi["bounce_depth"], contrast_gain=phi["contrast_gain"],
                          blinder_decay_s=phi["blinder_decay_s"], strobe_decay_s=phi["strobe_decay_s"], spatial_sigma=phi["spatial_sigma"])
    cfg.slew = dict(cfg.slew); cfg.slew["hue"] = phi["slew_hue"]; cfg.slew["wash_dimmer"] = phi["slew_wash_dimmer"]
    return ConductorHybrid(cfg=cfg, onset_percentile=phi["onset_percentile"])


def scale_brightness(frames: np.ndarray, rig, gain: float) -> np.ndarray:
    dec = decode_stream(rig, frames)
    P = {k: dec[k] for k in ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom")}
    P["gobo"] = np.zeros_like(P["dimmer"]); P["dimmer"] = np.clip(P["dimmer"] * gain, 0, 1)
    return pack_universe(rig, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--clip-s", type=float, default=30.0)
    ap.add_argument("--split", default="val_track")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    z, meta = load_tensor(CACHE_ROOT / "train_tensor_jd_none.npz"); sp = make_jd_splits(meta); idx = getattr(sp, a.split)
    idx = idx[rng.permutation(len(idx))]; by_name = {m["name"]: m for m in meta}
    lm = LightModel(CKPT); rig = canonical_rig(); ren = StageRenderer(rig, RenderConfig(width=854, height=480, fps=30))
    pairs = []; i = 0
    for k in idx:
        if len(pairs) >= a.n:
            break
        nm = meta[k]["name"]; d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz"); mid = by_name[nm]["music_id"]
        if d.duration_s < a.clip_s + 10:
            continue
        off = float(d.meta.get("start_s", 0.0)); start = float(rng.uniform(5.0, d.duration_s - a.clip_s - 5.0))
        bus = slice_bus(load_bus(CACHE_ROOT / "justdance_audio" / f"{mid}.npz"), a.clip_s, offset_s=off + start)
        if bus.beats.size < 6:
            continue
        desc = lm.descriptors(bus, seed=0)
        phi_a = {kk: v[0] for kk, v in KNOBS.items()} if rng.random() < 0.5 else sample_knobs(rng)
        phi_b = sample_knobs(rng)
        fa = conductor_of(phi_a).render(desc, bus).frames; fb = conductor_of(phi_b).render(desc, bus).frames
        la, lb = level_features(fa, rig)["mean"], level_features(fb, rig)["mean"]
        foil = rng.random() < 0.1
        g = float(np.clip(la / max(lb, 1e-3), 0.3, 3.0)) * (1.6 if foil else 1.0)
        fb = scale_brightness(fb, rig, g)
        wav = OUT / f"{nm}__{start:.0f}.wav"
        import subprocess
        src = next(Path("/mnt/lightsem-data/justdance").glob(f"*{mid}*"))
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(off + start), "-t", str(a.clip_s), "-i", str(src), "-vn", "-ac", "1", "-ar", "44100", "-y", str(wav)], check=True)
        pid = f"pair{len(pairs):03d}"
        order = rng.permutation(2)                     # which knob set is shown as A on screen
        shown = [(phi_a, fa), (phi_b, fb)]
        for tag, j in zip(("A", "B"), order):
            ren.render_video(shown[j][1], 40.0, OUT / f"{pid}__{tag}.mp4", audio_path=str(wav), duration_s=a.clip_s, progress_every=0)
        pairs.append({"id": pid, "song": nm, "music_id": mid, "start_s": start, "clip_s": a.clip_s,
                      "phi": {"A": shown[order[0]][0], "B": shown[order[1]][0]}, "foil_overbright": "B" if (foil and order[1] == 1) else ("A" if foil else None),
                      "brightness": {"A": level_features(shown[order[0]][1], rig)["mean"], "B": level_features(shown[order[1]][1], rig)["mean"]}})
        (OUT / "pairs.json").write_text(json.dumps({"knobs": list(KNOBS), "defaults": {kk: v[0] for kk, v in KNOBS.items()}, "pairs": pairs}, indent=1))
        print(f"[{len(pairs)}/{a.n}] {pid} {nm} @ {start:.0f}s foil={foil} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(pairs)} pairs in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
