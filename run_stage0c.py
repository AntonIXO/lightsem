#!/usr/bin/env python
"""Stage 0c — the two cheap human tests that shape the architecture (rb-ls2-00c).

    bash run.sh run_stage0c.py late     --n 8    # (i) how late may a light be?
    bash run.sh run_stage0c.py category --n 10   # (ii) dance-derived vs beat trigger

Both write brightness-matched, randomised video pairs plus a sealed KEY.json and
a score sheet. Neither can be answered by me: they need eyes and ears. What this
script guarantees is that when a human does answer, the answer means something —
matched brightness, blind order, the same music in both halves of a pair, and a
deliberately over-bright foil to catch a rater who is just picking the brighter
clip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from lightsem.conductor.beat_trigger import render_beat_trigger
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.conductor.v0 import ConductorV0
from lightsem.eval.ab import match_brightness, randomise, scale_brightness, write_key_and_form
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from lightsem.sim.renderer import RenderConfig, StageRenderer

OUT = Path("/root/lightsem/outputs/stage0c")
QUESTIONS_CATEGORY = ["Which one is ON THE MUSIC?",
                      "Which one looks like a LIGHTING ENGINEER did it?"]
QUESTIONS_LATE = ["Which one is ON THE MUSIC?"]


def pick(n, seed=7, min_dur=9.0):
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for i in rng.permutation(len(idx)):
        r = idx[i]
        key = (r["music_id"], r["choreo_id"])
        if key in seen or r["duration_s"] < min_dur:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


def _render(ren, frames, wav, dur, path):
    ren.render_video(frames, 40.0, path, audio_path=wav, duration_s=dur, progress_every=0)
    return str(path)


def late(n=8, delays=(0.040, 0.080, 0.120)):
    rig = canonical_rig()
    ren = StageRenderer(rig, RenderConfig(width=854, height=480, fps=30))
    out = OUT / "late_lights"
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for r in pick(n):
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        dur = min(d.duration_s, 8.0)
        bus = slice_bus(load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz"), dur)
        wav = f"/root/lightsem/data/wav/wav/{r['music_id']}.wav"
        base = render_beat_trigger(bus).frames
        f0 = _render(ren, base, wav, dur, out / f"{r['name']}__d0.mp4")
        for dl in delays:
            fr = render_beat_trigger(bus, delay_s=dl).frames
            fd = _render(ren, fr, wav, dur, out / f"{r['name']}__d{int(dl*1000)}.mp4")
            pairs.append((r["name"], "delay_0ms", f0, f"delay_{int(dl*1000)}ms", fd,
                          {"note": "same renderer, identical content, delay only"}))
    trials = randomise(pairs, seed=1)
    write_key_and_form(trials, out, QUESTIONS_LATE)
    print(f"{len(trials)} trials -> {out}")


def category(n=10, use_model=False):
    rig = canonical_rig()
    ren = StageRenderer(rig, RenderConfig(width=854, height=480, fps=30))
    out = OUT / ("category_model" if use_model else "category")
    out.mkdir(parents=True, exist_ok=True)
    cond = ConductorHybrid()
    v0 = ConductorV0()
    lm = None
    if use_model:
        from lightsem.model.infer import LightModel
        ck = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else None
        lm = LightModel(ck) if ck else LightModel()
        print(f"model checkpoint: {lm.ckpt if hasattr(lm, 'ckpt') else ck or 'default'}")
    pairs = []
    for r in pick(n):
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        dur = min(d.duration_s, 8.0)
        bus = slice_bus(load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz"), dur)
        wav = f"/root/lightsem/data/wav/wav/{r['music_id']}.wav"
        if use_model:
            dm = lm.descriptors(bus, seed=0)
            a = cond.render(dm, bus).frames
            label_a = "lightsem_model"
        else:
            a = cond.render(d, bus).frames
            label_a = "dance_derived_hybrid"
        b = render_beat_trigger(bus).frames
        a_m, b_m, match = match_brightness(a, b, rig)
        fa = _render(ren, a_m, wav, dur, out / f"{r['name']}__A.mp4")
        fb = _render(ren, b_m, wav, dur, out / f"{r['name']}__B.mp4")
        pairs.append((r["name"], label_a, fa, "beat_trigger", fb, match))
        # over-bright foil: the same beat trigger, 1.6x brighter, against itself
        foil = scale_brightness(b, rig, 1.6)
        ff = _render(ren, foil, wav, dur, out / f"{r['name']}__FOIL.mp4")
        pairs.append((r["name"] + "_FOIL", "beat_trigger", fb,
                      "beat_trigger_bright", ff, {"note": "identical content, 1.6x brighter"}))
    trials = randomise(pairs, seed=2)
    write_key_and_form(trials, out, QUESTIONS_CATEGORY)
    matched = [t for t in trials if t.match.get("matched")]
    print(f"{len(trials)} trials ({len(matched)} brightness-matched) -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "category"
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    if cmd == "late":
        late(n)
    elif cmd == "category":
        category(n, use_model="--model" in sys.argv)
