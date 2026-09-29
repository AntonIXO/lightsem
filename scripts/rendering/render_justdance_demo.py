#!/usr/bin/env python
"""Viewable output: model light on held-out Just Dance songs, next to light
mapped from the real coach, with the song audio. Two files per song under
experiments/legacy/stages/outputs/stage8/demo/: <song>__model.mp4 and <song>__coach.mp4, 40 s from 60 s in.

  bash run_torch.sh scripts/rendering/render_justdance_demo.py [--n 2] [--ckpt ...]
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from lightsem.conductor.hybrid import ConductorHybrid  # noqa: E402
from lightsem.conductor.rig import canonical_rig  # noqa: E402
from lightsem.model.infer import LightModel  # noqa: E402
from lightsem.model.jd_data import make_jd_splits  # noqa: E402
from lightsem.model.light_data import load_tensor  # noqa: E402
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus  # noqa: E402
from lightsem.sim.renderer import RenderConfig, StageRenderer  # noqa: E402

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage8/demo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ckpt", default="/mnt/lightsem-data/ckpt/light_trunk_jd_none_smooth.pt")
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--dur", type=float, default=40.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    z, meta = load_tensor(CACHE_ROOT / "train_tensor_jd_none.npz")
    sp = make_jd_splits(meta)
    rng = np.random.default_rng(1)
    picks = [meta[i] for i in rng.permutation(sp.test_dancer)[:a.n]]
    lm = LightModel(a.ckpt)
    cond = ConductorHybrid()
    ren = StageRenderer(canonical_rig(), RenderConfig(width=854, height=480, fps=30))
    songs = {r["yt_id"]: r for r in json.loads((CACHE_ROOT / "justdance_songs.json").read_text())["rows"]}
    for m in picks:
        d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{m['name']}.npz")
        off = float(d.meta.get("start_s", 0.0))
        bus_full = load_bus(CACHE_ROOT / "justdance_audio" / f"{m['music_id']}.npz")
        t0 = off + a.start
        bus = slice_bus(bus_full, a.dur, offset_s=t0)
        wav = OUT / f"{m['music_id']}__clip.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(a.dur), "-i",
                        str(CACHE_ROOT / "justdance_wav" / f"{m['music_id']}.wav"), "-y", str(wav)], check=True)
        dm = lm.descriptors(bus, seed=0)
        model_frames = cond.render(dm, bus).frames
        # the coach: its descriptors on the same window
        i0 = int(round(a.start * d.fps)); i1 = i0 + int(round(a.dur * d.fps))
        dc = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{m['name']}.npz")
        for k in list(dc.channels):
            dc.channels[k] = dc.channels[k][i0:i1]
        dc.times = dc.times[:i1 - i0]
        dc.events = {k: (np.asarray(v) - a.start)[(np.asarray(v) >= a.start) & (np.asarray(v) < a.start + a.dur)] for k, v in dc.events.items()}
        coach_frames = cond.render(dc, bus).frames
        title = songs.get(m["music_id"], {}).get("title_norm", m["music_id"]).replace(" ", "_")[:40] or m["music_id"]
        for tag, frames in (("model", model_frames), ("coach", coach_frames)):
            path = OUT / f"{title}__{tag}.mp4"
            ren.render_video(frames, 40.0, path, audio_path=str(wav), duration_s=a.dur, progress_every=0)
            print("wrote", path, flush=True)
    (OUT / "README.md").write_text(
        "Held-out Just Dance songs, 40 s from 60 s in. `__model.mp4`: light from the Just Dance-trained "
        "model (light_trunk_jd_none_smooth) hearing only the audio. `__coach.mp4`: the same conductor "
        "driven by the real coach's motion. Same rig, same audio, same window.\n")


if __name__ == "__main__":
    main()
