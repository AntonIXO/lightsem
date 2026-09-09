#!/usr/bin/env python
"""Sound in, light out, for any audio file: the product path rendered to MP4.

For each input file: the causal bus, the model's descriptors, the hybrid
conductor, the canonical rig, the headless renderer, the song's own audio
muxed in. Three videos per song -- the Just Dance-trained model, the AIST++-
trained model, and the 30-line beat trigger every claim is priced against.

  bash run_torch.sh render_song_demo.py <audio files...> [--out DIR]
"""
from __future__ import annotations

import argparse, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import extract_bus  # noqa: E402
from lightsem.conductor.beat_trigger import render_beat_trigger  # noqa: E402
from lightsem.conductor.hybrid import ConductorHybrid  # noqa: E402
from lightsem.conductor.rig import canonical_rig  # noqa: E402
from lightsem.model.infer import LightModel  # noqa: E402
from lightsem.sim.renderer import RenderConfig, StageRenderer  # noqa: E402

CK = {"model_jd": "/mnt/lightsem-data/ckpt/light_trunk_jd_none_smooth.pt",
      "model_aist": "/mnt/lightsem-data/ckpt/light_trunk_v3causal_smooth.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", default="/root/lightsem/outputs/stage8/demo_songs")
    ap.add_argument("--max-s", type=float, default=0.0, help="render only the first N seconds (0 = whole song)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ren = StageRenderer(canonical_rig(), RenderConfig(width=a.width, height=a.height, fps=30))
    cond = ConductorHybrid()
    models = {k: LightModel(v) for k, v in CK.items()}
    for src in a.audio:
        src = Path(src)
        name = re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"^[0-9a-f]{8}-\d+__", "", src.stem)).strip("_")
        wav = out / f"{name}.wav"
        cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "44100", "-y"]
        if a.max_s > 0:
            cmd += ["-t", str(a.max_s)]
        subprocess.run(cmd + [str(wav)], check=True)
        t0 = time.time()
        bus = extract_bus(wav)
        dur = bus.duration_s
        print(f"{name}: {dur:.0f} s, tempo {bus.tempo:.1f} bpm, bus in {time.time()-t0:.0f}s", flush=True)
        renders = {"beat_trigger": render_beat_trigger(bus).frames}
        for k, lm in models.items():
            renders[k] = cond.render(lm.descriptors(bus, seed=0), bus).frames
        for k, frames in renders.items():
            path = out / f"{name}__{k}.mp4"
            t1 = time.time()
            ren.render_video(frames, 40.0, path, audio_path=str(wav), duration_s=dur, progress_every=0)
            print(f"  wrote {path.name} ({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
