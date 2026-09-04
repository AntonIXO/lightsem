#!/usr/bin/env python
"""Build the Stage 0a caches: audio bus per track, descriptors per dance clip.

    bash run.sh build_caches.py audio
    bash run.sh build_caches.py motion

Caches live on /mnt (root disk is ~89 % full). Audio must be built first:
the descriptor extractor needs each track's beat grid to compute the metrical
level of the dancer's vertical bounce.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from lightsem.audio.online import extract_bus
from lightsem.data.aist import AistDataset, parse_name
from lightsem.motion.descriptors import extract
from lightsem.motion.io import CACHE_ROOT, load_bus, save_bus, save_descriptors
from lightsem.motion.skeleton import prepare
from lightsem.eval.bas import circular_phase

AUDIO_DIR = CACHE_ROOT / "audio"
MOTION_DIR = CACHE_ROOT / "motion"


def build_audio(force: bool = False) -> None:
    ds = AistDataset()
    wavs = sorted(ds.wav_dir.glob("*.wav"))
    print(f"{len(wavs)} tracks -> {AUDIO_DIR}")
    for i, w in enumerate(wavs):
        out = AUDIO_DIR / f"{w.stem}.npz"
        if out.exists() and not force:
            continue
        t0 = time.time()
        bus = extract_bus(w)
        save_bus(bus, out)
        print(f"[{i+1}/{len(wavs)}] {w.stem} {bus.duration_s:.1f}s "
              f"tempo {bus.tempo:.1f} beats {bus.beats.size} ({time.time()-t0:.1f}s)",
              flush=True)


def build_motion(force: bool = False, limit: int | None = None) -> None:
    ds = AistDataset()
    files = ds.keypoint_files()
    if limit:
        files = files[:limit]
    print(f"{len(files)} clips -> {MOTION_DIR}")
    bus_cache: dict[str, object] = {}
    rows = []
    t_start = time.time()
    for i, p in enumerate(files):
        out = MOTION_DIR / f"{p.stem}.npz"
        fields = parse_name(p.stem)
        if fields is None:
            continue
        mid = fields["music_id"]
        if out.exists() and not force:
            continue
        if mid not in bus_cache:
            bp = AUDIO_DIR / f"{mid}.npz"
            if not bp.exists():
                print(f"  skip {p.stem}: no audio cache for {mid}")
                continue
            bus_cache[mid] = load_bus(bp)
        bus = bus_cache[mid]
        try:
            kp = pickle.load(open(p, "rb"))["keypoints3d_optim"]
            sk = prepare(np.asarray(kp, dtype=np.float64), 60.0, name=p.stem)
            dur = sk.duration_s
            beats = bus.beats[bus.beats < dur]
            d = extract(sk, beats=beats, tempo=bus.tempo)
            d.meta.update(fields)
            d.meta["tempo"] = float(bus.tempo)
            # Per-clip beat phase of the dance. NOT applied as a correction
            # here: it is a choreography property the audio cannot predict,
            # and silently shifting it would hide that fact from Stage 0b.
            ph = circular_phase(d.events["footfall"], beats)
            d.meta["beat_phase_R"] = float(ph["R"]) if ph["R"] == ph["R"] else 0.0
            d.meta["beat_offset_s"] = float(ph["offset_s"])
            save_descriptors(d, out)
            rows.append({"name": p.stem, **fields, "duration_s": dur,
                         "beat_phase_R": d.meta["beat_phase_R"],
                         "beat_offset_s": d.meta["beat_offset_s"],
                         "n_footfalls": d.meta["n_footfalls"],
                         "n_jumps": d.meta["n_jumps"], "tempo": float(bus.tempo)})
        except Exception as e:                      # noqa: BLE001
            print(f"  FAIL {p.stem}: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(files)}] {time.time()-t_start:.0f}s", flush=True)
    if rows:
        (CACHE_ROOT / "motion_index.json").write_text(json.dumps(rows, indent=1))
    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "audio"
    force = "--force" in sys.argv
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    if what == "audio":
        build_audio(force)
    elif what == "motion":
        build_motion(force, lim)
    else:
        raise SystemExit(f"unknown target {what}")
