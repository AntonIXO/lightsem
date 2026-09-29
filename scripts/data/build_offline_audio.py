#!/usr/bin/env python
"""Offline (L = whole track) audio bus for the AIST++ tracks: the same
extract_bus with causal=False, into its own cache directory, for H-A.

  bash run.sh scripts/data/build_offline_audio.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/root/lightsem")
from lightsem.audio.online import extract_bus
from lightsem.motion.io import CACHE_ROOT, save_bus
from lightsem.data.aist import AistDataset

OUT = CACHE_ROOT / "audio_offline"
OUT.mkdir(parents=True, exist_ok=True)
ds = AistDataset()
wavs = sorted(ds.wav_dir.glob("*.wav"))
t0 = time.time()
for i, w in enumerate(wavs):
    o = OUT / f"{w.stem}.npz"
    if o.exists():
        continue
    bus = extract_bus(w, causal=False)
    save_bus(bus, o)
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{len(wavs)}] {time.time()-t0:.0f}s", flush=True)
print(f"done {len(list(OUT.glob('*.npz')))} offline buses in {time.time()-t0:.0f}s")
