#!/usr/bin/env python
"""Audio side of the Just Dance corpus: one wav and one causal bus per capture.

CPU work, meant to run while the GPU does pose. The wav is kept (44.1 kHz
mono, the bus rate) because the offline bus, the structure labeller and any
render with sound all need it again; the video stays the source of truth.

  bash run.sh build_justdance_audio.py [--workers 6] [--limit N] [--offline]
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "/root/lightsem")

JD = Path("/mnt/lightsem-data/justdance")
CACHE = Path("/mnt/lightsem-data/cache")
WAV_DIR = CACHE / "justdance_wav"
BUS_DIR = CACHE / "justdance_audio"
BUS_OFF_DIR = CACHE / "justdance_audio_offline"
MANIFEST = CACHE / "justdance_manifest.json"


def usable_rows():
    rows = json.loads(MANIFEST.read_text())
    return [r for r in rows if r.get("yt_id") and "error" not in r
            and (r.get("mean_db") if r.get("mean_db") is not None else -99) > -40]


def one(job):
    yt, fn, offline = job
    from lightsem.audio.online import extract_bus
    from lightsem.motion.io import save_bus
    wav = WAV_DIR / f"{yt}.wav"
    out = (BUS_OFF_DIR if offline else BUS_DIR) / f"{yt}.npz"
    if out.exists():
        return yt, "exists", 0.0
    t0 = time.time()
    if not wav.exists():
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(JD / fn), "-vn", "-ac", "1",
                            "-ar", "44100", "-y", str(wav)], capture_output=True, text=True)
        if r.returncode != 0:
            return yt, f"ffmpeg: {r.stderr[:120]}", time.time() - t0
    try:
        bus = extract_bus(str(wav), causal=not offline)
        save_bus(bus, out)
    except Exception as e:  # noqa: BLE001
        return yt, f"{type(e).__name__}: {e}", time.time() - t0
    return yt, f"ok tempo {bus.tempo:.1f}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    for d in (WAV_DIR, BUS_DIR, BUS_OFF_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rows = usable_rows()
    if a.limit:
        rows = rows[:a.limit]
    jobs = [(r["yt_id"], r["file"], a.offline) for r in rows]
    print(f"{len(jobs)} captures -> {'offline' if a.offline else 'causal'} bus, {a.workers} workers", flush=True)
    t0 = time.time(); n_ok = 0
    with Pool(a.workers) as p:
        for i, (yt, msg, dt) in enumerate(p.imap_unordered(one, jobs)):
            n_ok += msg.startswith("ok") or msg == "exists"
            if not (msg.startswith("ok") or msg == "exists") or i % 25 == 0:
                print(f"[{i+1}/{len(jobs)}] {yt} {msg} {dt:.0f}s", flush=True)
    print(f"done {n_ok}/{len(jobs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
