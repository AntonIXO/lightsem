#!/usr/bin/env python
"""Build the Stage 0a caches: audio bus per track, descriptors per dance clip.

    bash run.sh scripts/data/build_caches.py audio
    bash run.sh scripts/data/build_caches.py motion

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


def merge_index(rows: list[dict], index_path: Path) -> tuple[int, list[dict]]:
    """Fold this run's rows into the index on disk. Merge, never replace.

    `--limit` never visits the tail of the corpus at all, so even a run that
    indexed every clip it looked at would still drop the ones it did not. Ten
    downstream scripts enumerate the corpus from this file, and a truncated one
    silently turns every corpus-wide number into a number about a subset.
    """
    merged: dict[str, dict] = {}
    if index_path.exists():
        try:
            merged = {r["name"]: r for r in json.loads(index_path.read_text())}
        except Exception as e:                          # noqa: BLE001
            print(f"  existing index unreadable ({type(e).__name__}), rewriting")
    n_prior = len(merged)
    merged.update({r["name"]: r for r in rows})
    out = list(merged.values())
    if out:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(out, indent=1))
    return n_prior, out


def _row_from_cache(path: Path, fields: dict) -> dict | None:
    """Rebuild an index row from a descriptor cache that already exists.

    Without this, a run that skips an already-cached clip contributes no row,
    and the index written at the end would list only what this run happened to
    touch. Ten downstream scripts enumerate the corpus from that file.
    """
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        fps = float(meta.get("fps", 40.0))
        dur = float(meta.get("duration_s", z["times"].shape[0] / fps))
    except Exception as e:                              # noqa: BLE001
        print(f"  unreadable cache {path.name}: {type(e).__name__}: {e}")
        return None
    return {"name": path.stem, **fields, "duration_s": dur,
            "beat_phase_R": float(meta.get("beat_phase_R", 0.0)),
            "beat_offset_s": float(meta.get("beat_offset_s", 0.0)),
            "n_footfalls": int(meta.get("n_footfalls", 0)),
            "n_jumps": int(meta.get("n_jumps", 0)),
            "tempo": float(meta.get("tempo", 0.0))}


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
            row = _row_from_cache(out, fields)
            if row is not None:
                rows.append(row)
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
            d.meta["duration_s"] = float(dur)
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
    n_prior, merged = merge_index(rows, CACHE_ROOT / "motion_index.json")
    print(f"index: {n_prior} prior + {len(rows)} seen this run -> {len(merged)} clips")
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
