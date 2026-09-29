#!/usr/bin/env python
"""Whole-song structure labels for the Just Dance captures (H-C conditioning).

Laplacian segmentation plus the heuristic EDM role labeller from
lightsem/audio/structure.py, run on each capture's wav. Output per song: the
segments with roles, and two 40 fps conditioning channels -- a role index and
the signed seconds to the next drop start (negative after the last drop).
The labeller is a heuristic and has never been scored against ground truth;
H-C's shuffle control is what keeps a claim built on it honest.

  bash run.sh scripts/data/build_justdance_structure.py [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")

CACHE = Path("/mnt/lightsem-data/cache")
WAV_DIR = CACHE / "justdance_wav"
OUT_DIR = CACHE / "justdance_structure"
ROLES = ("intro", "breakdown", "build-up", "drop", "outro")
FPS = 40.0


def one(yt: str):
    out = OUT_DIR / f"{yt}.npz"
    if out.exists():
        return yt, "exists"
    try:
        from lightsem.audio.features import extract
        from lightsem.audio.structure import segment
        feats = extract(path=str(WAV_DIR / f"{yt}.wav"))
        # 6 clusters and 12 s minimum: pop songs come out as 12-20 sections.
        # The EDM labeller calls every loud section a "drop", so the energy
        # level and the distance to the next energy rise are carried as well.
        segs = segment(feats, n_segments=6, min_seg_s=12.0)
        dur = float(feats.frame_times[-1]) if len(feats.frame_times) else 0.0
        n = int(round(dur * FPS))
        t = np.arange(n) / FPS
        role = np.zeros(n, dtype=np.int8)
        energy = np.zeros(n, dtype=np.float32)
        rows = []
        for sg in segs:
            a, b, lab = float(sg.start_s), float(sg.end_s), str(sg.label)
            m = (t >= a) & (t < b)
            role[m] = ROLES.index(lab) if lab in ROLES else 0
            energy[m] = float(sg.energy)
            rows.append({"start_s": a, "end_s": b, "role": lab, "energy": float(sg.energy),
                         "brightness": float(sg.brightness), "density": float(sg.density)})

        def to_next(starts):
            out_ = np.full(n, np.nan, dtype=np.float32)
            if starts.size:
                nxt = np.searchsorted(starts, t, side="right")
                has = nxt < starts.size
                out_[has] = starts[nxt[has]] - t[has]
                out_[~has] = starts[-1] - t[~has]       # negative after the last one
            return out_

        bounds = np.array([r["start_s"] for r in rows[1:]])
        rises = np.array([rows[i]["start_s"] for i in range(1, len(rows))
                          if rows[i]["energy"] > rows[i - 1]["energy"] + 0.05])
        drops = np.array([r["start_s"] for r in rows if r["role"] == "drop"])
        np.savez_compressed(out, t=t.astype(np.float32), role=role, energy=energy,
                            to_boundary_s=to_next(bounds), to_rise_s=to_next(rises), to_drop_s=to_next(drops),
                            meta=json.dumps({"segments": rows, "duration_s": dur, "roles": ROLES}))
        return yt, f"ok {len(rows)} segs, {int(rises.size)} rises, {int(drops.size)} drops"
    except Exception as e:  # noqa: BLE001
        return yt, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yts = sorted(p.stem for p in WAV_DIR.glob("*.wav"))
    if a.limit:
        yts = yts[:a.limit]
    t0 = time.time(); n_ok = 0
    with Pool(a.workers) as pool:
        for i, (yt, msg) in enumerate(pool.imap_unordered(one, yts)):
            n_ok += msg.startswith("ok") or msg == "exists"
            if not (msg.startswith("ok") or msg == "exists") or i % 25 == 0:
                print(f"[{i+1}/{len(yts)}] {yt} {msg}", flush=True)
    print(f"done {n_ok}/{len(yts)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
