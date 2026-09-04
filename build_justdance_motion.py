#!/usr/bin/env python
"""Pose caches -> per-coach descriptor caches for the Just Dance corpus.

One entry per coach, `<yt_id>_c<p>`, in its own cache directory and index so
the AIST++ index (which ten scripts enumerate and which the H-A run needs
unchanged) is never touched. The descriptor path is the one validated on AIST
video against mocap: `to_skeleton` (interpolate, embed as (lat, up, 0)),
`prepare` (torso-normalise, 12 Hz zero-phase Butterworth), `extract`.

  bash run.sh build_justdance_motion.py [--min-cov 0.8] [--force]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from lightsem.motion.io import load_bus, save_descriptors  # noqa: E402
from lightsem.motion.pose2d import to_skeleton  # noqa: E402
from lightsem.motion.descriptors import extract  # noqa: E402
from lightsem.eval.bas import circular_phase  # noqa: E402

CACHE = Path("/mnt/lightsem-data/cache")
POSE_DIR = CACHE / "justdance_pose"
BUS_DIR = CACHE / "justdance_audio"
OUT_DIR = CACHE / "justdance_motion"
INDEX = CACHE / "justdance_motion_index.json"
SONGS = CACHE / "justdance_songs.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    songs = {r["yt_id"]: r for r in json.loads(SONGS.read_text())["rows"]}
    prior = {r["name"]: r for r in json.loads(INDEX.read_text())} if INDEX.exists() else {}
    rows: dict[str, dict] = dict(prior)
    files = sorted(POSE_DIR.glob("*.npz"))
    if a.limit:
        files = files[:a.limit]
    t0 = time.time(); n_new = n_skip = n_fail = 0
    for i, pf in enumerate(files):
        yt = pf.stem
        bp = BUS_DIR / f"{yt}.npz"
        if not bp.exists():
            continue
        z = np.load(pf, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        K, C = z["K"], z["C"]
        if K.ndim == 3:                       # single-coach layout from the trial era
            K, C = K[:, None], C[:, None]
        P = K.shape[1]
        fps = float(meta["fps_eff"])
        bus = None
        # The routine is the span where any coach is on screen; the menu lead-in
        # and the results screen are not the dancer's fault. Trim to it, and
        # carry the offset so the audio bus is sliced to the same span.
        seen = np.isfinite(K[:, :, 0, 0]).any(axis=1)
        if not seen.any():
            n_skip += P; continue
        f0, f1 = int(np.argmax(seen)), int(len(seen) - np.argmax(seen[::-1]))
        t_all = z["t"]
        t_off = float(t_all[f0]) if len(t_all) else f0 / fps
        K, C = K[f0:f1], C[f0:f1]
        for p in range(P):
            name = f"{yt}_c{p}"
            out = OUT_DIR / f"{name}.npz"
            cov = float(np.isfinite(K[:, p, 0, 0]).mean())
            if cov < a.min_cov:
                n_skip += 1
                rows.pop(name, None)
                continue
            if out.exists() and not a.force and name in rows:
                continue
            try:
                if bus is None:
                    bus = load_bus(bp)
                stats: dict = {}
                sk = to_skeleton(K[:, p].astype(np.float64), C[:, p].astype(np.float64), fps,
                                 name=name, stats=stats)
                dur = sk.duration_s
                beats = bus.beats - t_off
                beats = beats[(beats >= 0) & (beats < dur)]
                d = extract(sk, beats=beats, tempo=bus.tempo)
                song = songs.get(yt, {})
                fields = {"corpus": "jd", "genre": "jd", "setting": "jd", "camera": "c",
                          "dancer": f"{yt}_c{p}", "music_id": yt, "choreo_id": f"c{p}",
                          "song_key": song.get("song_key", yt), "n_coaches": int(P),
                          "coach_index": int(p), "view": "frontal", "view_provenance": "asserted",
                          "coverage": cov, "coverage_whole_file": float(meta.get("coverage_each", [0.0])[p]),
                          "start_s": t_off, "end_s": t_off + dur, "fps_src": meta.get("fps_src"), "fps_eff": fps,
                          "pose_model": meta.get("model"), "pose_imgsz": meta.get("imgsz"),
                          "filled_frac": stats.get("filled_frac"), "long_gap_frac": stats.get("long_gap_frac")}
                d.meta.update(fields)
                d.meta["tempo"] = float(bus.tempo)
                ph = circular_phase(d.events["footfall"], beats)
                d.meta["beat_phase_R"] = float(ph["R"]) if ph["R"] == ph["R"] else 0.0
                d.meta["beat_offset_s"] = float(ph["offset_s"])
                d.meta["duration_s"] = float(dur)
                save_descriptors(d, out)
                rows[name] = {"name": name, **fields, "duration_s": float(dur),
                              "beat_phase_R": d.meta["beat_phase_R"], "beat_offset_s": d.meta["beat_offset_s"],
                              "n_footfalls": int(d.meta.get("n_footfalls", 0)), "n_jumps": int(d.meta.get("n_jumps", 0)),
                              "tempo": float(bus.tempo)}
                n_new += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(files)}] {time.time()-t0:.0f}s new {n_new} skip {n_skip} fail {n_fail}", flush=True)
    INDEX.write_text(json.dumps(sorted(rows.values(), key=lambda r: r["name"]), indent=1))
    print(f"index {INDEX}: {len(rows)} entries ({n_new} new, {n_skip} below coverage, {n_fail} failed) "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
