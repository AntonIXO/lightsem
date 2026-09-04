#!/usr/bin/env python
"""Stage 0a gate G3 — do video-derived descriptors match mocap-derived ones?

    bash run_torch.sh run_pose_validation.py --n 12

Downloads AIST camera-1 videos for clips whose AIST++ mocap we already have,
runs a 2D pose model over them, computes the SAME descriptors through the SAME
code, and correlates channel by channel. Gate: the transferable channels
(speed, laterality, spread) correlate above r = 0.8. If they do, dance video
becomes a data source instead of a licence problem.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from lightsem.motion.descriptors import extract
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors
from lightsem.motion.pose2d import NOT_TRANSFERABLE, TRANSFERABLE, keypoints_from_video, to_skeleton

VID = Path("/mnt/lightsem-data/aist_video")
OUT = Path("/root/lightsem/outputs/stage0a")
BASE = "https://storage.repository.aist.go.jp/1095/v1.0.0/video/10M"


def video_name(clip: str) -> str:
    return clip.replace("_cAll_", "_c01_") + ".mp4"


def fetch(clip: str) -> Path | None:
    VID.mkdir(parents=True, exist_ok=True)
    name = video_name(clip)
    dst = VID / name
    if dst.exists() and dst.stat().st_size > 100000:
        return dst
    r = subprocess.run(["curl", "-sL", "--max-time", "180", "-o", str(dst),
                        f"{BASE}/{name}"], capture_output=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 100000:
        dst.unlink(missing_ok=True)
        return None
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="yolo11m-pose.pt")
    args = ap.parse_args()
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    rng = np.random.default_rng(0)
    seen, picks = set(), []
    for i in rng.permutation(len(idx)):
        r = idx[i]
        if r["genre"] in seen or r["duration_s"] < 9:
            continue
        seen.add(r["genre"])
        picks.append(r)
        if len(picks) >= args.n:
            break

    rows = []
    for r in picks:
        v = fetch(r["name"])
        if v is None:
            print(f"  no video for {r['name']}")
            continue
        t0 = time.time()
        try:
            K, C, meta = keypoints_from_video(str(v), model_name=args.model)
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {r['name']}: {type(e).__name__} {e}")
            continue
        fps_v = 60.0
        sk = to_skeleton(K, C, fps_v, name=r["name"] + "_video")
        bus = load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz")
        d_vid = extract(sk, beats=bus.beats[bus.beats < sk.duration_s], tempo=bus.tempo)
        d_moc = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        n = min(len(d_vid.times), len(d_moc.times))
        row = {"clip": r["name"], "genre": r["genre"], "coverage": meta["coverage"],
               "n_tracks": meta["n_tracks"], "seconds": round(time.time() - t0, 1),
               "n_frames_video": int(len(d_vid.times)), "n_frames_mocap": int(len(d_moc.times)),
               "r": {}}
        for ch in list(TRANSFERABLE) + list(NOT_TRANSFERABLE):
            if ch not in d_vid.channels or ch not in d_moc.channels:
                continue
            a, b = d_vid.channels[ch][:n], d_moc.channels[ch][:n]
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                row["r"][ch] = float("nan")
            else:
                row["r"][ch] = float(np.corrcoef(a, b)[0, 1])
        rows.append(row)
        print(f"  {r['name'][:30]:30s} cov {meta['coverage']:.2f} tracks {meta['n_tracks']} "
              f"qom r {row['r'].get('qom', float('nan')):+.3f} "
              f"({row['seconds']:.0f}s)", flush=True)

    if not rows:
        print("no clips processed")
        return
    chans = sorted({c for r in rows for c in r["r"]})
    summary = {}
    print(f"\n{'channel':18s} {'mean r':>8s} {'median':>8s} {'n':>4s}  transferable?")
    for ch in chans:
        v = np.array([r["r"][ch] for r in rows if np.isfinite(r["r"].get(ch, np.nan))])
        if not v.size:
            continue
        summary[ch] = {"mean_r": float(v.mean()), "median_r": float(np.median(v)),
                       "n": int(v.size), "transferable": ch in TRANSFERABLE}
        print(f"{ch:18s} {v.mean():+8.3f} {np.median(v):+8.3f} {v.size:4d}  "
              f"{'yes' if ch in TRANSFERABLE else 'no (3D only)'}")
    core = ["qom", "head_speed", "hand_speed", "spread", "laterality"]
    got = [summary[c]["mean_r"] for c in core if c in summary]
    passed = bool(got and np.mean(got) > 0.8)
    res = {"per_clip": rows, "per_channel": summary, "n_clips": len(rows),
           "gate_G3": {"passed": passed,
                       "evidence": f"mean r over {core} = {np.mean(got):.3f} "
                                   f"(gate 0.8), {len(rows)} clips"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pose_validation.json").write_text(json.dumps(res, indent=1))
    print(f"\nG3: {'PASS' if passed else 'FAIL'} — {res['gate_G3']['evidence']}")


if __name__ == "__main__":
    main()
