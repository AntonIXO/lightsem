#!/usr/bin/env python
"""H-F: the Just Dance background as a designed light show, in RPMC-L2's format.

Per capture: HSV histograms of every frame with the coaches (pose boxes,
dilated) and the UI masked out, reduced by `extract_rpmc.reduce_light` to the
same 6-d scene vector RPMC-L2 uses (hue cos/sin, saturation, value, value
p90, hue concentration), and the same causal 10 Hz audio features
`extract_rpmc_causal` computes for the operator head, so
`train_operator_head.py --root .../justdance_visual` runs unchanged. Needs
the pose cache (for the coach boxes), so it follows the extraction.

  bash run.sh build_justdance_visual.py [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
# extract_rpmc.py reads sys.argv[1] at import time (its song limit); hide ours.
_argv, sys.argv = sys.argv, sys.argv[:1]
from extract_rpmc import reduce_light  # noqa: E402
from extract_rpmc_causal import features_from_mel, SR, HOP, FPS_OUT  # noqa: E402
sys.argv = _argv

CACHE = Path("/mnt/lightsem-data/cache")
POSE_DIR = CACHE / "justdance_pose"
WAV_DIR = CACHE / "justdance_wav"
OUT_DIR = CACHE / "justdance_visual"
UI = ((0.00, 0.00, 1.00, 0.14),    # score / feedback band
      (0.70, 0.72, 1.00, 1.00),    # pictogram strip
      (0.00, 0.85, 0.45, 1.00),    # lyrics
      (0.00, 0.30, 0.09, 0.80))    # star meter


def scene_vector(path: Path, K, C, box, stride_check: int | None = None):
    """(F, 6) scene vector of the masked background, one row per decoded frame
    at the pose cache's stride; also returns the number of frames."""
    import cv2
    from extract_justdance import decode
    hist = []
    for i, (t, img, fps, stride) in enumerate(decode(path)):
        if i >= len(box):
            break
        H, W = img.shape[:2]
        small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        for x0, y0, x1, y1 in UI:
            mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        for p in range(box.shape[1]):
            b = box[i, p]
            if np.isfinite(b).all():
                bw, bh = (b[2] - b[0]) * 0.15, (b[3] - b[1]) * 0.15
                mask[max(int((b[1] - bh) / 2), 0):int((b[3] + bh) / 2) + 1,
                     max(int((b[0] - bw) / 2), 0):int((b[2] + bw) / 2) + 1] = 0
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        L = np.zeros((3, 256), dtype=np.float32)
        L[0, :180] = cv2.calcHist([hsv], [0], mask, [180], [0, 180]).ravel()
        L[1] = cv2.calcHist([hsv], [1], mask, [256], [0, 256]).ravel()
        L[2] = cv2.calcHist([hsv], [2], mask, [256], [0, 256]).ravel()
        hist.append(L)
    Lh = np.stack(hist) if hist else np.zeros((0, 3, 256), np.float32)
    return reduce_light(Lh), len(hist)


def one(yt: str):
    out = OUT_DIR / f"{yt}.npz"
    if out.exists():
        return yt, "exists", 0.0
    t0 = time.time()
    try:
        import librosa
        from lightsem.audio.online import causal_resample, ema
        z = np.load(POSE_DIR / f"{yt}.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        K, C, box, tt = z["K"], z["C"], z["box"], z["t"]
        if K.ndim == 3:
            K, C, box = K[:, None], C[:, None], box[:, None]
        src = json.loads((CACHE / "justdance_manifest.json").read_text())
        fn = next(r["file"] for r in src if r.get("yt_id") == yt)
        y6, n = scene_vector(Path("/mnt/lightsem-data/justdance") / fn, K, C, box)
        if n < 300:
            return yt, f"too few frames {n}", time.time() - t0
        t_vid = tt[:n]
        # audio features exactly as for the operators
        au, _ = librosa.load(str(WAV_DIR / f"{yt}.wav"), sr=SR)
        mel = librosa.feature.melspectrogram(y=au, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)
        mel_db = librosa.power_to_db(mel).astype(np.float32)
        ch, grid = features_from_mel(mel_db)
        fps_in = SR / HOP
        dur = min(float(t_vid[-1]), mel_db.shape[1] / fps_in)
        n_out = int(round(dur * FPS_OUT))
        t_out = np.arange(n_out) / FPS_OUT
        t_in = np.arange(mel_db.shape[1]) / fps_in
        cols, names = [], []
        for name, v in ch.items():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for tau in (0.2, 1.0, 4.0):
                cols.append(causal_resample(ema(v, fps_in, tau), t_in, t_out))
                names.append(f"{name}_ema{tau}")
        per = grid["period_s"]
        ph = np.mod((t_out - grid["phase_s"]) / per, 1.0)
        cols += [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), (1 - ph) ** 2.5, np.full(n_out, per)]
        names += ["beat_sin", "beat_cos", "beat_accent", "beat_period_s"]
        X = np.stack(cols, 1).astype(np.float32)
        Y = np.stack([np.interp(t_out, t_vid, y6[:, k]) for k in range(6)], 1).astype(np.float32)
        np.savez_compressed(out, x=X, y=Y, names=np.array(names), bpm=np.float32(grid["bpm"]),
                            meta=json.dumps({"n_video_frames": int(n), "fps_eff": meta.get("fps_eff"),
                                             "n_coaches": int(box.shape[1]), "duration_s": dur}))
        return yt, f"ok {n} frames, {n_out} rows", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return yt, f"{type(e).__name__}: {e}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yts = sorted(p.stem for p in POSE_DIR.glob("*.npz") if (WAV_DIR / f"{p.stem}.wav").exists())
    yts = [y for y in yts if not (OUT_DIR / f"{y}.npz").exists()]
    if a.limit:
        yts = yts[:a.limit]
    print(f"{len(yts)} captures -> {OUT_DIR}", flush=True)
    t0 = time.time(); n_ok = 0
    with Pool(a.workers) as pool:
        for i, (yt, msg, dt) in enumerate(pool.imap_unordered(one, yts)):
            n_ok += msg.startswith("ok")
            print(f"[{i+1}/{len(yts)}] {yt} {msg} {dt:.0f}s", flush=True)
    print(f"done {n_ok}/{len(yts)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
