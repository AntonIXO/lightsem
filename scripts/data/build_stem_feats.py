#!/usr/bin/env python
"""H-G, step 2: per-stem audio features for the Just Dance corpus, on both grids the audit uses.

For each stem (drums, bass, other, vocals) the SAME causal channel family the operator head
and the JD visual target already use (extract_rpmc_causal.features_from_mel: rms, flux, band
flux/energy, onset, centroid, flatness, rolloff), so a stem arm differs from the bus arm only by
WHICH SOURCE the channel was computed on -- the one thing PLAN.md 9.7 says can add anything.

  cache/justdance_stem_feats[_causal]/<yt>.npz
      x10      (T10, 4*16*3 + 3)  per-stem channels at EMA 0.2/1/4 s on the 10 Hz operator grid,
                                  plus vocal activity (vocal rms / mix rms) at the three EMAs
      names10  channel names
      x40      (T40, 4*16)         per-stem channels on the 40 fps bus grid (raw, causal resample)
      names40

  bash run.sh scripts/data/build_stem_feats.py [--mode whole|causal] [--workers 4]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
_argv, sys.argv = sys.argv, sys.argv[:1]
from scripts.data.extract_rpmc_causal import features_from_mel, SR, HOP, FPS_OUT  # noqa: E402
sys.argv = _argv
from lightsem.audio.online import causal_resample, ema  # noqa: E402

CACHE = Path("/mnt/lightsem-data/cache")
STEMS = {"whole": CACHE / "justdance_stems", "causal": CACHE / "justdance_stems_causal"}
OUTS = {"whole": CACHE / "justdance_stem_feats", "causal": CACHE / "justdance_stem_feats_causal"}
EMAS10 = (0.2, 1.0, 4.0)


def one(job):
    yt, mode = job
    out = OUTS[mode] / f"{yt}.npz"
    if out.exists():
        return yt, "exists", 0.0
    t0 = time.time()
    try:
        import librosa
        z = np.load(STEMS[mode] / f"{yt}.npz", allow_pickle=False)
        stems = z["stems"].astype(np.float32) / 32767.0
        sr = int(z["sr"]); assert sr == SR, sr
        srcs = [str(s) for s in z["sources"]]
        vis = np.load(CACHE / "justdance_visual" / f"{yt}.npz", allow_pickle=False)
        T10 = len(vis["x"]); t10 = np.arange(T10) / FPS_OUT
        dur = stems.shape[1] / sr
        T40 = int(round(dur * 40.0)); t40 = np.arange(T40) / 40.0
        fps_in = SR / HOP
        cols10, n10, cols40, n40 = [], [], [], []
        rms = {}
        for si, s in enumerate(srcs):
            mel = librosa.feature.melspectrogram(y=stems[si], sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)
            ch, _ = features_from_mel(librosa.power_to_db(mel).astype(np.float32))
            t_in = np.arange(mel.shape[1]) / fps_in
            rms[s] = ch["rms"]
            for name, v in ch.items():
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                for tau in EMAS10:
                    cols10.append(causal_resample(ema(v, fps_in, tau), t_in, t10)); n10.append(f"{s}_{name}_ema{tau}")
                cols40.append(causal_resample(v, t_in, t40)); n40.append(f"{s}_{name}")
        mix = np.sqrt(sum(rms[s] ** 2 for s in srcs)) + 1e-6
        va = rms["vocals"] / mix
        for tau in EMAS10:
            cols10.append(causal_resample(ema(va, fps_in, tau), t_in, t10)); n10.append(f"vocal_act_ema{tau}")
        cols40.append(causal_resample(va, t_in, t40)); n40.append("vocal_act")
        np.savez_compressed(out, x10=np.stack(cols10, 1).astype(np.float32), names10=np.array(n10),
                            x40=np.stack(cols40, 1).astype(np.float32), names40=np.array(n40),
                            meta=json.dumps({"mode": mode, "duration_s": dur, "seg_s": float(z["seg_s"]), "hop_s": float(z["hop_s"])}))
        return yt, f"ok {T10} rows", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return yt, f"{type(e).__name__}: {e}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("whole", "causal"), default="whole")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUTS[a.mode].mkdir(parents=True, exist_ok=True)
    yts = sorted(p.stem for p in STEMS[a.mode].glob("*.npz") if (CACHE / "justdance_visual" / f"{p.stem}.npz").exists())
    print(f"{len(yts)} songs ({a.mode}), {a.workers} workers", flush=True)
    t0 = time.time(); n_ok = 0
    with Pool(a.workers) as pool:
        for i, (yt, msg, dt) in enumerate(pool.imap_unordered(one, [(y, a.mode) for y in yts])):
            n_ok += msg.startswith("ok") or msg == "exists"
            if not (msg.startswith("ok") or msg == "exists") or i % 25 == 0:
                print(f"[{i+1}/{len(yts)}] {yt} {msg} {dt:.0f}s", flush=True)
    print(f"done {n_ok}/{len(yts)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
