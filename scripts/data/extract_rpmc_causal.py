#!/usr/bin/env python
"""RPMC-L2 -> the product's own causal audio features, at the light frame rate.

The Stage 2 probe used the openl3 embeddings the corpus ships. Those are fine
for asking whether a signal exists and wrong for building the deployed head:
openl3 is a 1 s-window embedding and the product's conditioning is defined in
PLAN.md section 4 as causal log-mel plus banded spectral flux, onset and beat
phase. So the operator head is trained on features computed here from the
corpus's own mel spectrogram, by the same recipe `lightsem/audio/online.py`
applies to a wav file.

Band edges are scaled to this corpus's 22.05 kHz sample rate, which cuts the top
of Burger's high band (6.4-12.8 kHz) at Nyquist; the band is kept as 6.4 kHz to
Nyquist and named honestly.
"""
from __future__ import annotations

import json
import os
import sys
import time

import h5py
import librosa
import numpy as np

from scripts.data.extract_rpmc import reduce_light
from lightsem.audio.beatgrid import fit_constant_grid
from lightsem.audio.online import causal_resample, ema

H5 = "/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5"
OUT = "/mnt/lightsem-data/cache/rpmc_causal"
SR, HOP = 22050, 512
FPS_OUT = 10.0

BANDS = {"lf": (50, 100), "lmid": (100, 400), "mid": (400, 1600),
         "hmid": (1600, 6400), "hf": (6400, 11025)}


def features_from_mel(mel_db: np.ndarray) -> tuple[dict, float]:
    """(128, L) log-mel -> causal scalar channels on the analysis grid."""
    fps = SR / HOP
    mel = librosa.db_to_power(mel_db)
    freqs = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0.0, fmax=SR / 2)
    logm = np.log1p(mel)
    flux = np.maximum(np.diff(logm, axis=1, prepend=logm[:, :1]), 0.0)
    ch = {"rms": np.sqrt(mel.mean(0)), "flux": flux.sum(0)}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        ch[f"flux_{name}"] = flux[m].sum(0) if m.any() else np.zeros(mel.shape[1])
        ch[f"energy_{name}"] = mel[m].mean(0) if m.any() else np.zeros(mel.shape[1])
    ch["onset"] = librosa.onset.onset_strength(S=mel_db, sr=SR, hop_length=HOP)[: mel.shape[1]]
    tot = np.maximum(mel.sum(0), 1e-9)
    ch["centroid"] = (mel * freqs[:, None]).sum(0) / tot
    ch["flatness"] = np.exp(np.log(mel + 1e-9).mean(0)) / (mel.mean(0) + 1e-9)
    cs = np.cumsum(mel, axis=0) / tot
    ch["rolloff"] = freqs[np.argmax(cs >= 0.95, axis=0)]
    grid = fit_constant_grid(ch["onset"], fps, phase_onset=ch["flux_lf"])
    return ch, grid


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    f = h5py.File(H5, "r")
    keys = list(f.keys())[:limit]
    print(f"{len(keys)} songs -> {OUT}", flush=True)
    t0, ok, fail = time.time(), 0, 0
    for i, k in enumerate(keys):
        dst = os.path.join(OUT, f"{k[:16]}.npz")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            g = f[k]
            mel_db = g["music"]["mel_spectrogram_db"][:].astype(np.float32)
            y = reduce_light(g["light"]["0"][:])                 # (F, 6) at 30 fps
            ch, grid = features_from_mel(mel_db)
            fps_in = SR / HOP
            n_out = int(round(y.shape[0] / 30.0 * FPS_OUT))
            t_out = np.arange(n_out) / FPS_OUT
            t_in = np.arange(mel_db.shape[1]) / fps_in
            cols, names = [], []
            for name, v in ch.items():
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                for tau in (0.2, 1.0, 4.0):
                    # Hold, not interpolate: interpolating onto the 10 fps grid
                    # reads the next analysis frame, which is 23 ms of the
                    # future in a file that calls itself causal.
                    cols.append(causal_resample(ema(v, fps_in, tau), t_in, t_out))
                    names.append(f"{name}_ema{tau}")
            # The grid is fitted over the whole song, so beat_sin/cos/accent/
            # period are an ORACLE beat, exactly as in lightsem.audio.online --
            # everything else here is causal, these four are not, and a
            # deployment replaces them with a running tracker.
            beats = grid["beats"]
            per = grid["period_s"]
            ph = np.mod((t_out - grid["phase_s"]) / per, 1.0)
            cols += [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph),
                     (1 - ph) ** 2.5, np.full(n_out, per)]
            names += ["beat_sin", "beat_cos", "beat_accent", "beat_period_s"]
            X = np.stack(cols, 1).astype(np.float32)
            idx = np.linspace(0, y.shape[0] - 1, n_out)
            lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
            w = (idx - lo).astype(np.float32)[:, None]
            Y = (y[lo] * (1 - w) + y[hi] * w).astype(np.float32)
            np.savez_compressed(dst, x=X, y=Y, names=np.array(names),
                                bpm=np.float32(grid["bpm"]))
            ok += 1
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  FAIL {k[:16]}: {type(e).__name__} {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(keys)} ok={ok} fail={fail} {time.time()-t0:.0f}s",
                  flush=True)
    json.dump({"ok": ok, "fail": fail}, open(f"{OUT}/manifest.json", "w"))
    print(f"DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
