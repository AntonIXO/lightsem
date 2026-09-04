#!/usr/bin/env python
"""
PROBE: Is dance-pose kinetic energy REDUNDANT with plain audio envelope?

The "Just Dance hack" assumes dancer motion is a free semantic encoder of music:
    audio -> (dancer) -> pose -> handcrafted formula -> light

Per multimodal-ai lecture 08: cross-modal transfer can ONLY move REDUNDANT
information. So the hack is only worth building if pose carries something the
audio envelope does not already give you for free via 3 lines of librosa.

Three measurements on AIST++ (real paired 3D motion + music, licensed):

  M1. r(KE(t), audio envelope) at several timescales.
      High r => the "kinetic energy -> dimmer" formula IS an audio envelope.

  M2. Ridge R^2 from a standard audio feature stack -> KE(t).
      How much of pose dynamics is linearly decodable from audio alone.

  M3. THE CEILING (the test nobody in the sketch proposed):
      AIST++ has the SAME music danced by MANY dancers/genres.
      r(KE_dancerA, KE_dancerB | same music) upper-bounds ANY audio->pose model,
      because audio cannot predict what differs between two dancers hearing it.

  Verdict logic:
      if M2 ~ M3  -> audio already extracts everything predictable. Pose is
                     redundant. The dataset hack buys nothing. Path A is dead.
      if M2 << M3 -> pose holds real, learnable, non-audio-obvious structure.
                     The hack is justified.
      if M3 low   -> audio->pose is ill-posed; a single dancer's motion is an
                     ARBITRARY choice of ground truth for light.
"""
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

import librosa
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

DATA = "/root/lightsem/data"
KP_FPS = 60.0
HOP = 512
SR = 22050
FPS = SR / HOP  # ~43 Hz analysis rate

# smoothing timescales in seconds: micro(beat) -> macro(phrase)
SCALES = [0.1, 0.5, 2.0, 8.0]


def parse_name(fn):
    """gBR_sBM_c01_d04_mBR0_ch01 -> dict"""
    m = re.match(r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)", os.path.basename(fn))
    if not m:
        return None
    g, s, c, d, mus, ch = m.groups()
    return dict(genre=g, sit=s, cam=c, dancer=d, music=mus, chore=ch)


def load_kp(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    for k in ("keypoints3d", "keypoints3d_optim"):
        if k in d:
            return np.asarray(d[k], dtype=np.float64)
    return None


def kinetic_energy(kp3d):
    """kp3d [T,J,3] -> KE(t) = sum_j ||dP_j/dt||^2, exactly the sketch's formula."""
    if kp3d is None or kp3d.ndim != 3 or len(kp3d) < 10:
        return None
    if not np.isfinite(kp3d).all():
        # AIST++ has NaN frames in some sequences; drop the sequence if heavily
        # corrupted, otherwise linearly interpolate the gaps.
        bad = ~np.isfinite(kp3d).all(axis=(1, 2))
        if bad.mean() > 0.10:
            return None
        idx = np.arange(len(kp3d))
        good = ~bad
        if good.sum() < 10:
            return None
        kp3d = kp3d.copy()
        for j in range(kp3d.shape[1]):
            for c in range(kp3d.shape[2]):
                kp3d[bad, j, c] = np.interp(idx[bad], idx[good], kp3d[good, j, c])
    # root-relative so camera/global translation doesn't dominate
    root = kp3d[:, :1, :]
    rel = kp3d - root
    v = np.diff(rel, axis=0) * KP_FPS
    ke = (v ** 2).sum(axis=(1, 2))
    if not np.isfinite(ke).all():
        return None
    return ke


def resample(x, n_out):
    if x is None or len(x) < 2:
        return None
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, n_out)
    return np.interp(dst, src, x)


def smooth(x, sec):
    w = max(1, int(round(sec * FPS)))
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def zs(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def audio_features(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y, hop_length=HOP))
    rms = librosa.feature.rms(S=S, hop_length=HOP)[0]
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S ** 2), sr=SR, hop_length=HOP)
    flux = np.concatenate([[0], np.sqrt(((np.diff(S, axis=1)) ** 2).sum(0))])
    cent = librosa.feature.spectral_centroid(S=S, sr=SR, hop_length=HOP)[0]
    flat = librosa.feature.spectral_flatness(S=S, hop_length=HOP)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), sr=SR, n_mfcc=20)
    n = min(len(rms), len(onset), len(flux), mfcc.shape[1], contrast.shape[1])
    feats = np.vstack([
        rms[:n], onset[:n], flux[:n], cent[:n], flat[:n],
        contrast[:, :n], mfcc[:, :n],
    ]).T
    return feats, rms[:n], onset[:n], n


def main():
    kp_files = sorted(glob.glob(os.path.join(DATA, "kp3d", "**", "*.pkl"), recursive=True))
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(DATA, "wav", "**", "*.wav"), recursive=True)}
    print(f"keypoint files: {len(kp_files)}   music files: {len(wavs)}", flush=True)
    if not kp_files or not wavs:
        print("MISSING DATA"); sys.exit(1)

    # one camera only (c01) - keypoints3d is camera-independent, avoid duplicates
    recs = []
    for p in kp_files:
        meta = parse_name(p)
        if meta and meta["music"] in wavs:
            recs.append((p, meta))
    print(f"paired sequences: {len(recs)}", flush=True)

    # cap work: sample music ids that have the most distinct dancers (for M3)
    by_music = defaultdict(list)
    for p, meta in recs:
        by_music[meta["music"]].append((p, meta))
    ranked = sorted(by_music.items(),
                    key=lambda kv: -len({m["dancer"] for _, m in kv[1]}))
    chosen = ranked[:12]
    print("music ids used:", [(k, len({m['dancer'] for _, m in v})) for k, v in chosen], flush=True)

    audio_cache = {}
    rows = []          # per-sequence aligned signals
    for mus, items in chosen:
        if mus not in audio_cache:
            audio_cache[mus] = audio_features(wavs[mus])
        feats, rms, onset, n = audio_cache[mus]

        # dedupe: one entry per (dancer, chore); keypoints3d identical across cameras
        seen = set()
        for p, meta in sorted(items, key=lambda x: x[0]):
            key = (meta["dancer"], meta["chore"], meta["genre"])
            if key in seen:
                continue
            seen.add(key)
            ke = kinetic_energy(load_kp(p))
            if ke is None:
                continue
            ke_r = resample(ke, n)
            if ke_r is None:
                continue
            rows.append(dict(music=mus, dancer=meta["dancer"], genre=meta["genre"],
                             chore=meta["chore"], ke=ke_r, feats=feats,
                             rms=rms, onset=onset))
    print(f"usable sequences: {len(rows)}", flush=True)
    if len(rows) < 10:
        print("TOO FEW"); sys.exit(1)

    # ---------------- M1: correlation with plain envelope ----------------
    print("\n=== M1: r(KE, audio envelope) by timescale ===")
    print(f"{'scale_s':>8} {'r_rms':>16} {'r_onset':>16}")
    m1 = {}
    for sc in SCALES:
        rr, ro = [], []
        for r in rows:
            k = zs(smooth(r["ke"], sc))
            a = zs(smooth(r["rms"], sc))
            o = zs(smooth(r["onset"], sc))
            if k.std() < 1e-9:
                continue
            rr.append(pearsonr(k, a)[0])
            ro.append(pearsonr(k, o)[0])
        rr, ro = np.array(rr), np.array(ro)
        m1[sc] = (np.nanmean(rr), np.nanmean(ro))
        print(f"{sc:>8.1f} {np.nanmean(rr):>8.3f}+-{np.nanstd(rr):<6.3f} "
              f"{np.nanmean(ro):>8.3f}+-{np.nanstd(ro):<6.3f}")

    # ---------------- M2: ridge audio -> KE, grouped CV by music ----------------
    print("\n=== M2: Ridge R^2 (audio features -> KE), GroupKFold by music ===")
    m2 = {}
    for sc in SCALES:
        X, Y, G = [], [], []
        for r in rows:
            f = np.column_stack([smooth(r["feats"][:, j], sc)
                                 for j in range(r["feats"].shape[1])])
            k = smooth(r["ke"], sc)
            if k.std() < 1e-9:
                continue
            X.append(f); Y.append(zs(k)); G.append(r["music"])
        Xc = np.vstack(X)
        Yc = np.concatenate(Y)
        Gc = np.concatenate([[g] * len(y) for g, y in zip(G, Y)])
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
        n_groups = len(set(Gc))
        cv = GroupKFold(n_splits=min(5, n_groups))
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)),
                                 Xc, Yc, groups=Gc, cv=cv)
        ss_res = ((Yc - pred) ** 2).sum()
        ss_tot = ((Yc - Yc.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        m2[sc] = r2
        print(f"  scale={sc:>4.1f}s  R^2={r2:>7.4f}   (r={np.sqrt(max(r2,0)):.3f})")

    # ---------------- M3: inter-dancer ceiling ----------------
    print("\n=== M3: CEILING - r(KE_A, KE_B) same music, different dancer ===")
    m3 = {}
    for sc in SCALES:
        vals = []
        for mus in {r["music"] for r in rows}:
            grp = [r for r in rows if r["music"] == mus]
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    if grp[i]["dancer"] == grp[j]["dancer"]:
                        continue
                    a = zs(smooth(grp[i]["ke"], sc))
                    b = zs(smooth(grp[j]["ke"], sc))
                    n = min(len(a), len(b))
                    if n < 50 or a.std() < 1e-9 or b.std() < 1e-9:
                        continue
                    vals.append(pearsonr(a[:n], b[:n])[0])
        vals = np.array(vals)
        m3[sc] = (np.nanmean(vals), np.nanstd(vals), len(vals))
        print(f"  scale={sc:>4.1f}s  r={np.nanmean(vals):>7.3f} +- {np.nanstd(vals):.3f}  "
              f"(n_pairs={len(vals)})  => R^2_ceiling={np.nanmean(vals)**2:.4f}")

    # ---------------- VERDICT ----------------
    print("\n" + "=" * 68)
    print("VERDICT: audio->pose R^2  vs  inter-dancer ceiling R^2")
    print("=" * 68)
    print(f"{'scale_s':>8} {'R2_audio':>10} {'R2_ceiling':>12} {'frac_of_ceiling':>16}")
    for sc in SCALES:
        ceil = m3[sc][0] ** 2
        frac = m2[sc] / ceil if ceil > 1e-6 else float("nan")
        print(f"{sc:>8.1f} {m2[sc]:>10.4f} {ceil:>12.4f} {frac:>16.2%}")

    np.save("/root/lightsem/probe1_results.npy",
            dict(m1=m1, m2=m2, m3=m3, n_seq=len(rows)), allow_pickle=True)
    print("\nsaved -> /root/lightsem/probe1_results.npy")


if __name__ == "__main__":
    main()
