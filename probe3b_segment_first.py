#!/usr/bin/env python
"""
PROBE 3b: SEGMENT FIRST, THEN ALIGN  (fixed version).

Bug found in probe3 v1: AIST++ pose clips have median duration 8.7s while the
audio tracks are ~30-45s. Computing beats/sections over the WHOLE track and then
requiring >=20 valid beats inside a clip discarded almost every pair (n=0 for
same-choreo). Fix: for every sequence, clip the audio analysis to that sequence's
own duration, and segment WITHIN that span.

Method per multimodal-ai lecture 06 (segment before aligning; abstract the
modality before fusing):
  T1  beat-synchronous KE  (natural musical element instead of a fixed window)
  T2  structural boundary agreement (pose sections vs audio sections)
  T3  section-level KE (audio sections as the shared, abstracted unit)

Every test reports SAME-choreo vs DIFF-choreo, because only that contrast
separates real shared structure from generic "human moving to music".
"""
import glob
import os
import pickle
import re
from collections import defaultdict
from itertools import combinations

import librosa
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA = "/root/lightsem/data"
KP_FPS = 60.0
SR = 22050
HOP = 512
BOUND_TOL_S = 0.5
MIN_BEATS = 8
MIN_DUR_S = 6.0
N_MUSIC = 12
SEC_PER_SECTION = 2.0     # aim for a section every ~2s inside the clip
MAX_PAIRS = 400


def parse_name(fn):
    m = re.match(r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)",
                 os.path.basename(fn))
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


def clean(kp3d):
    if kp3d is None or kp3d.ndim != 3 or len(kp3d) < 10:
        return None
    if not np.isfinite(kp3d).all():
        bad = ~np.isfinite(kp3d).all(axis=(1, 2))
        if bad.mean() > 0.10:
            return None
        idx, good = np.arange(len(kp3d)), ~bad
        if good.sum() < 10:
            return None
        kp3d = kp3d.copy()
        for j in range(kp3d.shape[1]):
            for c in range(kp3d.shape[2]):
                kp3d[bad, j, c] = np.interp(idx[bad], idx[good], kp3d[good, j, c])
    return kp3d


def ke_and_pose(kp3d):
    rel = kp3d - kp3d[:, :1, :]
    v = np.diff(rel, axis=0) * KP_FPS
    ke = (v ** 2).sum(axis=(1, 2))
    if not np.isfinite(ke).all():
        return None, None
    return ke, rel[1:].reshape(len(rel) - 1, -1)


def zs(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-12 else np.zeros_like(x)


def agg_edges(x, edges):
    out = []
    for i in range(len(edges) - 1):
        a, b = max(0, edges[i]), min(len(x), edges[i + 1])
        out.append(x[a:b].mean() if b > a else np.nan)
    return np.asarray(out)


def boundary_f1(ba, bb, tol):
    if len(ba) == 0 or len(bb) == 0:
        return np.nan
    bb_s = np.sort(bb)
    used, tp = set(), 0
    for t in ba:
        i = np.searchsorted(bb_s, t)
        for k in (i - 1, i):
            if 0 <= k < len(bb_s) and abs(bb_s[k] - t) <= tol and k not in used:
                used.add(k); tp += 1; break
    p, r = tp / len(ba), tp / len(bb)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def rstat(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (np.mean(v), np.std(v), len(v)) if len(v) else (np.nan, np.nan, 0)


def main():
    kp_files = sorted(glob.glob(os.path.join(DATA, "kp3d", "**", "*.pkl"), recursive=True))
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(DATA, "wav", "**", "*.wav"), recursive=True)}

    by_music = defaultdict(list)
    for p in kp_files:
        m = parse_name(p)
        if m and m["music"] in wavs:
            by_music[m["music"]].append((p, m))

    scored = []
    for mus, items in by_music.items():
        bych = defaultdict(set)
        for _, m in items:
            bych[m["chore"]].add(m["dancer"])
        scored.append((sum(1 for _, ds in bych.items() if len(ds) >= 2), mus))
    scored.sort(reverse=True)
    chosen = [mus for _, mus in scored[:N_MUSIC]]
    print("music ids:", chosen, flush=True)

    # ---- full-track audio analysis once per track -----------------------------
    track = {}
    for mus in chosen:
        y, _ = librosa.load(wavs[mus], sr=SR, mono=True)
        tempo, bt = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time")
        mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        track[mus] = dict(tempo=float(np.atleast_1d(tempo)[0]), beat_t=bt,
                          F=np.vstack([mfcc, chroma]), rms=rms, dur=len(y) / SR)
    print(f"audio prepared: {len(track)} tracks  "
          f"(dur {min(t['dur'] for t in track.values()):.0f}-"
          f"{max(t['dur'] for t in track.values()):.0f}s)", flush=True)

    # ---- pose sequences, each clipped against its own duration ----------------
    seqs = []
    for mus in chosen:
        seen = set()
        for p, m in sorted(by_music[mus], key=lambda x: x[0]):
            key = (m["dancer"], m["chore"])
            if key in seen:
                continue
            seen.add(key)
            kp = clean(load_kp(p))
            if kp is None:
                continue
            ke, feat = ke_and_pose(kp)
            if ke is None:
                continue
            dur = len(ke) / KP_FPS
            if dur < MIN_DUR_S:
                continue
            T = track[mus]
            bt = T["beat_t"][T["beat_t"] < dur]          # beats INSIDE this clip
            if len(bt) < MIN_BEATS:
                continue
            bf = librosa.time_to_frames(bt, sr=SR, hop_length=HOP)
            bf = bf[(bf >= 0) & (bf < T["F"].shape[1])]
            if len(bf) < MIN_BEATS:
                continue
            Fs = librosa.util.sync(T["F"], bf, aggregate=np.mean)
            rs = librosa.util.sync(T["rms"][np.newaxis, :], bf, aggregate=np.mean)[0]
            edges = np.clip((bt * KP_FPS).astype(int), 0, len(ke))
            ke_beat = agg_edges(ke, edges)

            # audio sections INSIDE the clip span
            k = max(3, min(int(dur / SEC_PER_SECTION), Fs.shape[1] - 1))
            aseg = np.sort(bt[np.clip(librosa.segment.agglomerative(Fs, k),
                                      0, len(bt) - 1)])
            # pose sections, computed independently, same k
            fb = librosa.util.sync(feat.T, np.clip(edges[:-1], 0, len(feat) - 1),
                                   aggregate=np.mean)
            pseg = (np.sort(bt[np.clip(librosa.segment.agglomerative(fb, k),
                                       0, len(bt) - 1)])
                    if fb.shape[1] > k else np.array([]))

            sec_edges = np.unique(np.clip(
                (np.concatenate([[0.0], aseg, [dur]]) * KP_FPS).astype(int), 0, len(ke)))
            seqs.append(dict(music=mus, dancer=m["dancer"], chore=m["chore"], dur=dur,
                             ke_beat=ke_beat, rms_sync=rs, Fs=Fs,
                             aseg=aseg, pseg=pseg,
                             ke_seg=agg_edges(ke, sec_edges) if len(sec_edges) > 2
                             else np.array([])))
    print(f"usable sequences: {len(seqs)}   "
          f"median dur {np.median([s['dur'] for s in seqs]):.1f}s   "
          f"median beats/clip {np.median([len(s['ke_beat']) for s in seqs]):.0f}",
          flush=True)

    same, diff = [], []
    for mus in chosen:
        grp = [s for s in seqs if s["music"] == mus]
        for a, b in combinations(grp, 2):
            if a["dancer"] == b["dancer"]:
                continue
            (same if a["chore"] == b["chore"] else diff).append((a, b))
    print(f"same-choreo pairs: {len(same)}   diff-choreo pairs: {len(diff)}\n", flush=True)

    def pair_r(pairs, field, minlen):
        vals = []
        for a, b in pairs[:MAX_PAIRS]:
            n = min(len(a[field]), len(b[field]))
            if n < minlen:
                continue
            x, y = a[field][:n], b[field][:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < minlen:
                continue
            vals.append(pearsonr(x[ok], y[ok])[0])
        return rstat(vals)

    print("=" * 74)
    print("T1  BEAT-SYNCHRONOUS KE  (natural musical element, not a fixed window)")
    print("=" * 74)
    for lbl, pr in (("SAME choreo", same), ("DIFF choreo", diff)):
        m, s, n = pair_r(pr, "ke_beat", MIN_BEATS)
        print(f"  inter-dancer r  {lbl}: {m:+.3f} +-{s:.3f}  (n={n})")

    v_rms = []
    for s_ in seqs:
        n = min(len(s_["ke_beat"]), len(s_["rms_sync"]))
        x, y = s_["ke_beat"][:n], s_["rms_sync"][:n]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= MIN_BEATS:
            v_rms.append(pearsonr(x[ok], y[ok])[0])
    m, s, n = rstat(v_rms)
    print(f"  r(beat-sync KE, beat-sync RMS)  : {m:+.3f} +-{s:.3f}  (n={n})")

    X, Y, G = [], [], []
    for s_ in seqs:
        n = min(s_["Fs"].shape[1], len(s_["ke_beat"]))
        x, y = s_["Fs"][:, :n].T, s_["ke_beat"][:n]
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.sum() >= MIN_BEATS:
            X.append(x[ok]); Y.append(zs(y[ok])); G += [s_["music"]] * int(ok.sum())
    X, Y, G = np.vstack(X), np.concatenate(Y), np.asarray(G)
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=min(5, len(set(G)))).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(
            sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[te]))
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    print(f"  audio->KE Ridge R^2 (beat-sync, GroupKFold by music): {r2:+.4f}  "
          f"(r={pearsonr(Y, pred)[0]:+.3f}, N={len(Y)})")

    print("\n" + "=" * 74)
    print("T2  STRUCTURAL BOUNDARY agreement (pose sections vs audio sections)")
    print("=" * 74)
    f1s, nul = [], []
    for s_ in seqs:
        if len(s_["pseg"]) < 3 or len(s_["aseg"]) < 3:
            continue
        f1s.append(boundary_f1(s_["pseg"], s_["aseg"], BOUND_TOL_S))
        nul.append(boundary_f1(s_["pseg"],
                               np.sort(np.random.uniform(0, s_["dur"], len(s_["aseg"]))),
                               BOUND_TOL_S))
    m, s, n = rstat(f1s); mn, _, _ = rstat(nul)
    print(f"  pose-boundary vs audio-boundary F1 (+-{BOUND_TOL_S}s): {m:.3f} +-{s:.3f}")
    print(f"  random-null F1 : {mn:.3f}     lift: {m - mn:+.3f}   (n={n})")
    v = [boundary_f1(a["pseg"], b["pseg"], BOUND_TOL_S) for a, b in same[:MAX_PAIRS]
         if len(a["pseg"]) >= 3 and len(b["pseg"]) >= 3]
    m2, s2, n2 = rstat(v)
    print(f"  [ceiling] inter-dancer pose-boundary F1 : {m2:.3f} +-{s2:.3f}  (n={n2})")

    print("\n" + "=" * 74)
    print("T3  SECTION-LEVEL KE (audio sections as shared abstracted unit)")
    print("=" * 74)
    for lbl, pr in (("SAME choreo", same), ("DIFF choreo", diff)):
        m, s, n = pair_r(pr, "ke_seg", 4)
        print(f"  inter-dancer r  {lbl}: {m:+.3f} +-{s:.3f}  (n={n})")

    print("\n" + "=" * 74)
    print("VERDICT: segmentation-first rescues the idea ONLY if same-choreo")
    print("clearly exceeds diff-choreo. Equal values = generic 'human moving to")
    print("music', no choreography-specific information, rule-out survives.")
    print("=" * 74)

    # ---- dump raw per-pair values for the significance probe ------------------
    import json
    dump = {"pairs": [], "boundary_f1": [], "boundary_null": [],
            "boundary_ceiling": [], "boundary_track": []}
    for lbl, pr in (("same", same), ("diff", diff)):
        for a, b in pr[:MAX_PAIRS]:
            n = min(len(a["ke_beat"]), len(b["ke_beat"]))
            if n < MIN_BEATS:
                continue
            x, y = a["ke_beat"][:n], b["ke_beat"][:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_BEATS:
                continue
            r = pearsonr(x[ok], y[ok])[0]
            if np.isfinite(r):
                dump["pairs"].append({"music": a["music"], "kind": lbl, "r": float(r)})
    for s_ in seqs:
        if len(s_["pseg"]) < 3 or len(s_["aseg"]) < 3:
            continue
        dump["boundary_f1"].append(float(boundary_f1(s_["pseg"], s_["aseg"], BOUND_TOL_S)))
        dump["boundary_null"].append(float(boundary_f1(
            s_["pseg"], np.sort(np.random.uniform(0, s_["dur"], len(s_["aseg"]))),
            BOUND_TOL_S)))
        dump["boundary_track"].append(s_["music"])
    dump["boundary_ceiling"] = [float(x) for x in v if np.isfinite(x)]
    dump["r_rms"] = float(np.nanmean([q for q in v_rms if np.isfinite(q)]))
    dump["ridge_r2"] = float(r2)
    with open("/root/lightsem/probe4_input.json", "w") as f:
        json.dump(dump, f)
    print(f"\ndumped {len(dump['pairs'])} pairs -> probe4_input.json")


if __name__ == "__main__":
    np.random.seed(0)
    main()
