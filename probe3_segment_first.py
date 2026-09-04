#!/usr/bin/env python
"""
PROBE 3: SEGMENT FIRST, THEN ALIGN.

Self-critique driven by multimodal-ai skill, lecture 06 (Connections & Alignment),
design rule:

  "When modalities are continuous with no natural elements (sensors, physiological
   time series, speech), solve segmentation before alignment."

and pitfall #4:

  "Applying discrete-alignment methods to continuous data. Sensor streams and raw
   speech have no natural elements. Skipping the segmentation/discretization step
   and pretending fixed windows are semantic units is a silent correctness bug."

Probes 1 and 2 smoothed with FIXED WINDOWS (0.1/0.5/2/8 s) and correlated frame-wise.
That is exactly the flagged bug. If the near-zero beat-scale agreement is an artifact
of using non-semantic units, then segmenting first should RESCUE it.

Three tests, all using natural musical elements instead of fixed windows:

  T1. BEAT-SYNCHRONOUS aggregation. Beats are the natural discrete element of music
      (the HuBERT-style "discretize then align" move). Aggregate KE within each
      beat interval, then correlate. Removes tempo/phase misalignment by
      construction. Also refits the audio->KE ridge on beat-sync features with
      GroupKFold by music.

  T2. STRUCTURAL BOUNDARY agreement. Segment the audio into k sections
      (beat-sync MFCC+chroma -> agglomerative) and the pose stream into k sections
      independently. Do the boundaries coincide above chance? This asks whether
      pose and audio share *structure*, not amplitude -- the thing a fixed-window
      Pearson cannot see.

  T3. SEGMENT-LEVEL agreement. Using the AUDIO's own section boundaries as the
      shared unit, take mean KE per section and correlate across dancers, and
      against mean audio energy per section. This is the "abstract the modality
      before fusing" move (lecture 06, key idea 2).

READ:
  If T1/T2/T3 show clearly higher agreement than the fixed-window result, the
  earlier rule-out was a methodology artifact and must be RETRACTED.
  If they do not, the rule-out survives the skill's own prescribed method.
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
N_SECTIONS = 8          # target number of structural sections per clip
BOUND_TOL_S = 1.0       # boundary match tolerance
N_MUSIC = 10


# ---------------------------------------------------------------- data loading
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
    """KE(t) at 60fps, plus root-relative pose feature matrix [T, J*3]."""
    rel = kp3d - kp3d[:, :1, :]
    v = np.diff(rel, axis=0) * KP_FPS
    ke = (v ** 2).sum(axis=(1, 2))
    if not np.isfinite(ke).all():
        return None, None
    feat = rel[1:].reshape(len(rel) - 1, -1)
    return ke, feat


def zs(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-12 else np.zeros_like(x)


def agg_by_edges(x, edges):
    """Mean of x within each [edges[i], edges[i+1]) window (frame indices)."""
    out = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        a, b = max(0, a), min(len(x), b)
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
                used.add(k)
                tp += 1
                break
    p, r = tp / len(ba), tp / len(bb)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


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

    # ---------------- audio side: beats + structural boundaries ----------------
    audio = {}
    for mus in chosen:
        y, _ = librosa.load(wavs[mus], sr=SR, mono=True)
        tempo, beat_t = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time")
        mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        beat_f = librosa.time_to_frames(beat_t, sr=SR, hop_length=HOP)
        beat_f = beat_f[(beat_f >= 0) & (beat_f < mfcc.shape[1])]
        if len(beat_f) < N_SECTIONS + 2:
            continue
        Msync = librosa.util.sync(np.vstack([mfcc, chroma]), beat_f, aggregate=np.mean)
        rms_sync = librosa.util.sync(rms[np.newaxis, :], beat_f, aggregate=np.mean)[0]
        # structural segmentation on beat-sync timbre+harmony
        k = min(N_SECTIONS, Msync.shape[1] - 1)
        seg_beats = librosa.segment.agglomerative(Msync, k)
        seg_t = librosa.frames_to_time(beat_f[np.clip(seg_beats, 0, len(beat_f) - 1)],
                                       sr=SR, hop_length=HOP)
        audio[mus] = dict(tempo=float(np.atleast_1d(tempo)[0]),
                          beat_t=librosa.frames_to_time(beat_f, sr=SR, hop_length=HOP),
                          rms_sync=rms_sync, feat_sync=Msync, seg_t=np.sort(seg_t))
    print(f"audio prepared for {len(audio)} tracks", flush=True)

    # ---------------- pose side ----------------
    seqs = []
    for mus in chosen:
        if mus not in audio:
            continue
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
            seqs.append(dict(music=mus, dancer=m["dancer"], chore=m["chore"],
                             ke=ke, feat=feat))
    print(f"loaded sequences: {len(seqs)}", flush=True)

    # ---- beat-sync KE per sequence -------------------------------------------
    for s in seqs:
        bt = audio[s["music"]]["beat_t"]
        edges = np.clip((bt * KP_FPS).astype(int), 0, len(s["ke"]))
        s["ke_beat"] = agg_by_edges(s["ke"], edges)
        # pose structural boundaries, computed INDEPENDENTLY of the audio
        fb = librosa.util.sync(s["feat"].T, np.clip(edges[:-1], 0, len(s["feat"]) - 1),
                               aggregate=np.mean)
        if fb.shape[1] > N_SECTIONS + 1:
            kk = min(N_SECTIONS, fb.shape[1] - 1)
            sb = librosa.segment.agglomerative(fb, kk)
            s["pose_seg_t"] = np.sort(bt[np.clip(sb, 0, len(bt) - 1)])
        else:
            s["pose_seg_t"] = np.array([])

    same_pairs, diff_pairs = [], []
    for mus in audio:
        grp = [s for s in seqs if s["music"] == mus]
        for a, b in combinations(grp, 2):
            if a["dancer"] == b["dancer"]:
                continue
            (same_pairs if a["chore"] == b["chore"] else diff_pairs).append((a, b))
    print(f"same-choreo pairs: {len(same_pairs)}  diff-choreo pairs: {len(diff_pairs)}\n",
          flush=True)

    # ================= T1: beat-synchronous agreement =========================
    print("=" * 72)
    print("T1  BEAT-SYNCHRONOUS aggregation (natural musical element, not a window)")
    print("=" * 72)
    for label, pairs in (("SAME choreo", same_pairs[:250]), ("DIFF choreo", diff_pairs[:250])):
        vals = []
        for a, b in pairs:
            n = min(len(a["ke_beat"]), len(b["ke_beat"]))
            x, y = a["ke_beat"][:n], b["ke_beat"][:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 20:
                continue
            r = pearsonr(x[ok], y[ok])[0]
            if np.isfinite(r):
                vals.append(r)
        print(f"  inter-dancer r (beat-sync KE)  {label}: "
              f"{np.mean(vals):+.3f} +-{np.std(vals):.3f}  (n={len(vals)})")

    vals = []
    for s in seqs:
        rs = audio[s["music"]]["rms_sync"]
        n = min(len(s["ke_beat"]), len(rs))
        x, y = s["ke_beat"][:n], rs[:n]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        r = pearsonr(x[ok], y[ok])[0]
        if np.isfinite(r):
            vals.append(r)
    print(f"  r(beat-sync KE, beat-sync RMS)           : "
          f"{np.mean(vals):+.3f} +-{np.std(vals):.3f}  (n={len(vals)})")

    # ridge: beat-sync audio features -> beat-sync KE, GroupKFold by music
    X, Y, G = [], [], []
    for s in seqs:
        F = audio[s["music"]]["feat_sync"]
        n = min(F.shape[1], len(s["ke_beat"]))
        x, y = F[:, :n].T, s["ke_beat"][:n]
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.sum() < 20:
            continue
        X.append(x[ok]); Y.append(zs(y[ok])); G += [s["music"]] * int(ok.sum())
    X, Y, G = np.vstack(X), np.concatenate(Y), np.asarray(G)
    preds = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=min(5, len(set(G)))).split(X, Y, G):
        sc = StandardScaler().fit(X[tr])
        mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), Y[tr])
        preds[te] = mdl.predict(sc.transform(X[te]))
    ss_res = ((Y - preds) ** 2).sum()
    ss_tot = ((Y - Y.mean()) ** 2).sum()
    print(f"  audio->KE Ridge R^2 (beat-sync, GroupKFold by music): "
          f"{1 - ss_res / ss_tot:+.4f}   (r={pearsonr(Y, preds)[0]:+.3f}, N={len(Y)})")

    # ================= T2: structural boundary agreement ======================
    print()
    print("=" * 72)
    print("T2  STRUCTURAL BOUNDARY agreement (pose sections vs audio sections)")
    print("=" * 72)
    f1s, nulls = [], []
    for s in seqs:
        pb, ab = s["pose_seg_t"], audio[s["music"]]["seg_t"]
        if len(pb) < 3 or len(ab) < 3:
            continue
        f1s.append(boundary_f1(pb, ab, BOUND_TOL_S))
        span = max(pb.max(), ab.max())
        nulls.append(boundary_f1(pb, np.sort(np.random.uniform(0, span, len(ab))),
                                 BOUND_TOL_S))
    print(f"  pose-boundary vs audio-boundary F1 (+-{BOUND_TOL_S}s): "
          f"{np.nanmean(f1s):.3f} +-{np.nanstd(f1s):.3f}")
    print(f"  random-null F1                            : {np.nanmean(nulls):.3f}")
    print(f"  lift                                      : "
          f"{np.nanmean(f1s) - np.nanmean(nulls):+.3f}   (n={len(f1s)})")

    # inter-dancer boundary agreement = the ceiling for T2
    vals = []
    for a, b in same_pairs[:250]:
        if len(a["pose_seg_t"]) >= 3 and len(b["pose_seg_t"]) >= 3:
            vals.append(boundary_f1(a["pose_seg_t"], b["pose_seg_t"], BOUND_TOL_S))
    print(f"  [ceiling] inter-dancer pose-boundary F1   : "
          f"{np.nanmean(vals):.3f} +-{np.nanstd(vals):.3f}  (n={len(vals)})")

    # ================= T3: audio-section-level agreement ======================
    print()
    print("=" * 72)
    print("T3  SEGMENT-LEVEL (audio sections as the shared unit; abstracted modality)")
    print("=" * 72)
    for s in seqs:
        st = audio[s["music"]]["seg_t"]
        edges = np.clip((np.concatenate([[0.0], st, [len(s["ke"]) / KP_FPS]]) * KP_FPS)
                        .astype(int), 0, len(s["ke"]))
        edges = np.unique(edges)
        s["ke_seg"] = agg_by_edges(s["ke"], edges) if len(edges) > 2 else np.array([])

    for label, pairs in (("SAME choreo", same_pairs[:250]), ("DIFF choreo", diff_pairs[:250])):
        vals = []
        for a, b in pairs:
            n = min(len(a["ke_seg"]), len(b["ke_seg"]))
            if n < 4:
                continue
            x, y = a["ke_seg"][:n], b["ke_seg"][:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4:
                continue
            r = pearsonr(x[ok], y[ok])[0]
            if np.isfinite(r):
                vals.append(r)
        print(f"  inter-dancer r (section-mean KE) {label}: "
              f"{np.mean(vals):+.3f} +-{np.std(vals):.3f}  (n={len(vals)})")

    print()
    print("=" * 72)
    print("VERDICT RULE: segmentation-first RESCUES the idea only if beat-sync /")
    print("section-level inter-dancer agreement is clearly ABOVE the fixed-window")
    print("numbers AND same-choreo clearly exceeds diff-choreo. Otherwise the")
    print("earlier rule-out survives the skill's own prescribed methodology.")
    print("=" * 72)


if __name__ == "__main__":
    np.random.seed(0)
    main()
