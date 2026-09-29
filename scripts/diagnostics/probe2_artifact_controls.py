#!/usr/bin/env python
"""
PROBE 2: Is the near-zero beat-scale ceiling REAL, or an artifact?

Probe 1 found: two dancers on the SAME music have r(KE_A, KE_B) ~ 0.02-0.05 at
0.1-0.5s, but 0.50 at 8s. Before concluding "pose is arbitrary at beat scale",
kill three artifacts that would fake that result:

  A1. FIXED LAG. Clips may be offset by a constant sync delay. Zero-lag Pearson
      would then read ~0 even for identical motion.
      -> recompute as MAX cross-correlation over +-1.0s of lag.

  A2. POSE-ESTIMATOR JITTER. If beat-scale KE is dominated by triangulation
      noise, low agreement says nothing about choreography.
      -> compare SAME-CHOREOGRAPHY pairs (same music, same ch id, different
         dancer) against DIFFERENT-CHOREOGRAPHY pairs (same music, other ch).
         Same-choreo is the reproducibility ceiling / noise floor.
         If same-choreo is HIGH and diff-choreo is LOW -> signal is real,
         and music->motion is genuinely one-to-many.
         If BOTH are low -> beat-scale KE is mostly noise.

  A3. AMPLITUDE vs TIMING. Pearson on a continuous signal is strict. Dancers may
      hit the SAME beats with DIFFERENT energy. The sketch's strobe trigger only
      needs event TIMING.
      -> extract motion-hit events (jerk peaks) and measure
         (a) F1 of event agreement between dancers at +-70ms
         (b) alignment of motion hits to librosa beat grid
         vs a shuffled-event null.
"""
import glob
import os
import pickle
import re
import sys
from collections import defaultdict
from itertools import combinations

import librosa
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

DATA = "/root/lightsem/data"
KP_FPS = 60.0
HOP = 512
SR = 22050
FPS = SR / HOP

SCALES = [0.1, 0.5, 2.0, 8.0]
MAX_LAG_S = 1.0
EVENT_TOL_S = 0.070


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


def ke_and_jerk(kp3d):
    """Return KE(t) at 60fps and jerk magnitude (for hit events)."""
    rel = kp3d - kp3d[:, :1, :]
    v = np.diff(rel, axis=0) * KP_FPS
    ke = (v ** 2).sum(axis=(1, 2))
    a = np.diff(v, axis=0) * KP_FPS
    jerk = np.linalg.norm(np.diff(a, axis=0) * KP_FPS, axis=2).sum(axis=1)
    if not (np.isfinite(ke).all() and np.isfinite(jerk).all()):
        return None, None
    return ke, jerk


def smooth_fps(x, sec, fps):
    w = max(1, int(round(sec * fps)))
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="same")


def zs(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def max_xcorr(a, b, max_lag):
    """Max Pearson over integer lags in [-max_lag, +max_lag]."""
    n = min(len(a), len(b))
    a, b = zs(a[:n]), zs(b[:n])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, 0
    best, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[-lag:], b[:n + lag]
        elif lag > 0:
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 50:
            continue
        r = pearsonr(x, y)[0]
        if np.isfinite(r) and r > best:
            best, best_lag = r, lag
    return (best, best_lag) if best > -2 else (np.nan, 0)


def hit_events(jerk, fps, min_sep_s=0.15):
    """Motion 'hits'/locks = prominent jerk peaks -> event times in seconds."""
    j = smooth_fps(jerk, 0.03, fps)
    if j.std() < 1e-12:
        return np.array([])
    thr = j.mean() + 1.0 * j.std()
    pk, _ = find_peaks(j, height=thr, distance=max(1, int(min_sep_s * fps)))
    return pk / fps


def event_f1(ta, tb, tol):
    if len(ta) == 0 or len(tb) == 0:
        return np.nan
    tb_sorted = np.sort(tb)
    matched_b, tp = set(), 0
    for t in ta:
        i = np.searchsorted(tb_sorted, t)
        cands = [k for k in (i - 1, i) if 0 <= k < len(tb_sorted)]
        hit = None
        for k in cands:
            if abs(tb_sorted[k] - t) <= tol and k not in matched_b:
                hit = k
                break
        if hit is not None:
            matched_b.add(hit)
            tp += 1
    prec = tp / len(ta)
    rec = tp / len(tb)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    kp_files = sorted(glob.glob(os.path.join(DATA, "kp3d", "**", "*.pkl"), recursive=True))
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(DATA, "wav", "**", "*.wav"), recursive=True)}

    recs = []
    for p in kp_files:
        m = parse_name(p)
        if m and m["music"] in wavs:
            recs.append((p, m))

    # group by (music, chore) so we can separate same-choreo from diff-choreo
    by_music = defaultdict(list)
    for p, m in recs:
        by_music[m["music"]].append((p, m))

    # pick music ids that actually contain same-choreo/different-dancer pairs
    scored = []
    for mus, items in by_music.items():
        bych = defaultdict(set)
        for _, m in items:
            bych[m["chore"]].add(m["dancer"])
        n_same = sum(1 for ch, ds in bych.items() if len(ds) >= 2)
        scored.append((n_same, len(items), mus))
    scored.sort(reverse=True)
    chosen = [mus for n_same, _, mus in scored[:10]]
    print("music ids:", chosen)
    print("same-choreo-multi-dancer counts:",
          [(m, n) for n, _, m in scored[:10]], flush=True)

    # load
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
            ke, jerk = ke_and_jerk(kp)
            if ke is None:
                continue
            seqs.append(dict(music=mus, dancer=m["dancer"], chore=m["chore"],
                             genre=m["genre"], ke=ke, jerk=jerk))
    print(f"loaded sequences: {len(seqs)}", flush=True)

    same_pairs, diff_pairs = [], []
    for mus in chosen:
        grp = [s for s in seqs if s["music"] == mus]
        for a, b in combinations(grp, 2):
            if a["dancer"] == b["dancer"]:
                continue
            (same_pairs if a["chore"] == b["chore"] else diff_pairs).append((a, b))
    print(f"same-choreo pairs: {len(same_pairs)}   diff-choreo pairs: {len(diff_pairs)}",
          flush=True)
    if not same_pairs:
        print("NO SAME-CHOREO PAIRS -> cannot run A2"); sys.exit(1)

    # ---------- A1 + A2: lag-corrected agreement, split by choreography ----------
    print("\n=== A1+A2: MAX cross-corr over +-1s lag, KE_A vs KE_B ===")
    print(f"{'scale_s':>8} | {'SAME choreo':>22} | {'DIFF choreo':>22}")
    print(f"{'':>8} | {'r_max (n)':>22} | {'r_max (n)':>22}")
    results = {}
    max_lag = int(MAX_LAG_S * KP_FPS)
    for sc in SCALES:
        out = {}
        for label, pairs in (("same", same_pairs), ("diff", diff_pairs)):
            sub = pairs[:250]
            vals = []
            for a, b in sub:
                ra, _ = max_xcorr(smooth_fps(a["ke"], sc, KP_FPS),
                                  smooth_fps(b["ke"], sc, KP_FPS), max_lag)
                if np.isfinite(ra):
                    vals.append(ra)
            out[label] = (np.mean(vals) if vals else np.nan,
                          np.std(vals) if vals else np.nan, len(vals))
        results[sc] = out
        s, d = out["same"], out["diff"]
        print(f"{sc:>8.1f} | {s[0]:>7.3f} +-{s[1]:<6.3f} (n={s[2]:>4}) | "
              f"{d[0]:>7.3f} +-{d[1]:<6.3f} (n={d[2]:>4})")

    # ---------- A3: event-timing agreement ----------
    print("\n=== A3: motion-hit EVENT agreement (jerk peaks, +-70ms) ===")
    for label, pairs in (("SAME choreo", same_pairs), ("DIFF choreo", diff_pairs)):
        sub = pairs[:250]
        f1s, f1_null = [], []
        for a, b in sub:
            ta = hit_events(a["jerk"], KP_FPS)
            tb = hit_events(b["jerk"], KP_FPS)
            if len(ta) < 5 or len(tb) < 5:
                continue
            f1s.append(event_f1(ta, tb, EVENT_TOL_S))
            # null: same rate, uniformly random times over same span
            span = max(ta.max(), tb.max())
            tn = np.sort(np.random.uniform(0, span, len(tb)))
            f1_null.append(event_f1(ta, tn, EVENT_TOL_S))
        f1s, f1_null = np.array(f1s), np.array(f1_null)
        print(f"  {label:<12} F1={np.nanmean(f1s):.3f} +-{np.nanstd(f1s):.3f}   "
              f"random-null F1={np.nanmean(f1_null):.3f}   "
              f"lift={np.nanmean(f1s) - np.nanmean(f1_null):+.3f}   (n={len(f1s)})")

    # ---------- A3b: are motion hits locked to the MUSICAL beat grid? ----------
    print("\n=== A3b: motion hits vs librosa beat grid (+-70ms) ===")
    beat_cache = {}
    for mus in chosen:
        y, _ = librosa.load(wavs[mus], sr=SR, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time")
        beat_cache[mus] = (float(np.atleast_1d(tempo)[0]), beats)
    f1b, f1b_null = [], []
    for s in seqs:
        tempo, beats = beat_cache[s["music"]]
        ta = hit_events(s["jerk"], KP_FPS)
        if len(ta) < 5 or len(beats) < 5:
            continue
        f1b.append(event_f1(ta, beats, EVENT_TOL_S))
        span = max(ta.max(), beats.max())
        f1b_null.append(event_f1(ta, np.sort(np.random.uniform(0, span, len(beats))),
                                 EVENT_TOL_S))
    print(f"  motion-hit vs beat-grid F1={np.nanmean(f1b):.3f} +-{np.nanstd(f1b):.3f}   "
          f"random-null={np.nanmean(f1b_null):.3f}   "
          f"lift={np.nanmean(f1b) - np.nanmean(f1b_null):+.3f}   (n={len(f1b)})")
    print(f"  mean tempo: {np.mean([beat_cache[m][0] for m in chosen]):.1f} BPM")

    print("\n" + "=" * 70)
    print("READ: if SAME-choreo r_max is HIGH at 0.1-0.5s while DIFF-choreo is")
    print("LOW, beat-scale pose is reproducible signal and music->motion is")
    print("genuinely one-to-many (ground truth arbitrary). If BOTH are low,")
    print("beat-scale KE is estimator noise and the formula is undefined there.")
    print("=" * 70)


if __name__ == "__main__":
    np.random.seed(0)
    main()
