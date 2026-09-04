#!/usr/bin/env python
"""Pose extraction over the Just Dance captures, and the trial that decides it.

Why a separate driver and not `pose2d.keypoints_from_video`: that path runs
ByteTrack frame by frame and keeps the longest track. On this corpus the
tracker fragments (Funkytown: 23 tracks for one coach) and its coverage is not
run-to-run stable (Havana: 1.000 over 750 frames, 0.627 over 600). The coach
here is the large figure in the middle of the frame; the pictogram strip's
little figures are a tenth of its height. So selection is per frame by box
size with a continuity preference, no tracker, and inference is batched.

`trial`  -- a few clips, several models, overlays and numbers; nothing cached.
`run`    -- every usable file, N worker processes each owning the GPU in
            turn (decode is CPU, inference is GPU, so workers overlap), one
            npz per file under CACHE/justdance_pose/<yt_id>.npz.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")

JD = Path("/mnt/lightsem-data/justdance")
CACHE = Path("/mnt/lightsem-data/cache")
POSE_DIR = CACHE / "justdance_pose"
MANIFEST = CACHE / "justdance_manifest.json"
SCRATCH = Path("/tmp/claude-0/-root-lightsem/71c8c079-19a2-4ec4-bea5-2ce7db778b3e/scratchpad")

#: A coach is at least this fraction of the frame height. Pictogram figures
#: sit around 0.10-0.15; a quartet's coaches around 0.35; a solo 0.45-0.75.
MIN_H_FRAC = 0.22
#: Regions where a "person" is UI, not a coach: (x0, y0, x1, y1) fractions.
UI_BOXES = ((0.70, 0.72, 1.00, 1.00),   # pictogram strip, bottom right
            (0.00, 0.00, 1.00, 0.14))   # score / feedback band, top

TRIAL = [  # (short name, file)
    ("funkytown", "Just Dance 2018 (Unlimited)： Funkytown [aSyRGVHMDRw].mkv"),
    ("ringmybell_jd1", "Anita Ward - Ring My Bell (Just Dance 1) [7UfXJFpmW-g].webm"),
    ("automaton", "Just Dance 2018： Automaton [rWo5v_99zn0].webm"),
    ("handclap_60fps", "HandClap ｜ Just Dance 2018 [esLEZZLO7aw].webm"),
    ("havana_1080p", "Just Dance 2019： Havana [zARI9hxqxRM].mkv"),
    ("wildwildwest_quartet", "Just Dance Hits： Wild Wild West by Will Smith [11.9k] [bQca5WpcNjI].mkv"),
]


# ------------------------------------------------------------------ decode
def decode(path: Path, start_s: float = 0.0, max_out: int | None = None,
           target_fps: float = 30.0):
    """Yield (t_seconds, BGR frame) at a stride that keeps <= ~target_fps.

    60 fps captures carry real motion (measured 2026-09-04), but the
    descriptor chain filters at 12 Hz and every other file is 25-30 fps, so
    every second frame of a 60 fps file is all the chain can use.
    """
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate) if s.average_rate else 30.0
        stride = max(1, int(round(fps / target_fps)))
        if start_s > 0 and s.time_base:
            c.seek(int(start_s / float(s.time_base)), stream=s)
        n_in = n_out = 0
        for fr in c.decode(s):
            if start_s > 0 and fr.time is not None and fr.time < start_s:
                continue
            if n_in % stride == 0:
                if max_out is not None and n_out >= max_out:
                    return
                t = fr.time if fr.time is not None else n_in / fps
                yield float(t), fr.to_ndarray(format="bgr24"), fps, stride
                n_out += 1
            n_in += 1


# ------------------------------------------------------------------ selection
def _in_ui(box, W, H):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    return any(a <= cx <= c and b <= cy <= d for a, b, c, d in UI_BOXES)


def count_pictogram_hues(frames, W, H):
    """Diagnostic only. The pictogram strip draws one figure per coach, each in
    its own hue; count distinct hues among saturated bright pixels. YOLO does
    not see the stylised figures (0 of 3 on a trial strip), and the decision
    rests on the tracks; this number is stored so the two can be compared."""
    import cv2
    x0, y0 = int(0.70 * W), int(0.72 * H)
    counts = []
    for f in frames:
        hsv = cv2.cvtColor(f[y0:, x0:], cv2.COLOR_BGR2HSV)
        m = (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
        if m.sum() < 200:
            counts.append(0); continue
        hist = np.bincount(hsv[..., 0][m] // 10, minlength=18).astype(float)
        hist = hist / hist.sum()
        counts.append(int((hist > 0.12).sum()))
    return int(np.median(counts)) if counts else 0, counts


# ------------------------------------------------------------------ models
class YoloPose:
    def __init__(self, name: str, imgsz: int = 960, half: bool = True, device=0,
                 conf: float = 0.10):
        from ultralytics import YOLO
        self.m = YOLO(name)
        self.name, self.imgsz, self.half, self.device, self.conf = name, imgsz, half, device, conf

    def __call__(self, frames):
        res = self.m.predict(frames, imgsz=self.imgsz, half=self.half, conf=self.conf,
                             device=self.device, verbose=False, classes=[0])
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                out.append((None, None, None)); continue
            b = r.boxes.xyxy.cpu().numpy()
            k = r.keypoints.xy.cpu().numpy()
            c = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                 else np.ones(k.shape[:2]))
            out.append((b, k, c))
        return out


class RtmBody:
    """rtmlib RTMPose (top-down, YOLOX detector), CPU through onnxruntime."""
    def __init__(self, mode: str = "balanced"):
        from rtmlib import Body
        self.b = Body(mode=mode, backend="onnxruntime", device="cpu", to_openpose=False)
        self.name = f"rtmpose-{mode}"

    def __call__(self, frames):
        out = []
        for f in frames:
            k, c = self.b(f)
            if k is None or len(k) == 0:
                out.append((None, None, None)); continue
            k = np.asarray(k); c = np.asarray(c)
            vis = c > 0.3
            b = np.stack([np.where(vis, k[..., 0], np.inf).min(1), np.where(vis, k[..., 1], np.inf).min(1),
                          np.where(vis, k[..., 0], -np.inf).max(1), np.where(vis, k[..., 1], -np.inf).max(1)], 1)
            b[~np.isfinite(b)] = 0
            out.append((b, k, c))
        return out


# ------------------------------------------------------------------ tracks
class Tracks:
    """Follow every coach-sized figure at once; decide afterwards which are coaches.

    Anton's point (2026-09-04): several Just Dance routines have two, three or
    four coaches, and each is a dancer hearing the same music -- the same
    structure AIST++ has, several dancers per track -- so each coach is its
    own channel, not a nuisance. Greedy nearest-box matching with a hold
    through short misses; background extras become tracks too and are
    dropped at the end by rank (coverage x height x keypoint confidence),
    with the game's own pictogram count as the number to keep.
    """

    def __init__(self, hold: int = 15):
        self.tr: list[dict] = []
        self.hold = hold

    @staticmethod
    def _c(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    @staticmethod
    def _ms(kxy, kcf, box):
        """Anchor point of a detection: the mid-shoulder keypoint when both
        shoulders are confident, else the box centre. A full-body box and an
        upper-body box of the same coach have centres 0.4-0.8 heights apart
        (measured, Funkytown and Automaton) but the same shoulders."""
        if kcf[5] > 0.3 and kcf[6] > 0.3:
            return (kxy[5] + kxy[6]) / 2
        return Tracks._c(box)

    @staticmethod
    def dedupe(boxes, kxy, kcf, ok, iou_thr=0.5):
        """One detection per figure. A low confidence floor makes the model emit
        a second, weaker box on the same coach; unhandled, every such frame
        spawns a new track and the coach fragments into dozens of them."""
        order = sorted(ok, key=lambda j: -float(kcf[j].mean()))
        keep = []
        for j in order:
            bj = boxes[j]
            dup = False
            for k in keep:
                bk = boxes[k]
                ix = max(0, min(bj[2], bk[2]) - max(bj[0], bk[0]))
                iy = max(0, min(bj[3], bk[3]) - max(bj[1], bk[1]))
                inter = ix * iy
                aj = (bj[2] - bj[0]) * (bj[3] - bj[1]); ak = (bk[2] - bk[0]) * (bk[3] - bk[1])
                union = aj + ak - inter
                small = min(aj, ak)
                hk = bk[3] - bk[1]
                cdist = float(np.linalg.norm(Tracks._c(bj) - Tracks._c(bk)))
                mdist = float(np.linalg.norm(Tracks._ms(kxy[j], kcf[j], bj) - Tracks._ms(kxy[k], kcf[k], bk)))
                if union > 0 and (inter / union > iou_thr or (small > 0 and inter / small > 0.8)
                                  or cdist < 0.25 * hk or mdist < 0.15 * hk):
                    dup = True; break
            if not dup:
                keep.append(j)
        return keep

    def step(self, i, boxes, kxy, kcf, ok):
        ok = self.dedupe(boxes, kxy, kcf, ok)
        act = [t for t in self.tr if t["miss"] <= self.hold]
        pairs = []
        for ti, t in enumerate(act):
            pa, ph = t["anchor"], t["box"][3] - t["box"][1]
            for j in ok:
                d = float(np.linalg.norm(self._ms(kxy[j], kcf[j], boxes[j]) - pa))
                if d < 0.5 * ph * (1 + 0.15 * t["miss"]):
                    pairs.append((d, ti, j))
        pairs.sort()
        used_t, used_j = set(), set()
        for d, ti, j in pairs:
            if ti in used_t or j in used_j:
                continue
            used_t.add(ti); used_j.add(j)
            t = act[ti]
            t["box"] = boxes[j]; t["miss"] = 0; t["anchor"] = self._ms(kxy[j], kcf[j], boxes[j])
            t["frames"][i] = (kxy[j], kcf[j], boxes[j])
        for ti, t in enumerate(act):
            if ti not in used_t:
                t["miss"] += 1
        for j in ok:
            if j not in used_j:
                self.tr.append({"box": boxes[j], "miss": 0, "anchor": self._ms(kxy[j], kcf[j], boxes[j]),
                                "frames": {i: (kxy[j], kcf[j], boxes[j])}})

    @staticmethod
    def _near_in_time(a: dict, b: dict, max_gap: int = 300) -> float | None:
        """Median distance between b's boxes and a's nearest-in-time boxes, in
        a-heights, or None when the two share a frame (then they are two
        figures, not one)."""
        fa = np.array(sorted(a["frames"])); fb = np.array(sorted(b["frames"]))
        shared = np.intersect1d(fa, fb).size
        if shared > 0.2 * min(len(fa), len(fb)) + 2:
            return None
        fb = np.setdiff1d(fb, fa)
        if fb.size == 0:
            return 0.0
        j = np.clip(np.searchsorted(fa, fb), 0, len(fa) - 1)
        j2 = np.clip(j - 1, 0, len(fa) - 1)
        near = np.where(np.abs(fa[j] - fb) <= np.abs(fa[j2] - fb), fa[j], fa[j2])
        gap = np.abs(near - fb)
        if gap.min() > max_gap:
            return None

        def dist(f_b, f_a):
            kb, cb, bb = b["frames"][int(f_b)]; ka, ca, ab = a["frames"][int(f_a)]
            h = max(ab[3] - ab[1], bb[3] - bb[1], 1)
            return np.linalg.norm(Tracks._ms(kb, cb, bb) - Tracks._ms(ka, ca, ab)) / h

        close = gap <= 30
        k = int(np.argmin(gap))
        d_close = (float(np.median([dist(f_b, f_a) for f_b, f_a in zip(fb[close], near[close])]))
                   if close.sum() >= 3 else None)
        d_junc = float(dist(fb[k], near[k]))
        if os.environ.get("JD_DEBUG") and len(a["frames"]) > 100 and len(b["frames"]) > 100:
            print(f"      pair a={len(a['frames'])} b={len(b['frames'])} n_close={int(close.sum())} "
                  f"d_close={d_close} d_junc={d_junc:.3f} gap_min={int(gap.min())} "
                  f"a_range={min(a['frames'])}-{max(a['frames'])} b_range={min(b['frames'])}-{max(b['frames'])}", flush=True)
        # Interleaved fragments must agree where they overlap in time; sequential
        # ones only at the junction, and a figure can travel or rescale during a
        # long miss or a camera cut (Automaton cuts between camera positions).
        if d_close is not None:
            return min(d_close, d_junc / 2.5)
        return d_junc / 2.5

    def merge(self, H: int):
        """Join fragments of one figure. Two tracks that never share a frame and
        whose boxes sit within 0.6 body heights of each other at the nearest
        times are one figure: missed for a while (occlusion, Gold Move flash),
        or split by a second, partial box that alternated with the full one."""
        tr = sorted(self.tr, key=lambda t: -len(t["frames"]))
        merged = True
        while merged:
            merged = False
            for a in tr:
                for b in tr:
                    if b is a or len(b["frames"]) > len(a["frames"]):
                        continue
                    d = self._near_in_time(a, b)
                    shared = len(set(a["frames"]) & set(b["frames"]))
                    if os.environ.get("JD_DEBUG") and len(a["frames"]) > 100 and len(b["frames"]) > 100:
                        print(f"      merge? a={len(a['frames'])} b={len(b['frames'])} shared={shared} d={d}", flush=True)
                    if d is not None and d < (0.35 if shared else 0.6):
                        for f, v in b["frames"].items():
                            a["frames"].setdefault(f, v)
                        tr = [t for t in tr if t is not b]      # by identity: dicts hold arrays
                        merged = True
                        break
                if merged:
                    break
        self.tr = tr
        self.hold_used = self.hold

    def finish(self, n_frames: int, H: int, k_picto: int):
        self.merge(H)
        for t in self.tr:
            fr = t["frames"].values()
            t["cov"] = len(t["frames"]) / max(n_frames, 1)
            t["h"] = float(np.median([b[3] - b[1] for _, _, b in fr])) / H
            t["conf"] = float(np.mean([c.mean() for _, c, _ in fr]))
            t["x"] = float(np.median([(b[0] + b[2]) / 2 for _, _, b in fr]))
            t["score"] = t["cov"] * t["h"] * t["conf"]
        ranked = sorted(self.tr, key=lambda t: -t["score"])
        if k_picto >= 1:
            k = k_picto
        elif ranked:                            # the strip was unreadable: coaches are the
            h0, c0 = ranked[0]["h"], ranked[0]["conf"]   # big, sharp, persistent figures
            k = max(1, min(4, sum(1 for t in ranked
                                  if t["cov"] >= 0.6 and t["h"] >= 0.8 * h0 and t["conf"] >= 0.92 * c0)))
        else:
            k = 1
        top = sorted(ranked[:k], key=lambda t: t["x"])     # left to right
        self.debug = [{"cov": round(t["cov"], 3), "h": round(t["h"], 3), "conf": round(t["conf"], 3),
                       "x": round(t["x"] / 1000, 3), "n": len(t["frames"])} for t in ranked[:8]]
        return top, k


# ------------------------------------------------------------------ one file
def extract(model, path: Path, start_s=0.0, max_out=None, batch=16,
            keep_frames: list[int] | None = None):
    """Run `model` over a file. Returns dict with K (T,P,17,2), C (T,P,17),
    box (T,P,4) for P coaches ordered left to right, n_large (T,), times,
    meta, and optionally the raw frames at `keep_frames`."""
    tracks = Tracks()
    NL, T = [], []
    kept = {}
    picto_frames = []
    W = H = None
    fps = stride = None
    t0 = time.time()
    buf, tbuf, ibuf = [], [], []
    n = 0

    def flush():
        for (b, k, c), t, i in zip(model(buf), tbuf, ibuf):
            T.append(t)
            if b is None:
                NL.append(0); continue
            h = b[:, 3] - b[:, 1]
            ok = [j for j in range(len(b)) if h[j] >= MIN_H_FRAC * H and not _in_ui(b[j], W, H)]
            NL.append(len(ok))
            tracks.step(i, b, k, c, ok)
        buf.clear(); tbuf.clear(); ibuf.clear()

    for t, img, fps, stride in decode(path, start_s, max_out):
        if W is None:
            H, W = img.shape[:2]
        if keep_frames and n in keep_frames:
            kept[n] = img.copy()
        if n % 150 == 0 and len(picto_frames) < 40:
            picto_frames.append(img.copy())
        buf.append(img); tbuf.append(t); ibuf.append(n); n += 1
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    dt = time.time() - t0
    n_picto, picto_counts = count_pictogram_hues(picto_frames, W, H) if picto_frames else (0, [])
    top, k = tracks.finish(n, H, 0)            # tracks decide; the hue count is a diagnostic
    P = max(len(top), 1)
    K = np.full((n, P, 17, 2), np.nan); C = np.zeros((n, P, 17)); B = np.full((n, P, 4), np.nan)
    for pi, t in enumerate(top):
        for i, (kxy, kcf, box) in t["frames"].items():
            K[i, pi] = kxy; C[i, pi] = kcf; B[i, pi] = box
    NL = np.array(NL); T = np.array(T)
    cov_each = [float(np.isfinite(K[:, pi, 0, 0]).mean()) for pi in range(P)] if top else [0.0]
    cov = float(min(cov_each))
    meta = {"file": path.name, "n_frames": int(n), "fps_src": fps, "stride": stride,
            "fps_eff": (fps / stride) if fps else None, "width": W, "height": H,
            "n_coaches": int(len(top)), "n_coaches_picto": int(n_picto), "k_used": int(k),
            "picto_counts": picto_counts, "n_tracks_total": len(tracks.tr),
            "tracks_top": tracks.debug,
            "coverage": cov, "coverage_each": cov_each,
            "mean_conf": float(np.mean([t["conf"] for t in top])) if top else 0.0,
            "track_height_each": [round(t["h"], 3) for t in top],
            "track_x_each": [round(t["x"] / max(W, 1), 3) for t in top],
            "n_large_median": int(np.median(NL)) if len(NL) else 0,
            "model": model.name, "imgsz": getattr(model, "imgsz", None),
            "conf_floor": getattr(model, "conf", None),
            "seconds": dt, "fps_proc": n / dt if dt else 0.0, "start_s": start_s}
    return {"K": K[:, 0], "C": C[:, 0], "box": B[:, 0], "K_all": K, "C_all": C, "box_all": B,
            "n_large": NL, "t": T, "meta": meta, "frames": kept}


# ------------------------------------------------------------------ trial metrics
def hf_fraction(x, fps, cut_hz=6.0):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 32:
        return float("nan")
    idx = np.arange(len(x)); x = np.interp(idx, idx[ok], x[ok]); x = x - x.mean()
    P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / fps)
    return float(P[f >= cut_hz].sum() / max(P[f > 0].sum(), 1e-12))


def torso(K):
    sh = (K[:, 5] + K[:, 6]) / 2; hp = (K[:, 11] + K[:, 12]) / 2
    return np.nanmedian(np.linalg.norm(sh - hp, axis=1))


def agreement(a, b):
    """Mean keypoint distance between two models, in torso lengths, where both found a coach."""
    both = np.isfinite(a["K"][:, 0, 0]) & np.isfinite(b["K"][:, 0, 0])
    if both.sum() < 10:
        return float("nan"), int(both.sum())
    d = np.linalg.norm(a["K"][both] - b["K"][both], axis=-1).mean()
    return float(d / torso(a["K"])), int(both.sum())


def draw_overlay(img, K_all_i, C_all_i, box_all_i, label):
    """All coaches of one frame, each in its own colour; wrists and ankles in red."""
    import cv2
    im = img.copy()
    edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
    cols = [(0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 200, 255)]
    for pi in range(K_all_i.shape[0]):
        k, c, box = K_all_i[pi], C_all_i[pi], box_all_i[pi]
        col = cols[pi % len(cols)]
        if np.isfinite(box).all():
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
        for a, b in edges:
            if c[a] > 0.3 and c[b] > 0.3:
                cv2.line(im, tuple(k[a].astype(int)), tuple(k[b].astype(int)), col, 3)
        for j in range(17):
            if c[j] > 0.3:
                cv2.circle(im, tuple(k[j].astype(int)), 5, (0, 0, 255) if j in (9, 10, 15, 16) else (255, 0, 0), -1)
    cv2.putText(im, label, (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return im


def beat_check(res, wav: Path, name: str):
    """Descriptor-level sanity: does the coach's vertical bounce period land on
    the beat grid of the same 30 s of audio?"""
    from lightsem.audio.online import extract_bus
    from lightsem.motion.pose2d import to_skeleton
    from lightsem.motion.descriptors import extract as dextract
    from lightsem.eval.bas import circular_phase
    fps = res["meta"]["fps_eff"]
    stats = {}
    sk = to_skeleton(res["K"], res["C"], fps, name=name, stats=stats)
    bus = extract_bus(str(wav), causal=False)
    beats = bus.beats[bus.beats < sk.duration_s]
    d = dextract(sk, beats=beats, tempo=bus.tempo)
    vp = d.channels["vert_period_s"]; vp = vp[np.isfinite(vp) & (vp > 0)]
    beat_s = 60.0 / bus.tempo
    ph = circular_phase(d.events["footfall"], beats)
    return {"tempo_bpm": float(bus.tempo), "beat_s": beat_s,
            "vert_period_median_s": float(np.median(vp)) if vp.size else None,
            "vert_period_over_beat": float(np.median(vp) / beat_s) if vp.size else None,
            "footfall_R": float(ph["R"]) if ph["R"] == ph["R"] else None,
            "n_footfalls": int(len(d.events["footfall"])),
            "qom_median": float(np.nanmedian(d.channels["qom"])),
            "filled_frac": stats.get("filled_frac"), "long_gap_frac": stats.get("long_gap_frac")}


def trial(args):
    import cv2
    out = {"clips": {}, "models": {}}
    start, n_out = args.start_s, args.frames
    keep = list(range(0, n_out, max(1, n_out // 6)))[:6]
    arms = [("yolo11m@960", lambda: YoloPose("yolo11m-pose.pt", 960, conf=args.conf)),
            ("yolo26m@960", lambda: YoloPose("yolo26m-pose.pt", 960, conf=args.conf)),
            ("yolo26m@640", lambda: YoloPose("yolo26m-pose.pt", 640, conf=args.conf)),
            ("yolo26x@960", lambda: YoloPose("yolo26x-pose.pt", 960, conf=args.conf))]
    if args.arms != "all":
        want = set(args.arms.split(","))
        arms = [a for a in arms if a[0] in want]
    overlay_arm = arms[0][0] if "yolo26m@960" not in {a[0] for a in arms} else "yolo26m@960"
    clips = TRIAL if args.clips == "all" else [c for c in TRIAL if c[0] in set(args.clips.split(","))]
    results: dict[str, dict[str, dict]] = {}
    for arm, mk in arms:
        model = mk()
        for short, fn in clips:
            r = extract(model, JD / fn, start_s=start, max_out=n_out, batch=args.batch,
                        keep_frames=keep if arm == overlay_arm else None)
            m = r["meta"]
            nose_y = r["K"][:, 0, 1] / torso(r["K"]); ank_y = r["K"][:, 15:17, 1].mean(1) / torso(r["K"])
            m["hf_nose"] = hf_fraction(nose_y, m["fps_eff"]); m["hf_ankle"] = hf_fraction(ank_y, m["fps_eff"])
            results.setdefault(arm, {})[short] = r
            print(f"{arm:14s} {short:22s} cov {m['coverage']:.3f} each {[round(x, 2) for x in m['coverage_each']]} "
                  f"conf {m['mean_conf']:.2f} coaches {m['n_coaches']} (picto {m['n_coaches_picto']}, "
                  f"large {m['n_large_median']}, tracks {m['n_tracks_total']}) hf_nose {m['hf_nose']:.3f} "
                  f"hf_ank {m['hf_ankle']:.3f} {m['fps_proc']:.0f} fps", flush=True)
            print("      tracks:", m["tracks_top"][:5], flush=True)
            if r["frames"]:
                strip = [draw_overlay(r["frames"][i], r["K_all"][i], r["C_all"][i], r["box_all"][i], f"{short} f{i}")
                         for i in sorted(r["frames"])]
                h = min(s.shape[0] for s in strip)
                strip = [cv2.resize(s, (int(s.shape[1] * h / s.shape[0]), h)) for s in strip]
                cv2.imwrite(str(SCRATCH / f"overlay_{short}.jpg"), np.concatenate(strip, 1), [cv2.IMWRITE_JPEG_QUALITY, 80])
        del model
    # RTMPose reference on CPU, two clips, fewer frames
    rtm = RtmBody("balanced") if args.rtm_frames > 0 else None
    for short, fn in (TRIAL[:1] + TRIAL[4:5]) if rtm else []:
        r = extract(rtm, JD / fn, start_s=start, max_out=args.rtm_frames, batch=8)
        results.setdefault("rtmpose-m@cpu", {})[short] = r
        m = r["meta"]
        print(f"{'rtmpose-m@cpu':14s} {short:22s} cov {m['coverage']:.3f} conf {m['mean_conf']:.2f} {m['fps_proc']:.1f} fps", flush=True)
    # agreement matrix against yolo26x (the most accurate arm)
    ref_arm = "yolo26x@960" if "yolo26x@960" in results else arms[-1][0]
    ref = results[ref_arm]
    for arm in results:
        for short, r in results[arm].items():
            if short in ref and arm != ref_arm:
                rr = ref[short]
                n = min(len(r["K"]), len(rr["K"]))
                a, nb = agreement({"K": r["K"][:n]}, {"K": rr["K"][:n]})
                r["meta"]["agree_vs_26x_torso"] = a; r["meta"]["agree_frames"] = nb
    # beat check on the production candidate
    for short, fn in TRIAL:
        wav = SCRATCH / f"trial_{short}.wav"
        if wav.exists() and short in results.get(overlay_arm, {}):
            try:
                bc = beat_check(results[overlay_arm][short], wav, short)
            except Exception as e:  # noqa: BLE001
                bc = {"error": f"{type(e).__name__}: {e}"}
            results[overlay_arm][short]["meta"]["beat"] = bc
            print(f"beat {short:22s} {bc}", flush=True)
    out["models"] = {arm: {s: r["meta"] for s, r in d.items()} for arm, d in results.items()}
    (SCRATCH / f"{args.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", SCRATCH / f"{args.tag}.json")


# ------------------------------------------------------------------ full run
def _worker(job):
    """One process: own model, own file list. Writes one npz per file."""
    model_name, imgsz, files, batch, wid, conf = job
    import torch
    torch.cuda.set_device(0)
    model = YoloPose(model_name, imgsz, conf=conf)
    done = []
    for yt, fn in files:
        out = POSE_DIR / f"{yt}.npz"
        if out.exists():
            continue
        try:
            r = extract(model, JD / fn, batch=batch)
            np.savez_compressed(out, K=r["K_all"].astype(np.float32), C=r["C_all"].astype(np.float32),
                                box=r["box_all"].astype(np.float32), n_large=r["n_large"].astype(np.int16),
                                t=r["t"].astype(np.float64), meta=json.dumps(r["meta"]))
            done.append((yt, r["meta"]["coverage"], r["meta"]["fps_proc"], r["meta"]["n_coaches"]))
            print(f"[w{wid}] {yt} cov {r['meta']['coverage']:.3f} coaches {r['meta']['n_coaches']} "
                  f"{r['meta']['n_frames']} fr {r['meta']['fps_proc']:.0f} fps", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[w{wid}] FAIL {yt} {fn[:50]}: {type(e).__name__}: {e}", flush=True)
    return done


def run(args):
    import multiprocessing as mp
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(MANIFEST.read_text())
    files = [(r["yt_id"], r["file"]) for r in rows
             if r.get("yt_id") and "error" not in r and (r.get("mean_db") or -99) > -40]
    files = [f for f in files if not (POSE_DIR / f"{f[0]}.npz").exists()]
    if args.limit:
        files = files[:args.limit]
    # longest files first so no worker is left holding one giant file at the end;
    # --reverse takes them shortest first, so a second run started later works
    # the other end of the same list and the two meet in the middle.
    dur = {r["yt_id"]: r["duration_s"] for r in rows if r.get("yt_id")}
    files.sort(key=lambda f: -dur.get(f[0], 0), reverse=args.reverse)
    W = args.workers
    jobs = [(args.model, args.imgsz, files[i::W], args.batch, i, args.conf) for i in range(W)]
    print(f"{len(files)} files, {W} workers, model {args.model}@{args.imgsz}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(W) as pool:
        res = pool.map(_worker, jobs)
    done = [d for r in res for d in r]
    print(f"done {len(done)} files in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trial"); t.add_argument("--start-s", type=float, default=60.0)
    t.add_argument("--arms", default="all"); t.add_argument("--clips", default="all")
    t.add_argument("--frames", type=int, default=900); t.add_argument("--rtm-frames", type=int, default=240)
    t.add_argument("--batch", type=int, default=16); t.add_argument("--conf", type=float, default=0.10)
    t.add_argument("--tag", default="trial_pose")
    r = sub.add_parser("run"); r.add_argument("--model", default="yolo26m-pose.pt")
    r.add_argument("--conf", type=float, default=0.10)
    r.add_argument("--imgsz", type=int, default=960); r.add_argument("--workers", type=int, default=3)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--reverse", action="store_true", help="shortest files first (a second run from the other end)")
    a = ap.parse_args()
    trial(a) if a.cmd == "trial" else run(a)
