"""Motion descriptors from ordinary video, via an off-the-shelf 2D pose model.

Anton's route to scale: mocap corpora are small and licence-bound, but dance
video is everywhere, and a pose model turns video into the same COCO-17 skeleton
AIST++ ships. What survives the loss of depth is most of what the light mapping
uses — speed, laterality, spread, hands versus head, periodicity. What does not
is anything needing the third dimension: travel toward the camera, foot contact,
true jump height, torso twist. Those are dropped rather than guessed.

The 2D skeleton is embedded as (lateral, up, 0) so it flows through exactly the
same `prepare` and `extract` path as mocap; a descriptor computed here and a
descriptor computed from mocap are then the same function of the same skeleton,
which is what makes the validation in `scripts/legacy_stages/run_pose_validation.py` meaningful.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .skeleton import LSHO, LHIP, RSHO, RHIP, Skeleton, prepare

#: channels a 2D image-plane skeleton can carry (validated in Stage 0a G3)
TRANSFERABLE = ("qom", "head_speed", "foot_speed", "hand_speed", "hand_spread",
                "spread", "vert_bounce", "vert_bounce_amp", "laterality",
                "symmetry", "hip_yaw_speed", "jerk", "flow_eff", "time_eff",
                "weight", "com_speed", "footfall_rate", "vert_period_s")
#: channels it cannot, and which must come from 3D lifting or be dropped
NOT_TRANSFERABLE = ("travel_speed", "travel_sin", "travel_cos", "twist",
                    "space_eff", "vert_root", "metric_level")


def _decode_frames(path: str, start_s: float = 0.0,
                   max_frames: int | None = None):
    """Yield BGR frames and the stream's rate, decoding through PyAV.

    Not OpenCV, which is what `ultralytics` reaches for when handed a path.
    Measured on the Just Dance corpus: 147 of 338 files are AV1, and every one
    of them dies in OpenCV here with "Your platform doesn't support hardware
    accelerated AV1 decoding / Get current frame error", which surfaces as
    `no person tracked` and reads exactly like a corpus that has no dancers in
    it. The same files decode fine through libdav1d, so the fix is to own the
    decode rather than to transcode 43 % of a corpus around a reader.
    """
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if start_s > 0 and stream.time_base:
            container.seek(int(start_s / float(stream.time_base)), stream=stream)
        n = 0
        for frame in container.decode(stream):
            if start_s > 0 and frame.time is not None and frame.time < start_s:
                continue
            if max_frames is not None and n >= max_frames:
                return
            yield frame.to_ndarray(format="bgr24"), fps
            n += 1


DEFAULT_POSE_MODEL = str(Path(__file__).resolve().parents[2] / "assets/models/pose/yolo11m-pose.pt")


def keypoints_from_video(path: str, model_name: str = DEFAULT_POSE_MODEL,
                         conf: float = 0.35, device: str = "cuda",
                         max_frames: int | None = None,
                         imgsz: int = 960,
                         start_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Video -> (T, 17, 2) keypoints and (T, 17) confidences for one dancer.

    Person selection is by track continuity, not by "largest box": AIST clips
    have a mirror wall and occasional crew, and picking the biggest detection
    per frame swaps subject mid-clip, which shows up as enormous fake velocity.

    `start_s` skips a lead-in. Gameplay captures open on titles and menus with
    nobody on screen, and measuring from frame zero charges that emptiness to
    the pose model: on one clip it read as coverage 0.31 where the routine
    itself gives 1.00. It is the caller's business where a routine starts, so
    the default stays at zero and the trim is explicit.

    `meta["fps"]` is the decoded stream's rate, carried out because every
    descriptor downstream is a velocity and a wrong rate scales all of them
    silently.
    """
    from ultralytics import YOLO
    model = YOLO(model_name)
    per_id: dict[int, list] = {}
    frames = 0
    fps = 0.0
    for i, (img, fps) in enumerate(_decode_frames(path, start_s, max_frames)):
        frames += 1
        r = model.track(img, persist=True, conf=conf, device=device,
                        imgsz=imgsz, verbose=False, tracker="bytetrack.yaml")[0]
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        xy = r.keypoints.xy.cpu().numpy()
        cf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
              else np.ones(xy.shape[:2]))
        for k, tid in enumerate(ids):
            per_id.setdefault(int(tid), []).append((i, xy[k], cf[k]))
    if not per_id:
        raise RuntimeError(f"no person tracked in {path}")
    tid = max(per_id, key=lambda t: len(per_id[t]))
    seq = per_id[tid]
    K = np.full((frames, 17, 2), np.nan)
    C = np.zeros((frames, 17))
    for i, xy, cf in seq:
        K[i] = xy
        C[i] = cf
    meta = {"n_frames": frames, "track_id": tid,
            "coverage": float(len(seq) / max(frames, 1)),
            "n_tracks": len(per_id), "model": model_name, "fps": float(fps)}
    return K, C, meta


def to_skeleton(K: np.ndarray, C: np.ndarray, fps: float, name: str = "",
                min_conf: float = 0.3, max_gap_s: float = 0.4,
                stats: dict | None = None) -> Skeleton:
    """(T,17,2) image coordinates -> a prepared Skeleton in the (lat, up, 0) frame.

    Every low-confidence gap is filled by linear interpolation, because the
    Butterworth in `prepare` and every descriptor downstream want a finite
    series. `max_gap_s` does not change that -- it is the line above which a
    filled frame is invention rather than a blink, and the fraction of frames
    above it is written into `stats` so a caller can discount the clip. A short
    gap glides through a real trajectory; a three-second one glides through a
    trajectory the dancer never took, at a constant velocity nobody moved at.
    """
    K = np.asarray(K, dtype=np.float64).copy()
    K[C < min_conf] = np.nan
    T = K.shape[0]
    idx = np.arange(T)
    max_gap = int(max_gap_s * fps)
    n_filled = n_long = 0
    for j in range(17):
        for c in range(2):
            v = K[:, j, c]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                v[:] = 0.0
                K[:, j, c] = v
                n_long += T
                n_filled += T
                continue
            interp = np.interp(idx, idx[ok], v[ok])
            runs = np.diff(np.r_[0, (~ok).astype(int), 0])
            for s, e in zip(np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)):
                n_filled += e - s
                if e - s > max_gap:
                    n_long += e - s
            v[~ok] = interp[~ok]
            K[:, j, c] = v
    if stats is not None:
        denom = max(T * 17 * 2, 1)
        stats["filled_frac"] = n_filled / denom
        stats["long_gap_frac"] = n_long / denom
        stats["max_gap_s"] = max_gap_s
    P = np.zeros((T, 17, 3))
    P[:, :, 0] = K[:, :, 0]            # image x -> lateral
    P[:, :, 1] = -K[:, :, 1]           # image y grows downward; up is -y
    return prepare(P, fps, name=name, up_hint=1)
