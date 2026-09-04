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
which is what makes the validation in `run_pose_validation.py` meaningful.
"""

from __future__ import annotations

import numpy as np

from .skeleton import LSHO, LHIP, RSHO, RHIP, Skeleton, prepare

#: channels a 2D image-plane skeleton can carry (validated in Stage 0a G3)
TRANSFERABLE = ("qom", "head_speed", "foot_speed", "hand_speed", "hand_spread",
                "spread", "vert_bounce", "vert_bounce_amp", "laterality",
                "symmetry", "hip_yaw_speed", "jerk", "flow_eff", "time_eff",
                "weight", "com_speed", "footfall_rate", "vert_period_s")
#: channels it cannot, and which must come from 3D lifting or be dropped
NOT_TRANSFERABLE = ("travel_speed", "travel_sin", "travel_cos", "twist",
                    "space_eff", "vert_root", "metric_level")


def keypoints_from_video(path: str, model_name: str = "yolo11m-pose.pt",
                         conf: float = 0.35, device: str = "cuda",
                         max_frames: int | None = None,
                         imgsz: int = 960) -> tuple[np.ndarray, np.ndarray, dict]:
    """Video -> (T, 17, 2) keypoints and (T, 17) confidences for one dancer.

    Person selection is by track continuity, not by "largest box": AIST clips
    have a mirror wall and occasional crew, and picking the biggest detection
    per frame swaps subject mid-clip, which shows up as enormous fake velocity.
    """
    from ultralytics import YOLO
    model = YOLO(model_name)
    res = model.track(path, stream=True, conf=conf, device=device, imgsz=imgsz,
                      verbose=False, persist=True, tracker="bytetrack.yaml")
    per_id: dict[int, list] = {}
    frames = 0
    for i, r in enumerate(res):
        if max_frames and i >= max_frames:
            break
        frames += 1
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
            "n_tracks": len(per_id), "model": model_name}
    return K, C, meta


def to_skeleton(K: np.ndarray, C: np.ndarray, fps: float, name: str = "",
                min_conf: float = 0.3, max_gap_s: float = 0.4) -> Skeleton:
    """(T,17,2) image coordinates -> a prepared Skeleton in the (lat, up, 0) frame."""
    K = np.asarray(K, dtype=np.float64).copy()
    K[C < min_conf] = np.nan
    T = K.shape[0]
    idx = np.arange(T)
    max_gap = int(max_gap_s * fps)
    for j in range(17):
        for c in range(2):
            v = K[:, j, c]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                v[:] = 0.0
                K[:, j, c] = v
                continue
            interp = np.interp(idx, idx[ok], v[ok])
            runs = np.diff(np.r_[0, (~ok).astype(int), 0])
            for s, e in zip(np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)):
                if e - s <= max_gap:
                    v[s:e] = interp[s:e]
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = interp[bad]         # long gaps: fill, and let coverage report it
            K[:, j, c] = v
    P = np.zeros((T, 17, 3))
    P[:, :, 0] = K[:, :, 0]            # image x -> lateral
    P[:, :, 1] = -K[:, :, 1]           # image y grows downward; up is -y
    return prepare(P, fps, name=name, up_hint=1)
