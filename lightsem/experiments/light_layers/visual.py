"""Streaming visual extraction: TRAIN by default; historical VAL by explicit gate.

Historical development requires --allow-development AND a frozen manifest with
exposure="historical_development". TEST is always forbidden. This permission is
not a quality qualification or authorization to fit on unannotated targets.

The two masks observe the SAME decoded RGB frames. SAM2 is used as a stateless
box-prompted image model, not a video tracker. Its Ultralytics implementation is
recorded; exact equivalence to Meta's official inference is NOT assumed.
No palette fitting, segmentation accuracy estimate, or production changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from . import targets
from ..stage11_data import UI, resize_mask

FPS = 10
DISK_FLOOR_BYTES = 1024**3
MODES = ("box", "sam2")


@dataclass(frozen=True)
class VisualConfig:
    decode_width: int = 640
    mask_dilation_px: int = 0
    max_pose_age_s: float = .2
    max_frame_error_s: float = .1
    min_retained_pixels: int = 100
    audit: bool = False
    device: str = "0"
    allow_development: bool = False
    invalidate_missing_frames: bool = False

    def __post_init__(self):
        for name in ("allow_development", "invalidate_missing_frames"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("decode_width", "mask_dilation_px", "min_retained_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "mask_dilation_px" else 1):
                raise ValueError(f"invalid {name}")
        for name in ("max_pose_age_s", "max_frame_error_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _publish(temp: Path, path: Path):
    # Atomic, no-clobber publication on the same filesystem.
    os.link(temp, path)
    temp.unlink()


def write_json(path: Path, value):
    temp = path.with_name(path.name + ".partial")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _publish(temp, path)


def write_arrays(path: Path, arrays: dict):
    """Compact, non-pickle NPZ, atomically published without overwriting."""
    temp = path.with_name(path.name + ".partial")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    _publish(temp, path)


def require_disk_floor(path: Path, reserve_bytes: int = 0):
    existing = Path(path).resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < DISK_FLOOR_BYTES + reserve_bytes:
        raise RuntimeError(f"disk floor: {free} free bytes; require >=1 GiB plus {reserve_bytes} reserve")
    return free


def decode_geometry(original_wh, max_width=640):
    ow, oh = original_wh
    if min(ow, oh, max_width) <= 0:
        raise ValueError("image dimensions must be positive")
    w = min(max_width, ow)
    h = max(2, round(oh * w / ow / 2) * 2)
    return w, h


def transform_boxes(boxes, original_wh, decode_wh):
    """Original-pixel XYXY -> decoded-pixel XYXY, without imgsz assumptions."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("boxes must be finite (N,4)")
    ow, oh = original_wh
    w, h = decode_wh
    if min(ow, oh, w, h) <= 0 or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("invalid box/dimensions")
    out = boxes * np.array([w / ow, h / oh, w / ow, h / oh])
    out[:, (0, 2)] = np.clip(out[:, (0, 2)], 0, w)
    out[:, (1, 3)] = np.clip(out[:, (1, 3)], 0, h)
    if np.any(out[:, 2:] <= out[:, :2]):
        raise ValueError("box lies outside image")
    return out.astype(np.float32)


def ui_mask(decode_wh):
    """True means excluded. Regions and truncation match latest Stage 11."""
    w, h = decode_wh
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in UI:
        mask[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = True
    return mask


def box_foreground(boxes, decode_wh, dilation=.0):
    w, h = decode_wh
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        dx, dy = (x1-x0)*dilation, (y1-y0)*dilation
        mask[max(0, math.floor(y0-dy)):min(h, math.ceil(y1+dy)+1),
             max(0, math.floor(x0-dx)):min(w, math.ceil(x1+dx)+1)] = True
    return mask


def baseline_dilation(boxes, indices, original_wh, decode_wh, min_pixels=100):
    # Exactly Stage11's recording-level .15 -> 0 fallback, not per-frame tuning.
    for dilation in (.15, 0.):
        minimum = decode_wh[0] * decode_wh[1]
        for index in np.unique(indices):
            visible = int(np.count_nonzero(resize_mask(boxes[index], original_wh, decode_wh, dilation=dilation)))
            minimum = min(minimum, visible)
        if minimum >= min_pixels:
            return dilation, minimum
    raise ValueError("box baseline leaves fewer than fixed minimum background pixels")


def retained_mask(foreground, ui, dilation_px=0):
    foreground = np.asarray(foreground)
    if foreground.dtype != bool or foreground.shape != ui.shape:
        raise ValueError("foreground must be boolean and match UI mask")
    if dilation_px:
        from scipy.ndimage import maximum_filter
        foreground = maximum_filter(foreground, size=2*dilation_px+1, mode="constant", cval=0)
    return ~(foreground | ui)


def sample_grid(start_s, duration_s):
    if (not math.isfinite(start_s) or start_s < 0
            or not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("TRAIN window requires nonnegative start and positive duration")
    count = int(math.ceil(duration_s * FPS - 1e-9))
    return float(start_s) + np.arange(count, dtype=np.float64) / FPS


def nearest_indices(source_times, requested_times, max_error_s):
    """Nearest source PTS, tie goes to earlier frame. No invented precision."""
    source = np.asarray(source_times, dtype=np.float64)
    requested = np.asarray(requested_times, dtype=np.float64)
    if (source.ndim != 1 or not len(source) or not np.isfinite(source).all()
            or np.any(np.diff(source) <= 0) or requested.ndim != 1
            or not np.isfinite(requested).all() or np.any(np.diff(requested) <= 0)):
        raise ValueError("timestamps must be finite and strictly increasing")
    right = np.minimum(np.searchsorted(source, requested), len(source)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(requested-source[left] <= source[right]-requested + 1e-12, left, right)
    if np.any(np.abs(source[indices]-requested) > max_error_s + 1e-9):
        raise ValueError("source/pose timestamp exceeds fixed age limit")
    return indices


def previous_pose_indices(pose_times, source_times, max_age_s=.2):
    pose_times = np.asarray(pose_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    nearest_indices(pose_times, source_times, max_age_s)  # validate arrays
    indices = np.searchsorted(pose_times, source_times + 1e-9, side="right") - 1
    if np.any(indices < 0) or np.any(source_times - pose_times[indices] > max_age_s + 1e-9):
        raise ValueError("previous pose timestamp exceeds fixed age limit")
    return indices


def pose_frame_support(boxes, pose_times, source_times, max_age_s=.2, registered_indices=None):
    """No-imputation support: previous timestamp and ALL registered coach slots.

    Missing/stale timestamps and any nonfinite/nonpositive coach box invalidate
    the observation. An invalid index before pose support is -1, never wrapped.
    The caller must not turn this diagnostic into an empty foreground mask.
    """
    pose_times = np.asarray(pose_times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    nearest_indices(pose_times, source_times, float("inf"))
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[0] != len(pose_times) or boxes.shape[2] != 4 or not boxes.shape[1]:
        raise ValueError("pose boxes must have nonempty (T,coaches,4) shape")
    indices = np.searchsorted(pose_times, source_times + 1e-9, side="right") - 1
    exists = indices >= 0
    safe_indices = np.maximum(indices, 0)
    timestamp_valid = exists & (source_times - pose_times[safe_indices] <= max_age_s + 1e-9)
    selected = boxes[safe_indices]
    coach_valid = np.isfinite(selected).all(axis=2) & (selected[..., 2:] > selected[..., :2]).all(axis=2)
    coach_valid[~exists] = False
    registered = np.arange(boxes.shape[1]) if registered_indices is None else np.asarray(registered_indices)
    if (registered.ndim != 1 or not len(registered) or registered.dtype.kind not in "iu"
            or np.any(registered < 0) or np.any(registered >= boxes.shape[1])
            or len(np.unique(registered)) != len(registered)):
        raise ValueError("registered coach indices must be unique existing pose slots")
    frame_valid = timestamp_valid & coach_valid[:, registered].all(axis=1)
    return indices, coach_valid, timestamp_valid, frame_valid


def boxes_at_time(boxes, pose_times, source_time, max_age_s=.2):
    index = int(previous_pose_indices(pose_times, [source_time], max_age_s)[0])
    all_boxes = np.asarray(boxes[index], dtype=np.float64).reshape(-1, 4)
    valid = np.isfinite(all_boxes).all(axis=1) & (all_boxes[:, 2:] > all_boxes[:, :2]).all(axis=1)
    if not valid.any():
        raise ValueError(f"all coaches missing/nonfinite at source time {source_time:.6f}; invalid record")
    return all_boxes[valid], index, valid


def load_pose(row):
    with np.load(row["pose_path"], allow_pickle=False) as z:
        boxes = np.asarray(z["box"], dtype=np.float64)
        times = np.asarray(z["t"], dtype=np.float64)
        meta = json.loads(str(z["meta"]))
    if boxes.ndim == 2:
        boxes = boxes[:, None, :]
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or len(boxes) != len(times):
        raise ValueError("pose box shape must be (T,coaches,4)")
    nearest_indices(times, times[:1], .2)
    if (meta["width"], meta["height"]) != (row["original_width"], row["original_height"]):
        raise ValueError("manifest/pose original geometry differs")
    return boxes, times, meta


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def probe_frame_mapping(path, requested_times, original_wh, max_error_s=.1):
    """Scan numeric frame PTS only; never retain source images.

    Decode selection uses original frame indices without seeking, so ffmpeg and
    ffprobe have the same frame index origin, including VFR/B-frame sources.
    Record-relative seconds subtract the stream start_time; raw PTS are saved.
    """
    cp = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_streams", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(cp.stdout)["streams"][0]
    if (stream["width"], stream["height"]) != tuple(original_wh):
        raise ValueError("video/manifest original geometry differs")
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    rotations = [float(x.get("rotation", 0)) for x in stream.get("side_data_list", [])]
    if rotation % 360 or any(x % 360 for x in rotations):
        raise ValueError("rotated video requires a frozen explicit geometry transform")
    time_base = Fraction(stream["time_base"])
    origin = float(stream.get("start_time", 0.))
    times, pts = [], []
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                    "-show_frames", "-show_entries", "frame=best_effort_timestamp",
                                    "-of", "default=noprint_wrappers=0", str(path)],
                                   stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            current = None
            in_frame = False
            for line in process.stdout:
                line = line.strip()
                if line == "[FRAME]":
                    in_frame, current = True, None
                elif line.startswith("best_effort_timestamp=") and in_frame:
                    current = int(line.split("=", 1)[1])
                elif line == "[/FRAME]":
                    if current is None:
                        raise ValueError("frame lacks source PTS")
                    value = float(current * time_base) - origin
                    if times and value <= times[-1]:
                        raise ValueError("source PTS not strictly increasing")
                    pts.append(current)
                    times.append(value)
                    in_frame = False
                    if value >= requested_times[-1] + max_error_s:
                        break
            else:
                if process.wait(timeout=10):
                    errors.seek(0)
                    raise ValueError(errors.read().decode(errors="replace")[-1000:])
        finally:
            _stop(process)
            process.stdout.close()
    indices = nearest_indices(times, requested_times, max_error_s)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("source cannot support distinct observations at 10 Hz")
    return {"frame_indices": indices.astype(np.int64), "source_times_s": np.asarray(times)[indices],
            "source_pts": np.asarray(pts, dtype=np.int64)[indices], "requested_times_s": requested_times,
            "time_base": str(time_base), "stream_start_time_s": origin}


def balanced_select_expression(frame_indices):
    """Exact index selection with logarithmic FFmpeg expression-parser depth.

    A flat 640-term sum exceeds libavutil's recursive expression parser limit
    (reported as Cannot allocate memory). Parenthesizing a balanced tree keeps
    identical eq(n,index) leaves without approximating the source-frame mapping.
    """
    indices = np.asarray(frame_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(np.diff(indices.astype(np.int64)) <= 0)):
        raise ValueError("frame indices must be nonempty, nonnegative, strictly increasing integers")

    def expression(first, end):
        if end - first == 1:
            return f"eq(n\\,{int(indices[first])})"
        middle = (first + end) // 2
        return "(" + expression(first, middle) + "+" + expression(middle, end) + ")"

    return expression(0, len(indices))


def iter_rgb_frames(path, mapping, decode_wh):
    """One sampled RGB image at a time. No image cache or full-video loader."""
    w, h = decode_wh
    expression = balanced_select_expression(mapping["frame_indices"])
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select={expression},scale={w}:{h}",
               "-frames:v", str(len(mapping["frame_indices"])), "-fps_mode", "passthrough",
               "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for _ in mapping["frame_indices"]:
                raw = bytearray(w*h*3)
                view = memoryview(raw)
                offset = 0
                while offset < len(raw):
                    n = process.stdout.readinto(view[offset:])
                    if not n:
                        process.wait(timeout=30)
                        errors.seek(0)
                        detail = errors.read().decode(errors="replace")[-3000:]
                        raise ValueError(f"truncated sampled RGB stream: {detail}")
                    offset += n
                yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                del view, raw
            if process.wait(timeout=30):
                errors.seek(0)
                raise ValueError(errors.read().decode(errors="replace")[-1000:])
        finally:
            _stop(process)
            process.stdout.close()


class SAM2FrameSegmenter:
    """Lazy CUDA dependency; never imported for CPU tools/tests."""
    def __init__(self, checkpoint, config):
        import torch
        import ultralytics
        from ultralytics import SAM
        from ultralytics.models.sam import predict, build, model
        if not torch.cuda.is_available():
            raise RuntimeError("SAM2 stage requires CUDA")
        # SAM loads on CPU. Initialize the CUDA allocator before per-record
        # peak counters (Torch 2.14 otherwise raises Invalid device argument).
        torch.cuda.init()
        self.torch = torch
        self.device = config.device
        self.cuda_device = int(config.device) if config.device.isdigit() else config.device
        self.model = SAM(str(checkpoint))
        sources = {str(Path(module.__file__).resolve()): sha256(module.__file__)
                   for module in (predict, build, model)}
        self.provenance = {
            "wrapper": "ultralytics.SAM / SAM2Predictor", "wrapper_version": ultralytics.__version__,
            "wrapper_source_sha256": sources, "torch_version": torch.__version__,
            "official_inference_equivalence": "not assumed or tested",
            "inference": "independent box-prompted images; no temporal tracking or full-video loader",
            "input_order": "decoded RGB; BGR view passed to Ultralytics numpy interface",
            "imgsz": 1024, "conf": 0.0, "multimask_output": False,
            "licenses": "existing Ultralytics dependency AGPL-3.0; official SAM2.1 weights Apache-2.0; distinct licenses",
        }

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats(self.cuda_device)

    def peak_bytes(self):
        return int(self.torch.cuda.max_memory_allocated(self.cuda_device))

    def __call__(self, rgb, boxes):
        result = None
        try:
            with self.torch.inference_mode():
                result = self.model.predict(source=rgb[..., ::-1], bboxes=boxes.tolist(),
                                            device=self.device, imgsz=1024, conf=0.0,
                                            verbose=False, save=False, stream=False)[0]
            if result.masks is None or len(result.masks.data) != len(boxes):
                raise ValueError("SAM2 omitted a prompted coach; invalid record, not empty foreground")
            masks = result.masks.data
            if tuple(masks.shape[1:]) != tuple(rgb.shape[:2]):
                raise ValueError("SAM2 mask is not in decoded image coordinates")
            if not bool(masks.flatten(1).any(1).all()):
                raise ValueError("SAM2 returned an empty prompted coach mask")
            foreground = masks.bool().any(dim=0).cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("SAM2 predicted nonfinite quality scores")
            return foreground, scores
        finally:
            predictor = self.model.predictor
            if predictor is not None:
                predictor.reset_image()
                # Release retained Results and current source before next frame.
                predictor.results = None
                predictor.batch = None
                predictor.dataset = None
            del result


def audit_indices(n):
    return sorted({0, n//2, n-1}) if n else []


def save_audit(path, rgb, box_mask, sam_mask):
    """RGB plus excluded-pixel overlays. Perception aid, NOT mask truth."""
    from PIL import Image
    h, w = rgb.shape[:2]
    panel = np.empty((h, 3*w, 3), dtype=np.uint8)
    panel[:, :w] = rgb
    for j, mask in enumerate((box_mask, sam_mask), start=1):
        panel[:, j*w:(j+1)*w] = rgb
        region = panel[:, j*w:(j+1)*w]
        region[~mask] = (.5*region[~mask] + .5*np.array([255, 0, 255])).astype(np.uint8)
    temp = path.with_name(path.name + ".partial")
    with temp.open("xb") as stream:
        Image.fromarray(panel).save(stream, format="PNG")
    _publish(temp, path)


def save_invalid_audit(path, rgb):
    """Save original RGB only; never invent an exclusion overlay for a gap."""
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0]+26), (0, 0, 0))
    image.paste(Image.fromarray(rgb), (0, 26))
    draw = ImageDraw.Draw(image)
    draw.text((3, 3), "INVALID: missing pose support; no masks computed", fill=(255, 180, 255))
    temp = path.with_name(path.name + ".partial")
    with temp.open("xb") as stream:
        image.save(stream, format="PNG")
    _publish(temp, path)


def _frame_arrays(n):
    shapes = {"hs_histogram": (n, targets.HS_DIM), "rich96": (n, targets.RICH_DIM),
              "value_summary": (n, 4), "retained_fraction": (n,), "valid_fraction": (n,),
              "chromatic_fraction": (n,), "mask_activity": (n,)}
    # Float32 while calculating novelty; disk representations are float16.
    arrays = {key: np.zeros(shape, np.float32) for key, shape in shapes.items()}
    arrays.update(color_valid=np.zeros(n, bool), quadrant_color_valid=np.zeros((n, 4), bool),
                  frame_valid=np.zeros(n, bool), mask_activity_known=np.zeros(n, bool),
                  value_valid=np.zeros(n, bool), retained_pixels=np.zeros(n, np.int32),
                  valid_pixels=np.zeros(n, np.int32))
    return arrays


def extract_record(row, directory, segmenter, config):
    started = time.monotonic()
    segmenter.reset_peak()
    original_wh = (row["original_width"], row["original_height"])
    decode_wh = decode_geometry(original_wh, config.decode_width)
    requested = sample_grid(row["start_s"], row["duration_s"])
    mapping = probe_frame_mapping(row["source_path"], requested, original_wh, config.max_frame_error_s)
    boxes, pose_times, pose_meta = load_pose(row)
    n = len(requested)
    registered_indices = (np.asarray([c["coach_index"] for c in row["coaches"]], dtype=np.int64)
                          if config.invalidate_missing_frames else np.arange(boxes.shape[1]))
    if config.invalidate_missing_frames:
        pi, pose_valid, pose_timestamp_valid, frame_valid = pose_frame_support(
            boxes, pose_times, mapping["source_times_s"], config.max_pose_age_s, registered_indices)
        if not frame_valid.any():
            raise ValueError("no supported visual frames; zero placeholders cannot qualify a record")
    else:
        # Preserve strict v3 policy: any all-coaches-missing observation fails.
        pi = previous_pose_indices(pose_times, mapping["source_times_s"], config.max_pose_age_s)
        pose_valid = np.zeros((n, boxes.shape[1]), bool)
        for i, source_time in enumerate(mapping["source_times_s"]):
            _, _, pose_valid[i] = boxes_at_time(boxes, pose_times, source_time, config.max_pose_age_s)
        pose_timestamp_valid = np.ones(n, bool)
        frame_valid = np.ones(n, bool)
    sampled_pose_times = np.zeros(n, np.float64)
    sampled_pose_times[pi >= 0] = pose_times[pi[pi >= 0]]
    box_dilation, box_minimum = baseline_dilation(
        boxes, pi[frame_valid], original_wh, decode_wh, config.min_retained_pixels)
    arrays = {mode: _frame_arrays(n) for mode in MODES}
    previous = {mode: None for mode in MODES}
    sam_score = np.zeros((n, 2), np.float32)
    ui = ui_mask(decode_wh)
    chosen_audits = audit_indices(n) if config.audit else []
    frames = iter_rgb_frames(row["source_path"], mapping, decode_wh)
    try:
        for i, rgb in enumerate(frames):
            source_time = mapping["source_times_s"][i]
            if not frame_valid[i]:
                # Arrays were initialized as invalid zero placeholders. Neither
                # method observes a fabricated human mask/background/label.
                for mode in MODES:
                    previous[mode] = None
                if i in chosen_audits:
                    require_disk_floor(directory, reserve_bytes=rgb.nbytes+1024**2)
                    save_invalid_audit(directory/f"audit_{i:04d}.png", rgb)
                if (i+1) % 100 == 0:
                    print(json.dumps({"recording_id": row["recording_id"], "frames": i+1,
                                      "invalid_frames_so_far": int((~frame_valid[:i+1]).sum()),
                                      "elapsed_s": time.monotonic()-started}), flush=True)
                del rgb
                continue
            original_prompts, _, _ = boxes_at_time(boxes, pose_times, source_time, config.max_pose_age_s)
            prompts = transform_boxes(original_prompts, original_wh, decode_wh)
            foreground, scores = segmenter(rgb, prompts)
            sam_score[i] = (float(np.min(scores)), float(np.mean(scores)))
            masks = {
                "box": resize_mask(original_prompts, original_wh, decode_wh, dilation=box_dilation) != 0,
                "sam2": retained_mask(foreground, ui, config.mask_dilation_px),
            }
            for mode, retained in masks.items():
                if int(retained.sum()) < config.min_retained_pixels:
                    raise ValueError(f"{mode}: below fixed retained-pixel safety floor at {source_time:.6f}")
                target = targets.extract_color_targets(rgb, retained)
                out = arrays[mode]
                out["hs_histogram"][i], out["rich96"][i] = target.hs_histogram, target.rich
                out["value_summary"][i] = (target.value.mean, target.value.std, target.value.p90, target.value.maximum)
                for key in ("color_valid", "quadrant_color_valid", "retained_pixels", "valid_pixels",
                            "retained_fraction", "valid_fraction", "chromatic_fraction"):
                    out[key][i] = getattr(target, key)
                out["value_valid"][i] = target.value.valid
                out["frame_valid"][i] = True
                if previous[mode] is not None:
                    out["mask_activity_known"][i] = True
                    out["mask_activity"][i] = np.mean(previous[mode] != retained)
                previous[mode] = retained
            if i in chosen_audits:
                require_disk_floor(directory, reserve_bytes=3*rgb.nbytes+1024**2)
                save_audit(directory/f"audit_{i:04d}.png", rgb, masks["box"], masks["sam2"])
            if (i+1) % 100 == 0:
                print(json.dumps({"recording_id": row["recording_id"], "frames": i+1,
                                  "elapsed_s": time.monotonic()-started}), flush=True)
            del rgb, masks, foreground
    finally:
        frames.close()
    artifacts = {}
    summaries = {}
    for mode, data in arrays.items():
        novelty = targets.chromatic_novelty(
            mapping["source_times_s"], data["hs_histogram"], data["color_valid"],
            retained_fraction=data["retained_fraction"], chromatic_fraction=data["chromatic_fraction"],
            mask_activity=data["mask_activity"])
        output = {key: value.astype(np.float16) if value.dtype.kind == "f" else value
                  for key, value in data.items()}
        for key, value in vars(novelty).items():
            if isinstance(value, np.ndarray):
                output["novelty_"+key] = value.astype(np.float16) if value.dtype.kind == "f" and key != "times_s" else value
        output.update({key: mapping[key] for key in ("source_times_s", "requested_times_s", "source_pts", "frame_indices")})
        output.update(pose_indices=pi.astype(np.int32), pose_times_s=sampled_pose_times,
                      pose_timestamp_valid=pose_timestamp_valid, pose_coach_valid=pose_valid,
                      sam_prediction_valid=frame_valid, sam_predicted_score_min_mean=sam_score.astype(np.float16))
        # Activity zeros at start/gaps are unknown, not measured stability.
        known = data["mask_activity_known"]
        times = mapping["source_times_s"]
        activity_window_known = np.zeros(n, bool)
        for i, center in enumerate(times):
            boundaries = (times > center-targets.NOVELTY_WINDOW_S+1e-10) & (times < center+targets.NOVELTY_WINDOW_S-1e-10)
            activity_window_known[i] = novelty.window_eligible[i] and bool(known[boundaries].all())
        output["novelty_window_mask_activity_known"] = activity_window_known
        path = directory/f"{mode}.npz"
        require_disk_floor(directory, reserve_bytes=sum(a.nbytes for a in output.values()) + 1024**2)
        write_arrays(path, output)
        artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        summaries[mode] = {"retained_fraction_mean": float(data["retained_fraction"][frame_valid].mean()),
                           "retained_fraction_min": float(data["retained_fraction"][frame_valid].min()),
                           "chromatic_fraction_mean": float(data["chromatic_fraction"][frame_valid].mean()),
                           "coverage_summary_scope": "supported frames only; invalid zero placeholders excluded",
                           "color_valid_frames": int(data["color_valid"].sum()),
                           "mask_activity_known_boundaries": int(known.sum()),
                           "mask_activity_mean": float(data["mask_activity"][known].mean()) if known.any() else None}
    receipt = {
        "status": "complete", "recording_id": row["recording_id"], "canonical_id": row["canonical_id"],
        "partition": row["partition"], "input": row, "config": asdict(config), "frames": n,
        "valid_frames": int(frame_valid.sum()), "invalid_frames": int((~frame_valid).sum()),
        "frame_valid_fraction": float(frame_valid.mean()),
        "gap_policy": "invalidate_frame_no_imputation" if config.invalidate_missing_frames else "strict_record",
        "validity_contract": "zero placeholders are not labels; require frame/color/value and novelty eligibility masks for losses",

        "modes": summaries, "artifacts": artifacts, "elapsed_s": time.monotonic()-started,
        "gpu_peak_allocated_bytes": segmenter.peak_bytes(),
        "geometry": {"original_wh": original_wh, "decode_wh": decode_wh,
                     "original_to_decode_scale_xy": [decode_wh[0]/original_wh[0], decode_wh[1]/original_wh[1]],
                     "ui_normalized_xyxy": UI, "box_rasterization": "floor start; ceil end + 1, Stage11 convention"},
        "sampling": {"fps": FPS, "policy": "nearest source PTS; ties earlier; distinct source frames required",
                     "time_base": mapping["time_base"], "stream_start_time_s": mapping["stream_start_time_s"],
                     "max_abs_grid_error_s": float(np.abs(mapping["source_times_s"]-requested).max()),
                     "source_times": "full-record relative seconds; raw source_pts also preserved"},
        "box_baseline": {"dilation": box_dilation, "minimum_visible_pixels": box_minimum,
                         "policy": "Stage11 whole-window .15 then 0 fallback; no per-frame padding adaptation"},
        "pose": {"policy": "previous timestamp on actual source time",
                 "max_age_s": float(np.abs(sampled_pose_times[pose_timestamp_valid]-mapping["source_times_s"][pose_timestamp_valid]).max()),
                 "all_coaches_present_fraction": float(pose_valid.all(axis=1).mean()),
                 "timestamp_valid_fraction": float(pose_timestamp_valid.mean()),
                 "registered_coach_indices": registered_indices.tolist(),
                 "unregistered_slot_indices": sorted(set(range(boxes.shape[1]))-set(registered_indices.tolist())),
                 "unregistered_slot_count": boxes.shape[1]-len(registered_indices),
                 "human_presence_scope": "registered pose slots only; no claim that unknown people are absent or segmented",
                 "all_registered_coaches_present_fraction": float(pose_valid[:, registered_indices].all(axis=1).mean()),
                 "cache_metadata": pose_meta},
        "novelty": {"window_s": targets.NOVELTY_WINDOW_S, "max_gap_s": targets.MAX_SAMPLE_GAP_S,
                    "cut_annotations": "not supplied; no independent shot-cut claim", "event_threshold": "not fitted or applied"},
        "value_summary_columns": ["mean", "std", "p90", "maximum"],
        "audit": {"frame_indices": chosen_audits, "panels": ["decoded original RGB", "box + UI exclusion", "SAM2 + UI exclusion"],
                  "invalid_frame_indices": [i for i in chosen_audits if not frame_valid[i]],
                  "invalid_frame_panels": "original RGB only, marked invalid; no computed masks or overlays",
                  "interpretation": "perception aids only, not segmentation ground truth"},
        "claims": "coverage/activity/runtime only; no independent IoU or segmentation accuracy without annotations",
    }
    write_json(directory/"complete.json", receipt)
    return receipt


def validate_manifest(manifest, checkpoint, limit_records=None, *, allow_development=False,
                      invalidate_missing_frames=False):
    for name, flag in (("allow_development", allow_development), ("invalidate_missing_frames", invalidate_missing_frames)):
        if type(flag) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if invalidate_missing_frames and manifest.get("gap_policy") != "invalidate_frame_no_imputation":
        raise ValueError("--invalidate-missing-frames requires manifest gap_policy=invalidate_frame_no_imputation")
    if allow_development and manifest.get("exposure") != "historical_development":
        raise ValueError("--allow-development requires manifest exposure=historical_development")
    rows = manifest.get("inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen manifest needs nonempty inputs")
    required = {"recording_id", "canonical_id", "partition", "source_path", "source_sha256", "pose_path",
                "pose_sha256", "start_s", "duration_s", "original_width", "original_height"}
    ids = set()
    # Validate the ENTIRE manifest before touching data/checkpoint files.
    # A prefix limit never hides forbidden TEST or undeclared VAL exposure.
    allowed_partitions = ("train", "val") if allow_development else ("train",)
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("every manifest input must be a complete TRAIN/development row")
        if row["partition"] not in allowed_partitions:
            raise ValueError("TEST opening always forbidden; historical VAL requires explicit development gate; default TRAIN only")
        if invalidate_missing_frames:
            coaches = row.get("coaches")
            if (not isinstance(coaches, list) or not coaches
                    or any(not isinstance(c, dict) or type(c.get("coach_index")) is not int
                           or c["coach_index"] < 0 for c in coaches)
                    or len({c["coach_index"] for c in coaches}) != len(coaches)):
                raise ValueError("gap opt-in requires nonempty unique registered coaches[].coach_index")
        rid = row["recording_id"]
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", rid) or rid in ids:
            raise ValueError("unsafe or duplicate recording_id")
        ids.add(rid)
        sample_grid(row["start_s"], row["duration_s"])
        if any(type(row[k]) is not int or row[k] <= 0 for k in ("original_width", "original_height")):
            raise ValueError("manifest dimensions must be positive integers")
    if limit_records is not None and (type(limit_records) is not int or not 1 <= limit_records <= len(rows)):
        raise ValueError("limit-records must select a nonempty prefix of frozen inputs")
    selected = rows if limit_records is None else rows[:limit_records]
    if not Path(checkpoint).is_file() or not Path(checkpoint).name.endswith("sam2.1_s.pt"):
        raise ValueError("existing official SAM2.1 Small checkpoint must end in sam2.1_s.pt; no downloads")
    if sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint hash differs from frozen manifest")
    for row in selected:
        for key in ("source", "pose"):
            if sha256(row[key+"_path"]) != row[key+"_sha256"]:
                raise ValueError(f"{row['recording_id']}: {key} hash differs from frozen manifest")
    return selected


def run(manifest_path, checkpoint, output, config, *, limit_records=None):
    manifest_path, checkpoint, output = Path(manifest_path), Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("output already exists; choose a new run directory, never overwrite")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rows = validate_manifest(manifest, checkpoint, limit_records,
                             allow_development=config.allow_development,
                             invalidate_missing_frames=config.invalidate_missing_frames)
    require_disk_floor(output, reserve_bytes=16*1024**2)
    output.mkdir(parents=True, exist_ok=False)
    # Fixed safety/config/selection recorded BEFORE CUDA initialization.
    config_record = {"schema_version": 1, "manifest_path": str(manifest_path.resolve()),
                     "manifest_sha256": manifest_hash, "checkpoint_path": str(checkpoint.resolve()),
                     "checkpoint_sha256": manifest["checkpoint_sha256"], "config": asdict(config),
                     "exposure": manifest.get("exposure", "train_only"),
                     "allowed_partitions": ["train", "val"] if config.allow_development else ["train"],
                     "gap_policy": "invalidate_frame_no_imputation" if config.invalidate_missing_frames else "strict_record",
                     "selection": "all frozen inputs" if limit_records is None else "frozen manifest prefix",
                     "recording_ids": [row["recording_id"] for row in rows], "fps": FPS,
                     "runner_sha256": sha256(__file__), "targets_sha256": sha256(targets.__file__),
                     "disk_floor_bytes": DISK_FLOOR_BYTES, "ui_normalized_xyxy": UI}
    write_json(output/"run_config.json", config_record)
    segmenter = SAM2FrameSegmenter(checkpoint, config)
    write_json(output/"wrapper.json", segmenter.provenance)
    results = []
    for row in rows:
        require_disk_floor(output, reserve_bytes=16*1024**2)
        directory = output/row["recording_id"]
        directory.mkdir(exist_ok=False)
        started = time.monotonic()
        try:
            result = extract_record(row, directory, segmenter, config)
        except Exception as exc:
            result = {"status": "invalid", "recording_id": row["recording_id"], "error": f"{type(exc).__name__}: {exc}",
                      "elapsed_s": time.monotonic()-started, "gpu_peak_allocated_bytes": segmenter.peak_bytes()}
            write_json(directory/"invalid.json", result)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("recording_id", "status", "elapsed_s", "gpu_peak_allocated_bytes")}
                         | {"frames": result.get("frames", 0), "valid_frames": result.get("valid_frames", 0),
                            "invalid_frames": result.get("invalid_frames"), "frame_valid_fraction": result.get("frame_valid_fraction"),
                            "modes": result.get("modes"), "error": result.get("error")}), flush=True)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("manifest changed during run; no campaign completion receipt")
    summary = {"status": "complete" if all(r["status"] == "complete" for r in results) else "complete_with_invalid_records",
               "manifest_sha256": manifest_hash, "results": results}
    write_json(output/"results.json", summary)
    write_json(output/"complete.json", {"status": summary["status"], "results_sha256": sha256(output/"results.json"),
                                       "records": len(results), "complete_records": sum(r["status"] == "complete" for r in results)})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, help="prefix of frozen manifest only")
    parser.add_argument("--decode-width", type=int, default=640)
    parser.add_argument("--mask-dilation-px", type=int, default=0)
    parser.add_argument("--max-pose-age-s", type=float, default=.2)
    parser.add_argument("--max-frame-error-s", type=float, default=.1)
    parser.add_argument("--min-retained-pixels", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--allow-development", action="store_true",
                        help="permit historical VAL only with manifest exposure=historical_development; TEST never allowed")
    parser.add_argument("--invalidate-missing-frames", action="store_true",
                        help="keep missing-pose observations as invalid zero placeholders only with matching frozen gap_policy")
    args = parser.parse_args(argv)
    config = VisualConfig(**{key: getattr(args, key) for key in VisualConfig.__dataclass_fields__})
    result = run(args.manifest, args.checkpoint, args.output, config, limit_records=args.limit_records)
    return 0 if result["status"] == "complete" else 2
