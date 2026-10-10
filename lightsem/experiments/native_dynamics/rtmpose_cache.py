"""RTMPose observations with detector-native boxes and conservative identities.

Raw detections and keypoint scores survive admission. Scores are metadata, never
an admission threshold. Tracking never fills an absent observation or merges
two existing track IDs. The cache is monocular model output, not 3D truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from lightsem.experiments.native_dynamics.data import Quality

MODEL_DIR = Path('/mnt/lightsem-data/torch-home/hub/checkpoints')
DETECTOR = MODEL_DIR/'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'
FP32_POSE_MODEL = MODEL_DIR/'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx'
BF16_POSE_ENGINE = Path(os.environ.get(
    'RTMPOSE_BF16_ENGINE',
    '/root/lightsem/experiments/pose_bf16_20261009/rtmpose_m_autocast_bf16.plan'))
POSE_PROFILE = os.environ.get('RTMPOSE_POSE_PROFILE', 'bf16').lower()
if POSE_PROFILE == 'bf16':
    POSE_MODEL = BF16_POSE_ENGINE
elif POSE_PROFILE == 'fp32':
    POSE_MODEL = FP32_POSE_MODEL
else:
    raise ValueError(f'Unsupported RTMPOSE_POSE_PROFILE={POSE_PROFILE!r}; use bf16 or fp32')
COCO = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def infer_people(detector, pose_model, frame):
    """Empty detector output MUST NOT invoke rtmlib's whole-frame fallback."""
    boxes = np.asarray(detector(frame), dtype=np.float32).reshape(-1, 4)
    boxes = boxes[np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) &
                  (boxes[:, 3] > boxes[:, 1])]
    if not len(boxes):
        return boxes, np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
    keypoints, scores = pose_model(frame, bboxes=boxes)
    keypoints, scores = np.asarray(keypoints, np.float32), np.asarray(scores, np.float32)
    if keypoints.shape != (len(boxes), 17, 2) or scores.shape != (len(boxes), 17):
        raise ValueError('RTMPose COCO17 output contract changed')
    return boxes, keypoints, scores


class TensorRTPose:
    """TensorRT AutoCast BF16 RTMPose pose stage with FP32 I/O.

    The YOLOX detector remains the CUDA ONNX stage. The engine is deliberately
    pose-only because the HumanArt detector's static TopK graph does not build
    under the installed TensorRT release. Raw boxes and PTS remain unchanged.
    """
    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda.bindings import runtime
        from rtmlib.tools.pose_estimation.pre_processings import (
            bbox_xyxy2cs, top_down_affine)
        self.trt = trt
        self.runtime = runtime
        self.bbox_xyxy2cs = bbox_xyxy2cs
        self.top_down_affine = top_down_affine
        logger = trt.Logger(trt.Logger.WARNING)
        with Path(engine_path).open('rb') as stream:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError(f'Unable to deserialize BF16 TensorRT engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape('input', (1, 3, 256, 192))
        self.stream = runtime.cudaStreamCreate()[1]
        self.host = {}
        self.device = {}
        for name in ('input', 'simcc_x', 'simcc_y'):
            shape = tuple(self.context.get_tensor_shape(name))
            array = np.empty(shape, np.float32)
            self.host[name] = array
            self.device[name] = runtime.cudaMalloc(array.nbytes)[1]
            self.context.set_tensor_address(name, self.device[name])
        self.mean = np.array((123.675, 116.28, 103.53), np.float32)
        self.std = np.array((58.395, 57.12, 57.375), np.float32)

    def __call__(self, image, bboxes=None):
        bboxes = [] if bboxes is None else list(bboxes)
        if not bboxes:
            return (np.empty((0, 17, 2), np.float32),
                    np.empty((0, 17), np.float32))
        keypoints, scores = [], []
        for bbox in bboxes:
            center, scale = self.bbox_xyxy2cs(np.asarray(bbox), padding=1.25)
            crop, scale = self.top_down_affine((192, 256), scale, center, image)
            crop = (crop.astype(np.float32) - self.mean) / self.std
            inp = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], np.float32)
            self.host['input'][...] = inp
            self.runtime.cudaMemcpyAsync(
                self.device['input'], self.host['input'].ctypes.data,
                self.host['input'].nbytes,
                self.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream)
            if not self.context.execute_async_v3(self.stream):
                raise RuntimeError('BF16 TensorRT pose execution failed')
            for name in ('simcc_x', 'simcc_y'):
                self.runtime.cudaMemcpyAsync(
                    self.host[name].ctypes.data, self.device[name],
                    self.host[name].nbytes,
                    self.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream)
            self.runtime.cudaStreamSynchronize(self.stream)
            simcc_x, simcc_y = self.host['simcc_x'], self.host['simcc_y']
            x = np.argmax(simcc_x[0], axis=-1)
            y = np.argmax(simcc_y[0], axis=-1)
            score = (np.max(simcc_x[0], axis=-1) +
                     np.max(simcc_y[0], axis=-1)) * .5
            points = np.stack([x / 2., y / 2.], axis=-1)
            points = points / np.array([192., 256.], np.float32) * scale
            points = points + center - scale / 2
            keypoints.append(points.astype(np.float32))
            scores.append(score.astype(np.float32))
        return np.stack(keypoints), np.stack(scores)


class Backend:
    def __init__(self, profile_root=None):
        import onnxruntime as ort
        from rtmlib import YOLOX
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('GPU ONNX Runtime is required; no CPU fallback')
        # Use local frozen ONNX files; no runtime downloads or different weights.
        # This is the detector's published person/background decision. Joint
        # SimCC scores are not probabilities and are never admission gates.
        self.detector = YOLOX(str(DETECTOR), score_thr=.7,
                              backend='onnxruntime', device='cuda')
        if POSE_PROFILE == 'bf16':
            self.pose = TensorRTPose(str(POSE_MODEL))
        else:
            from rtmlib import RTMPose
            self.pose = RTMPose(str(POSE_MODEL), model_input_size=(192, 256),
                                backend='onnxruntime', device='cuda', to_openpose=False)
        self.sessions = []
        for name, tool in [('detector', self.detector)] + (
                [] if POSE_PROFILE == 'bf16' else [('pose', self.pose)]):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            if profile_root is not None:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(profile_root)/name)
            tool.session = ort.InferenceSession(tool.onnx_model, sess_options=options,
                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            if tool.session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'{name} silently fell back to CPU')
            self.sessions.append((name, tool.session))
        self.profile_enabled = profile_root is not None

    def __call__(self, frame):
        return infer_people(self.detector, self.pose, frame)

    def batch(self, frames):
        """Run detector and top-down pose in batches while preserving PTS order."""
        frames = list(frames)
        if not frames:
            return []
        # The BF16 TensorRT wrapper intentionally exposes only the validated
        # single-crop path (it has no rtmlib preprocess/postprocess methods).
        # Always use that path here; entering the generic dynamic-batch branch
        # would fabricate an API contract and fail before writing observations.
        if POSE_PROFILE == 'bf16':
            return [self(frame) for frame in frames]
        # The pinned OpenMMLab ONNX exports have a static batch dimension of
        # one. Do not reshape or pad it: a batch-size request falls back to
        # the exact single-frame path until a separately authenticated dynamic
        # export is available.
        if len(frames) > 1 and self.detector.session.get_inputs()[0].shape[0] == 1:
            return [self(frame) for frame in frames]
        det_images, ratios = zip(*(self.detector.preprocess(frame) for frame in frames))
        det_input = np.ascontiguousarray(np.stack(det_images).transpose(0, 3, 1, 2), dtype=np.float32)
        det_name = self.detector.session.get_inputs()[0].name
        det_outputs = self.detector.session.run(None, {det_name: det_input})[0]
        boxes_per = [self.detector.postprocess(det_outputs[i:i+1], ratios[i]).reshape(-1, 4)
                     for i in range(len(frames))]
        pose_images, centers, scales, owners = [], [], [], []
        for fi, (frame, boxes) in enumerate(zip(frames, boxes_per)):
            for box in boxes:
                image, center, scale = self.pose.preprocess(frame, box)
                pose_images.append(image); centers.append(center); scales.append(scale); owners.append(fi)
        result = [(b, np.empty((len(b), 17, 2), np.float32), np.empty((len(b), 17), np.float32))
                  for b in boxes_per]
        if not pose_images:
            return [(b, np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)) for b in boxes_per]
        pose_input = np.ascontiguousarray(np.stack(pose_images).transpose(0, 3, 1, 2), dtype=np.float32)
        pose_name = self.pose.session.get_inputs()[0].name
        pose_outputs = self.pose.session.run(None, {pose_name: pose_input})
        k_per = [[] for _ in frames]; c_per = [[] for _ in frames]
        for i, fi in enumerate(owners):
            key, score = self.pose.postprocess([o[i:i+1] for o in pose_outputs], centers[i], scales[i])
            k_per[fi].append(key[0]); c_per[fi].append(score[0])
        return [(boxes_per[i], np.asarray(k_per[i], np.float32).reshape(-1,17,2),
                 np.asarray(c_per[i], np.float32).reshape(-1,17)) for i in range(len(frames))]

    def finish_profiles(self):
        if not self.profile_enabled:
            raise RuntimeError('Profiling was not enabled')
        results = {}
        for name, session in self.sessions:
            path = Path(session.end_profiling())
            events = json.loads(path.read_text())
            kernels = [e.get('args', {}) for e in events if e.get('cat') == 'Node']
            conv = [e for e in kernels if 'Conv' in e.get('op_name', '')]
            cuda = [e for e in conv if e.get('provider') == 'CUDAExecutionProvider']
            if not cuda:
                raise RuntimeError(f'No CUDA convolution observed for {name}')
            results[name] = {'profile': str(path), 'sha256': sha(path),
                             'cuda_convolutions': len(cuda), 'convolutions': len(conv)}
        return results


def coach_candidates(boxes, width, height):
    """Geometric UI filter only; retain original detections in the raw cache."""
    boxes = np.asarray(boxes)
    cx = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
    cy = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
    return ((boxes[:, 3] - boxes[:, 1] >= .22 * height) & (cy > .14) &
            ~((cx >= .70) & (cy >= .72)))


def appearance(frame, box):
    import cv2
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    dx, dy = x1-x0, y1-y0
    a, b = max(0, int(x0+.2*dx)), min(w, int(x1-.2*dx)+1)
    c, d = max(0, int(y0+.15*dy)), min(h, int(y1-.25*dy)+1)
    if b <= a or d <= c:
        return np.zeros(48, np.float32)
    hsv = cv2.cvtColor(cv2.resize(frame[c:d, a:b], (24, 32)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).reshape(-1)
    return hist / max(float(hist.sum()), 1)


@dataclass
class Track:
    identity: int
    time: float
    box: np.ndarray
    style: np.ndarray


class CoachTracker:
    """One-to-one matching within 0.5 s; ambiguous/cut transitions split IDs."""
    def __init__(self, maximum_gap_s=.5):
        self.maximum_gap_s = maximum_gap_s
        self.active = []
        self.next_id = 0

    def step(self, time, boxes, styles, *, cut=False):
        if cut:
            self.active.clear()
        self.active = [r for r in self.active if time-r.time <= self.maximum_gap_s]
        ids = np.full(len(boxes), -1, np.int32)
        costs = np.full((len(self.active), len(boxes)), 1e6)
        ambiguous_tracks = set()
        for i, old in enumerate(self.active):
            a = old.box
            for j, b in enumerate(boxes):
                scale = max(a[3]-a[1], b[3]-b[1], 1)
                delta = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/scale
                style = .5*float(np.abs(old.style-styles[j]).sum())
                inter = float(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod())
                area = float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter)
                if delta <= 1.0 and style <= 2.0:
                    costs[i, j] = delta+.25*(1-inter/max(area, 1))+.35*style
        if costs.size:
            rr, cc = linear_sum_assignment(costs)
            for i, j in zip(rr, cc):
                if costs[i, j] >= 1e5:
                    continue
                others = np.delete(costs[:, j], i)
                row_others = np.delete(costs[i], j)
                # Equal-cost alternatives mean the image evidence cannot
                # identify which person continued. Split the identity instead
                # of silently accepting scipy's arbitrary tie break.
                assigned = costs[i, j]
                if ((others.size and np.min(others) <= assigned + 1e-6) or
                        (row_others.size and np.min(row_others) <= assigned + 1e-6)):
                    ambiguous_tracks.add(i)
                    if others.size and np.min(others) <= assigned + 1e-6:
                        ambiguous_tracks.update(np.flatnonzero(costs[:, j] <= assigned + 1e-6))
                    if row_others.size and np.min(row_others) <= assigned + 1e-6:
                        ambiguous_tracks.add(i)
                    continue
                ids[j] = self.active[i].identity
                self.active[i] = Track(int(ids[j]), time, boxes[j].copy(), styles[j].copy())
        if ambiguous_tracks:
            self.active = [track for i, track in enumerate(self.active) if i not in ambiguous_tracks]
        for j in np.flatnonzero(ids < 0):
            ids[j] = self.next_id
            self.next_id += 1
            self.active.append(Track(int(ids[j]), time, boxes[j].copy(), styles[j].copy()))
        return ids


def masked_track(k, boxes, times, duration_s, height, *, confidence=None,
                 confidence_gate=False):
    """Apply geometry/timestamp quality; scores remain metadata by default."""
    k = np.asarray(k, np.float64)
    times = np.asarray(times, np.float64)
    boxes = np.asarray(boxes, np.float64)
    confidence = np.asarray(confidence, np.float64) if confidence is not None else None
    cfg = Quality()
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Strict finite source timestamps required')
    if k.shape != (len(times),17,2) or boxes.shape != (len(times),4):
        raise ValueError('Invalid keypoint/box layout')
    if confidence is None or confidence.shape != (len(times),17):
        raise ValueError('Confidence array required with shape (frames,17)')
    visible = (np.isfinite(boxes).all(-1) &
               ((boxes[:, 3]-boxes[:, 1]) >= cfg.minimum_box_height_fraction*height))
    valid = np.isfinite(k).all(-1) & visible[:, None]
    if confidence_gate:
        valid &= np.isfinite(confidence) & (confidence >= cfg.confidence)
    shoulder, hip = k[:, [5,6]].mean(1), k[:, [11,12]].mean(1)
    length = np.linalg.norm(shoulder-hip, axis=-1)
    torso = (valid[:, [5,6,11,12]].all(1) & (length > .05*height))
    if confidence_gate:
        torso &= (confidence[:, [5,6,11,12]] >= cfg.torso_confidence).all(1)
    if np.count_nonzero(torso) < 30:
        raise ValueError('Insufficient finite nondegenerate torso geometry')
    scale = float(np.median(length[torso]))
    pair = valid[1:] & valid[:-1]
    jump = (np.linalg.norm(np.diff(k, axis=0), axis=-1)/scale > 1.25) & pair
    valid[1:] &= ~jump
    valid[:-1] &= ~jump
    valid &= valid[:, [11,12]].all(1)[:, None]
    normalized = (k-hip[:, None])/scale
    normalized[..., 1] *= -1
    normalized = np.where(valid[...,None], normalized, 0)[:, COCO]
    valid = valid[:, COCO]
    target = np.arange(int(np.ceil(duration_s*30)), dtype=np.float64)/30
    upper = np.searchsorted(times, target, side='left')
    upper = np.clip(upper, 1, len(times)-1)
    lower = upper-1
    gap = times[upper]-times[lower]
    ok = (target >= times[0]) & (target <= times[-1]) & (gap <= .060)
    mask = valid[lower] & valid[upper] & ok[:,None]
    fraction = (target-times[lower])/gap
    values = normalized[lower]*(1-fraction[:,None,None])+normalized[upper]*fraction[:,None,None]
    # An exact sample does not depend on the preceding sample/gap.
    for index in (upper, lower):
        exact = np.isclose(target, times[index], atol=1e-8, rtol=0)
        values[exact], mask[exact] = normalized[index[exact]], valid[index[exact]]
    values = np.where(mask[...,None], values, 0).astype(np.float32)
    good = mask.sum(1) >= 10
    if len(good) < 480:
        starts = np.empty(0, np.int32)
    else:
        counts = np.r_[0, np.cumsum(good)]
        starts = np.flatnonzero(counts[480:]-counts[:-480] >= 384)[::15].astype(np.int32)
    return {'time':target, 'pose':values, 'mask':mask, 'starts':starts}, {
        'coverage':float(good.mean()), 'eligible_windows':len(starts),
        'joint_jump_edges':int(jump.sum()), 'torso_scale_px':scale,
        'confidence_filter':bool(confidence_gate),
        'confidence_threshold':cfg.confidence if confidence_gate else None,
        'torso_confidence_threshold':cfg.torso_confidence if confidence_gate else None,
        'minimum_box_height_fraction':cfg.minimum_box_height_fraction,
        'whole_record_filter':False,
        'valid_observation_seconds':float(good.sum()/30)}
