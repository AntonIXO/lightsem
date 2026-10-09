"""Sequential, authenticated RTMPose cache with separate track identities.

Stores ragged frame detections, original PTS, scores, boxes, candidate masks,
and track IDs. No post-hoc merging, averaging, filling, or whole-record gate.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import av
import numpy as np

from lightsem.experiments.native_dynamics.rtmpose_cache import (
    Backend, CoachTracker, DETECTOR, POSE_MODEL, POSE_PROFILE, coach_candidates,
    sha, write_new,
)


def dedupe_geometry(boxes):
    """Remove overlapping detections of one figure without using scores."""
    boxes = np.asarray(boxes, np.float32)
    keep = []
    order = sorted(range(len(boxes)), key=lambda j: -float(
        max(0., boxes[j,2]-boxes[j,0])*max(0., boxes[j,3]-boxes[j,1])))
    for j in order:
        b = boxes[j]; area = max(0., b[2]-b[0])*max(0., b[3]-b[1]); duplicate=False
        for k in keep:
            a = boxes[k]
            inter = max(0., min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
            other = max(0., a[2]-a[0])*max(0.,a[3]-a[1])
            union = area+other-inter
            scale = max(a[3]-a[1],b[3]-b[1],1.)
            center = np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2.)
            if (union and inter/union > .5) or (min(area,other) and inter/min(area,other) > .8) or center < .25*scale:
                duplicate=True; break
        if not duplicate: keep.append(j)
    return np.asarray(sorted(keep), np.int64)

LOCKS = [Path('/tmp/lightsem-audio-understanding-gpu.lock'),
         Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock')]


@contextmanager
def gpu_locks():
    if os.environ.get('RTMPOSE_LOCK_ALREADY_HELD') == '1':
        yield
        return
    with ExitStack() as stack:
        for path in LOCKS:
            handle = stack.enter_context(path.open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        other = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True).strip()
        if other:
            raise RuntimeError(f'Accelerator already occupied: {other}')
        yield


def decode(path, *, start_s=0., max_frames=None):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate)
        stride = max(1, round(fps / 30.))
        previous = -np.inf
        count = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise ValueError('Missing source frame PTS; no synthetic clock permitted')
            stamp = float(frame.time)
            if stamp <= previous:
                raise ValueError('Non-increasing video PTS')
            previous = stamp
            if stamp < start_s or frame_index % stride:
                continue
            if max_frames is not None and count >= max_frames:
                break
            yield stamp, frame.to_ndarray(format='bgr24'), fps, stride
            count += 1


def extract(backend, source, *, start_s=0., max_frames=None, batch_size=8):
    tracker = CoachTracker()
    stamps, all_k, all_c, all_b, all_ids, all_candidates, frame_ids = [], [], [], [], [], [], []
    begun = time.monotonic(); pending=[]; frame_counter=0
    width = height = fps = stride = None
    def consume(items):
        nonlocal frame_counter, width, height, fps, stride
        if not items: return
        predictions=backend.batch([item[1] for item in items])
        for (stamp,image,fps,stride),(boxes,keypoints,scores) in zip(items,predictions):
            height, width = image.shape[:2]
            candidate = coach_candidates(boxes, width, height)
            selected_indices = np.flatnonzero(candidate)
            selected_indices = selected_indices[dedupe_geometry(boxes[selected_indices])]
            selected = boxes[selected_indices]
            styles = [np.zeros(48, np.float32) for _ in selected]
            identities = np.full(len(boxes), -1, np.int32)
            identities[selected_indices] = tracker.step(stamp, selected, styles)
            stamps.append(stamp); all_k.append(keypoints); all_c.append(scores); all_b.append(boxes)
            all_ids.append(identities); all_candidates.append(candidate)
            frame_ids.append(np.full(len(boxes), frame_counter, np.int32)); frame_counter += 1
    for item in decode(source, start_s=start_s, max_frames=max_frames):
        pending.append(item)
        if len(pending) >= batch_size:
            consume(pending); pending=[]
    consume(pending)
    '''
    for i, (stamp, image, fps, stride) in enumerate(decode(source, start_s=start_s, max_frames=max_frames)):
        height, width = image.shape[:2]
        boxes, keypoints, scores = backend(image)
        candidate = coach_candidates(boxes, width, height)
        selected_indices = np.flatnonzero(candidate)
        selected_indices = selected_indices[dedupe_geometry(boxes[selected_indices])]
        selected = boxes[selected_indices]
        # Appearance histograms are not needed for identity here and made the
        # CPU path dominate runtime. Geometry/one-to-one assignment is the
        # identity contract; keep a zero style vector for API compatibility.
        styles = [np.zeros(48, np.float32) for _ in selected]
        identities = np.full(len(boxes), -1, np.int32)
        identities[selected_indices] = tracker.step(stamp, selected, styles)
        stamps.append(stamp)
        all_k.append(keypoints); all_c.append(scores); all_b.append(boxes)
        all_ids.append(identities); all_candidates.append(candidate)
        frame_ids.append(np.full(len(boxes), i, np.int32))
    '''
    if not stamps:
        raise ValueError('Video yielded no source frames')
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(source)], text=True))
    audio = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    meta = {
        'schema_version': 2, 'source_path': str(source), 'source_sha256': sha(source),
        'n_frames': len(stamps), 'width': width, 'height': height, 'fps_src': fps,
        'stride': stride, 'fps_eff': fps/stride, 'n_track_segments': tracker.next_id,
        'audio_start_s': float(audio.get('start_time', 0.)) if audio else None,
        'start_s': start_s, 'max_frames': max_frames,
        'model': f'rtmpose-m-body7-{POSE_PROFILE}', 'detector': 'yolox-m-humanart',
        'pose_backend': 'tensorrt_autocast' if POSE_PROFILE == 'bf16' else 'onnxruntime_cuda',
        'detector_score_threshold': 0.7, 'keypoint_score_threshold': None,
        'masks_mean': 'Finite model estimates; not verified anatomical visibility',
        'track_contract': 'one-to-one; max gap 0.5s; split ambiguous identity; never merge or fill',
        'seconds': time.monotonic()-begun,
    }
    return dict(t=np.asarray(stamps, np.float64), K=np.concatenate(all_k),
                C=np.concatenate(all_c), box=np.concatenate(all_b),
                track_id=np.concatenate(all_ids), candidate=np.concatenate(all_candidates),
                frame_index=np.concatenate(frame_ids), meta=np.asarray(json.dumps(meta)))


def save_new(path, arrays):
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix('.partial.npz')
    with partial.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, default=Path('/mnt/lightsem-data/cache/justdance_manifest.json'))
    p.add_argument('--ids-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-s', type=float, default=0.)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--no-lock', action='store_true',
                   help='parent process owns the exclusive GPU locks')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ids = sorted({s.strip() for s in a.ids_file.read_text().splitlines() if s.strip()})
    rows = {r['yt_id']: r for r in json.loads(a.catalog.read_text()) if r.get('yt_id')}
    if set(ids)-rows.keys():
        raise ValueError('IDs absent from source catalog')
    identity = {'extractor_sha256': sha(Path(__file__)),
                'tracking_sha256': sha(Path(__import__(CoachTracker.__module__, fromlist=['__file__']).__file__)),
                'detector_sha256': sha(DETECTOR), 'pose_sha256': sha(POSE_MODEL),
                'ids_sha256': sha(a.ids_file), 'start_s': a.start_s, 'max_frames': a.max_frames}
    receipt = a.output/'RUN.json'
    if os.environ.get('RTMPOSE_SKIP_RECEIPT') != '1':
        if receipt.exists():
            if json.loads(receipt.read_text()) != identity:
                raise RuntimeError('Cache identity changed; use a new output directory')
        else:
            try:
                write_new(receipt, identity)
            except FileExistsError:
                if json.loads(receipt.read_text()) != identity:
                    raise RuntimeError('Concurrent worker changed cache identity')
    if a.no_lock:
        os.environ['RTMPOSE_LOCK_ALREADY_HELD'] = '1'
    with gpu_locks():
        backend = Backend()
        failures = []
        for index, rid in enumerate(ids, 1):
            if shutil.disk_usage('/mnt/lightsem-data').free < 20*2**30:
                raise RuntimeError('20 GiB free-space floor reached')
            source = Path('/mnt/lightsem-data/justdance')/rows[rid]['file']
            output = a.output/f'{rid}.npz'
            if output.exists():
                with np.load(output, allow_pickle=False) as z:
                    metadata = json.loads(str(z['meta']))
                if metadata['source_sha256'] != sha(source):
                    raise RuntimeError(f'Source hash changed for {rid}')
                continue
            try:
                result = extract(backend, source, start_s=a.start_s, max_frames=a.max_frames, batch_size=a.batch_size)
                save_new(output, result)
                meta = json.loads(str(result['meta']))
                print(json.dumps({'index':index,'total':len(ids),'recording_id':rid,
                                  'frames':meta['n_frames'],'track_segments':meta['n_track_segments'],
                                  'seconds':meta['seconds'],'sha256':sha(output)}), flush=True)
            except Exception as error:
                failure = {'recording_id':rid,'error':f'{type(error).__name__}: {error}'}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                # Runtime failures invalidate the backend, rather than silently
                # marking the rest of the corpus unusable.
                if any(s in str(error).lower() for s in ('cuda', 'cudnn', 'alloc', 'onnxruntime')):
                    raise
        write_new(a.output/f'COMPLETE_{time.time_ns()}.json',
                  {'requested_recordings':len(ids),'failures':failures,'identity':identity})
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
