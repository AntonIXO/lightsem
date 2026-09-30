#!/usr/bin/env python3
"""Authenticate the frozen cohort and prepare a prediction-blind annotation pack.

Only frozen enrolled sources are opened. No model inference, labels or TEST access.
The small still cache uses the exact original observer frame indices and geometry.
Annotation video is silent; its original PTS are published for event conversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from lightsem.model.light_layers.visual import iter_rgb_frames, sha256, write_json

RECORDS = Path('/root/lightsem/experiments/2026-09-29-visual-next/records')
ROOT = Path('/root/lightsem/runs/visual_next_20260929')


def authenticate(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'Frozen input mismatch: {path}: {actual}')
    return {'path': str(path), 'sha256': actual, 'bytes': Path(path).stat().st_size}


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--enrollment', type=Path, default=RECORDS/'ANNOTATION_ENROLLMENT.json')
    ap.add_argument('--output', type=Path, default=ROOT/'annotations')
    args = ap.parse_args()
    started = time.monotonic()
    enrollment = json.loads(args.enrollment.read_text())
    bindings = json.loads((RECORDS/'INPUTS.json').read_text())
    verified = []
    for path, expected in bindings['source_files'].items():
        verified.append(authenticate(path, expected))
    verified.append(authenticate(bindings['training_manifest']['path'], bindings['training_manifest']['sha256']))
    baseline = bindings['baseline']
    verified.append(authenticate(baseline['reference_streams'], baseline['reference_streams_sha256']))
    verified.append(authenticate(baseline['reference_video'], baseline['reference_video_sha256']))
    verified.append(authenticate('/root/shpongle.flac', '95e4388424ef62950bdeae78e47b8e71839ed28c6e01189fe15f9ae6c902b6c0'))
    seen = set()
    for row in enrollment['records']:
        if row['canonical_id'] in seen or row['role'] not in (
                'annotation_calibration_TRAIN', 'historical_DEV_assessment'):
            raise RuntimeError('Duplicate family or unregistered cohort role')
        seen.add(row['canonical_id'])
        for key in ('source_video', 'source_pose', 'capture_receipt'):
            verified.append(authenticate(row[key]['path'], row[key]['sha256']))
        capture = json.loads(Path(row['capture_receipt']['path']).read_text())
        for name, entry in capture['artifacts'].items():
            verified.append(authenticate(Path(row['capture_receipt']['path']).parent/name, entry['sha256']))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'SOURCE_AUTHENTICATION.json', {
        'status':'all_inputs_authenticated', 'enrollment_sha256':sha256(args.enrollment),
        'records':verified, 'families':len(seen), 'test_access':False})
    rows = []
    for number, row in enumerate(enrollment['records']):
        if shutil.disk_usage(ROOT).free < 2*1024**3 or shutil.disk_usage('/mnt').free < 1024**3:
            raise RuntimeError('Resource free-space floor')
        folder = args.output/row['canonical_id']; folder.mkdir()
        capture = json.loads(Path(row['capture_receipt']['path']).read_text())
        sample_path = Path(row['capture_receipt']['path']).parent/'sam2.npz'
        with np.load(sample_path, allow_pickle=False) as z:
            # Read timestamp/index metadata only; no predicted targets enter the pack.
            indices = z['frame_indices'].copy()
            source_times = z['source_times_s'].copy()
        take = np.arange(0, 640, 70)
        mapping = {'frame_indices':indices[take]}
        geometry = capture['geometry']['decode_wh']
        stills = []
        for offset, source_s, rgb in zip(take/10, source_times[take],
                iter_rgb_frames(row['source_video']['path'], mapping, geometry)):
            image = folder/f'frame_{int(offset):02d}.png'
            Image.fromarray(rgb).save(image)
            stills.append({'offset_s':float(offset),'source_time_s':float(source_s),
                           'path':str(image.relative_to(args.output)), 'sha256':sha256(image)})
        video = folder/'source.mp4'
        command = ['ffmpeg','-v','error','-nostdin','-threads','2','-ss',str(row['start_s']),
                   '-i',row['source_video']['path'],'-t',str(row['duration_s']),'-map','0:v:0',
                   '-an','-sn','-dn','-vf',f'scale={geometry[0]}:{geometry[1]}',
                   '-c:v','libx264','-preset','fast','-crf','18','-threads','2',
                   '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video)]
        subprocess.run(command, check=True, timeout=240, capture_output=True)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                          '-show_format','-of','json',str(video)],text=True))
        assert len(probe['streams']) == 1 and probe['streams'][0]['codec_type'] == 'video'
        assert abs(float(probe['format']['duration'])-64) < .1
        rows.append({**{k:row[k] for k in ('canonical_id','recording_id','role','start_s','duration_s')},
            'video':str(video.relative_to(args.output)), 'video_sha256':sha256(video),
            'stills':stills,'geometry':geometry, 'command':command,
            'source_video_sha256':row['source_video']['sha256'],
            'event_time_mapping':'clip time + start_s; native frame quantization <=1 frame',
            'still_time_mapping':'exact original observer frame_indices; source_time_s above'})
        write_json(folder/'MEDIA.json', rows[-1])
        print(f'ANNOTATION_MEDIA {number+1}/{len(enrollment["records"])} {row["canonical_id"]}',flush=True)
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 1024**3:
            raise RuntimeError('Annotation pack exceeds 1 GiB sub-budget')
    write_json(args.output/'PACK.json', {'status':'media_ready_annotations_absent',
        'enrollment_sha256':sha256(args.enrollment), 'elapsed_s':time.monotonic()-started,
        'records':rows,'predictions_included':False,'labels_collected':0,
        'annotation_contract':'Two independent humans, blinded; events and 10 mask frames each; adjudicate before scoring.'})


if __name__ == '__main__':
    run()
