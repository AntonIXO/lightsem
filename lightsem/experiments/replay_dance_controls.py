"""Replay authenticated cached dance predictions through the repaired rig map.

No model inference or training. Historical predictions and videos stay intact.
All raster rendering runs on NVIDIA GPU; shader animation stays disabled.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .stage11_common import authenticated_json, read_authenticated, digest, verify_artifacts, artifact_manifest, gpu_lock
from .dance_no_clock_render import controls_to_dmx, render_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    previous = read_authenticated(root/'previews/shpongle/report.json')
    verify_artifacts(previous['artifacts'])
    audio = Path(previous['demo_config']['audio_path'])
    if digest(audio) != previous['demo_config']['audio_sha256']:
        raise RuntimeError('Source audio changed')
    source = root/'previews/shpongle/learned_motion.npz'
    with np.load(source, allow_pickle=False) as saved:
        controls = saved['standardized'].copy()
        names = saved['target_names'].tolist()
    duration = previous['demo_config']['duration_s']
    frames, trace, mapping = controls_to_dmx(controls, names, duration)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    dmx = out/'body_motion_dmx.npy'
    np.save(dmx, frames)
    trace_path = out/'body_motion_trace.npz'
    np.savez_compressed(trace_path, **trace)
    print(json.dumps({'stage':'mapping_ready','mapping':mapping,'prediction_sha256':digest(source)}), flush=True)
    with gpu_lock(root):
        video = out/'shpongle_body_motion_v2.mp4'
        render = render_gpu(frames, audio, video, duration,
                            on_progress=lambda value: print(json.dumps(value), flush=True))
    report = {'status':'completed','previous_report_sha256':digest(root/'previews/shpongle/report.json'),
              'prediction_path':str(source),'prediction_sha256':digest(source),
              'checkpoint_sha256':previous['model_checkpoint_sha256'],
              'audio_path':str(audio),'audio_sha256':digest(audio),
              'new_training':False,'new_model_inference':False,
              'mapping':mapping,'render':render,'video':str(video),
              'mapper_source_sha256':digest(Path(__file__).with_name('dance_no_clock_render.py')),
              'replay_source_sha256':digest(__file__),
              'artifacts':artifact_manifest([dmx,trace_path,video])}
    authenticated_json(out/'report.json', report)
    print(json.dumps({'status':'completed','video':str(video),'render_seconds':render['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
