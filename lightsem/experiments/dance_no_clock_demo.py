"""Apply the one frozen dance model to user music and render on NVIDIA GPU."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .stage11_common import (artifact_manifest, authenticated_json, digest, gpu_lock,
                             json_digest, read_authenticated, verify_artifacts)
from .dance_no_clock_training import context, generate_standardized, load_model, noise_for
from .dance_no_clock_render import controls_to_dmx, render_gpu


def run(root):
    root = Path(root)
    protocol, manifest, norms, names, identity = context(root)
    cfg = protocol['demo']
    for path_key, hash_key in [('audio_path', 'audio_sha256'), ('features_path', 'features_sha256')]:
        if digest(cfg[path_key]) != cfg[hash_key]:
            raise RuntimeError('Frozen demo input changed: ' + path_key)
    out = root/'previews/shpongle'
    if (out/'report.json').exists():
        report = read_authenticated(out/'report.json')
        if report['identity'] != identity or report['demo_config'] != cfg:
            raise RuntimeError('Existing demo uses different inputs')
        verify_artifacts(report['artifacts'])
        return report
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Partial demo exists; preserve it before explicit recovery')
    with np.load(cfg['features_path'], allow_pickle=False) as data:
        features = data['x'].copy()
        feature_names = data['names'].tolist()
        metadata = json.loads(str(data['meta']))
    if feature_names != manifest['feature_names'] or json_digest(metadata['preprocessing']) != json_digest(manifest['preprocessing']):
        raise RuntimeError('Inference feature schema/filter differs from training')
    if abs(metadata['source_duration_s'] - cfg['duration_s']) > .01:
        raise RuntimeError('Feature/audio durations differ')
    started = time.monotonic()
    model, blob = load_model(root, 'temporal', 0)
    noise = noise_for(len(features), 101)
    generation_started = time.monotonic()
    controls = generate_standardized(model, features, norms, noise=noise, steps=8, chunk_size=600)
    inference_seconds = time.monotonic() - generation_started
    frames, trace, mapping = controls_to_dmx(controls, names, cfg['duration_s'])
    out.mkdir(parents=True, exist_ok=True)
    generated_path = out/'learned_motion.npz'
    np.savez_compressed(generated_path, standardized=controls, explicit_noise=noise.numpy(),
                        target_names=np.asarray(names), **trace)
    dmx_path = out/'light_dmx.npy'
    np.save(dmx_path, frames)
    checkpoint_sha = digest(root/'runs/temporal/seed_0/final.pt')
    del model, blob
    gc.collect()
    torch.cuda.empty_cache()
    video = out/'shpongle_dance_no_clock.mp4'
    render = render_gpu(frames, cfg['audio_path'], video, cfg['duration_s'],
                        on_progress=lambda value: print(json.dumps({'stage':'render', **value}), flush=True))
    streams = render['media']['streams']
    videos = [s for s in streams if s['codec_type'] == 'video']
    audios = [s for s in streams if s['codec_type'] == 'audio']
    if (len(videos) != 1 or len(audios) != 1 or videos[0]['disposition'].get('attached_pic')
            or audios[0]['channels'] != 2 or abs(float(render['media']['format']['duration']) - cfg['duration_s']) > .1):
        raise RuntimeError('Unexpected output video/audio streams')
    report = {'status':'completed', 'identity':identity, 'demo_config':cfg,
              'model_checkpoint_sha256':checkpoint_sha, 'training_seed':0, 'generation_seed':101,
              'inference_seconds':inference_seconds, 'total_demo_seconds':time.monotonic()-started,
              'input_is_audio_only':True, 'new_training_in_demo':False,
              'algorithmic_beat_or_temporal_effect_generator':False, 'mapping':mapping,
              'motion_output_std':controls.std(0).tolist(), 'render':render, 'video':str(video),
              'validation_status':'development prototype; musical fit requires evaluation',
              'artifacts':artifact_manifest([generated_path, dmx_path, video])}
    authenticated_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with gpu_lock(args.root):
        report = run(args.root)
    print(json.dumps({'status':report['status'], 'video':report['video'],
                      'inference_seconds':report['inference_seconds'],
                      'render_seconds':report['render']['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
