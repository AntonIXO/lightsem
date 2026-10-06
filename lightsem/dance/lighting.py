"""Fixed P2 colors and historical brightness applied to the generated body."""
from pathlib import Path
import json
import os
import subprocess
import numpy as np

from .baseline import REPO, checked_asset, sha
from .pipeline import cached_track, write_json


def prepare_colors(audio, root, config, *, color_path=None, audio_python=None, audio_device=None):
    _, cached = cached_track(config, audio)
    if color_path is not None:
        color_path = Path(color_path).resolve()
        report = json.loads((color_path.parent / 'REPORT.json').read_text())
        if report.get('audio_sha256') != sha(audio):
            raise ValueError('P2 colors belong to a different waveform')
    elif cached is not None:
        color_path = checked_asset(cached['colors'])
        checked_asset(cached['color_report'])
    else:
        command = [audio_python or config['extraction']['python'], '-m', 'lightsem.dance.p2',
                   '--wav', str(audio), '--out', str(root / 'colors'),
                   '--device', audio_device or config['extraction']['device']]
        subprocess.run(command, cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO)}, check=True)
        color_path = root / 'colors/P2.npz'
    with np.load(color_path, allow_pickle=False) as data:
        left, right = data['left'].copy(), data['right'].copy()
        palette = data['palette_ids40'].copy()
    return left, right, palette, {'path': str(color_path), 'sha256': sha(color_path)}


def map_lights(body, audio, root, config, *, counts=None, color_path=None,
               audio_python=None, audio_device=None):
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
    from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
    from lightsem.experiments.motion_study.audio_dynamics import loudness
    from lightsem.experiments.dynamic_dance.retiming import resample_body
    from lightsem.conductor.body_transfer_v2 import make_outputs
    from lightsem.conductor.body_transfer import decode
    import soundfile as sf
    color_config = REPO / config['lighting']['color_config']
    if sha(color_config) != config['lighting']['color_config_sha256']:
        raise ValueError('Accepted P2 color configuration changed')
    _, cached = cached_track(config, audio)
    if cached:
        receipt = checked_asset(cached['loudness_receipt'])
    else:
        loudness(audio, root / 'loudness')
        receipt = root / 'loudness/loudness.json'
    levels, level_identity = load_verified_loudness(receipt, audio)
    left, right, palette, color_identity = prepare_colors(audio, root, config, color_path=color_path,
                                                         audio_python=audio_python, audio_device=audio_device)
    info = sf.info(audio); duration = info.frames / info.samplerate
    frames = int(np.ceil(duration * 40 - 1e-9)); time = np.arange(frames) / 40
    if left.shape != (frames, 3) or right.shape != left.shape:
        raise ValueError('P2 colors must cover the full original40Hz song')
    world40 = resample_body(body.world, np.minimum(time, body.time[-1]), 30.)
    counts = counts or config['lighting']['counts']
    if any(not isinstance(v, int) or v < 0 for v in counts.values()) or sum(counts.values()) < 1:
        raise ValueError('Output counts must be nonnegative with at least one fixture')
    outputs = make_outputs(**counts)
    universes, patch, intent, brightness, hard = compile_world(world40, time, outputs, left, right,
                                                             levels, config['lighting']['transfer'])
    controls = decode(universes, patch)
    if not all(np.isfinite(value).all() for value in controls.values()):
        raise ValueError('Nonfinite actual emitted DMX')
    with (root / 'lights.npz').open('xb') as stream:
        np.savez_compressed(stream, time=time, body=world40, universes=universes,
                            targets=intent['targets'], dimmer=intent['dimmer'], palette_ids40=palette)
    write_json(root / 'patch.json', patch)
    report = {'mapper': config['lighting']['mapper'], 'color': color_identity,
              'loudness': level_identity, 'counts': counts, 'fixture_motion_limits': False,
              'hard_checks': hard, 'lights_sha256': sha(root / 'lights.npz'),
              'body_sha256': sha(root / 'body.npz'),
              'brightness_scope': 'Historical full-song LUFS law with established body modulation',
              'palette_occupancy_ticks': {str(int(k)): int(v) for k, v in zip(*np.unique(palette, return_counts=True))},
              'actual_rgb_transition_ticks': int(np.any(np.diff(controls['rgb'], axis=0), axis=(1, 2)).sum()),
              'emitted_dimmer_min': float(controls['dimmer'].min()),
              'emitted_dimmer_max': float(controls['dimmer'].max()),
              'finite': True, 'no_autonomous_motion_source': True}
    write_json(root / 'LIGHTING.json', report)
    return time, patch, controls, report
