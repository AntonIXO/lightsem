"""Continuous learned dance controls to DMX, with no temporal effects engine.

The inputs are standardized pose-derived predictions, never beats or audio.
Every fixture value is a pointwise transform of those predictions, followed
only by interpolation and declared actuator speed limits. Constant controls
produce constant DMX. The GPU renderer is also held at a fixed shader time.
This is an explicit rig adapter, not a fixture trajectory head trained on DMX.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.encode import pack_universe


REQUIRED = (
    'com_speed', 'vert_bounce_amp', 'qom', 'head_speed', 'foot_speed',
    'hand_speed', 'hand_spread', 'lateral_pos', 'laterality', 'spread',
    'section_energy', 'vert_bounce', 'weight',
)


@dataclass(frozen=True)
class DanceRigConfig:
    model_fps: float = 40.
    dmx_fps: float = 40.
    pan_span_deg: float = 180.
    tilt_span_deg: float = 30.
    hand_fan_deg: float = 18.
    asymmetry_tilt_deg: float = 12.
    pan_speed_deg_s: float = 180.
    tilt_speed_deg_s: float = 120.
    spatial_sigma: float = .42
    spatial_floor: float = .3
    colour_speed_turns_s: float = .45


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _level(x):
    """Training-standardized movement to expressive 0..1 control range."""
    return .5 + .5 * np.tanh(np.asarray(x, dtype=float))


def _rate_limit(values, max_step):
    out = np.asarray(values, dtype=float).copy()
    for i in range(1, len(out)):
        out[i] = out[i-1] + np.clip(out[i]-out[i-1], -max_step, max_step)
    return out


def _angles(vectors):
    length = np.linalg.norm(vectors, axis=-1)
    if np.any(length <= 0):
        raise ValueError('Cannot aim at the fixture itself')
    return (np.degrees(np.arctan2(vectors[..., 0], -vectors[..., 1])),
            np.degrees(np.arcsin(np.clip(vectors[..., 2] / length, -1, 1))))


def _limited_ticks(angles, angle_range, speed, fps):
    units = .5 + np.asarray(angles) / angle_range
    if not np.isfinite(units).all() or np.any((units < 0) | (units > 1)):
        raise ValueError('Aiming outside fixture range')
    targets = np.rint(units * 65535).astype(np.int64)
    step = math.floor(speed * 65535 / (angle_range * fps))
    if step < 1:
        raise ValueError('Speed limit smaller than one DMX tick')
    result = targets.copy()
    for i in range(1, len(result)):
        result[i] = result[i-1] + np.clip(targets[i]-result[i-1], -step, step)
    return result


def controls_to_dmx(standardized_controls, target_names, duration_s, *, rig=None, config=None, event_scores=None):
    """Return DMX, model-to-device traces and a description of the fixed map.

    `standardized_controls` uses the model's TRAIN-only target transform. No
    track statistics, thresholds, event counts, clocks or RNG enter this map.
    """
    cfg = config or DanceRigConfig()
    rig = rig or canonical_rig()
    z = np.asarray(standardized_controls, dtype=np.float64)
    names = list(target_names)
    if (z.ndim != 2 or not len(z) or z.shape[1] != len(names)
            or len(set(names)) != len(names) or not np.isfinite(z).all()):
        raise ValueError('Controls must be a finite aligned matrix with unique names')
    if any(name not in names for name in REQUIRED):
        raise ValueError('Missing learned control channel')
    numeric = [duration_s, cfg.model_fps, cfg.dmx_fps, cfg.pan_speed_deg_s, cfg.tilt_speed_deg_s]
    if not np.isfinite(numeric).all() or min(numeric) <= 0:
        raise ValueError('Duration, sample rates and actuator limits must be positive')
    if abs(len(z) / cfg.model_fps - duration_s) > 1 / cfg.model_fps + 1e-8:
        raise ValueError('Control duration differs from audio duration')
    count = int(round(duration_s * cfg.dmx_fps))
    if count < 1:
        raise ValueError('Duration does not cover a DMX frame')
    source_times = np.arange(len(z)) / cfg.model_fps
    times = np.arange(count) / cfg.dmx_fps
    c = {name: np.interp(times, source_times, z[:, names.index(name)]) for name in REQUIRED}
    lateral = -np.tanh(1.2*c['lateral_pos'])
    lateral_balance = np.tanh(c['laterality'])
    bounce = np.tanh(1.2*c['vert_bounce'])
    spread = _level(c['spread'])
    hand_spread = _level(c['hand_spread'])
    body = np.clip(.05 + .95*(.6*_level(c['qom']) + .4*_level(c['weight'])), 0, 1)
    depth = .68*_level(c['vert_bounce_amp'])
    # This follows predicted body displacement, not an oscillator or beat phase.
    bounce_level = body * (1-depth + depth*(.5-.5*bounce))
    # Accents directly expose positive learned movement excursions. There is
    # no peak quota, event scheduler, synthetic decay envelope or hardware strobe.
    foot_accent = np.maximum(np.tanh(1.4*c['foot_speed']), 0.)
    hand_accent = np.maximum(np.tanh(1.4*c['hand_speed']), 0.)
    if event_scores is not None:
        events = np.asarray(event_scores, float)
        if events.shape != (len(z), 3) or not np.isfinite(events).all() or np.any((events < 0) | (events > 1)):
            raise ValueError('Expected aligned footfall/jump/hit scores in [0,1]')
        events = np.stack([np.interp(times, source_times, events[:,j]) for j in range(3)], 1)
        # Weighted-BCE balanced score > .5 is positive evidence versus the
        # TRAIN event prior. No per-song percentile or event quota.
        accents = np.maximum(2*events-1, 0.)
        foot_accent = np.maximum(accents[:,0], accents[:,1])
        hand_accent = accents[:,2]
    section = _level(c['section_energy'])
    hue = .55 + .34*np.tanh(1.1*c['section_energy'] + .3*c['laterality'])
    hue = _rate_limit(hue, cfg.colour_speed_turns_s/cfg.dmx_fps)
    nf = len(rig.fixtures)
    params = {key: np.zeros((count, nf)) for key in
              ('dimmer', 'strobe', 'hue', 'sat', 'val', 'pan', 'tilt', 'zoom', 'gobo')}
    params['val'][:] = 1.
    params['hue'][:] = hue[:, None]
    params['sat'][:] = (.78 + .18*_level(c['qom']))[:, None]
    params['pan'][:] = .5
    params['tilt'][:] = .5
    params['zoom'][:] = .5
    xs = np.asarray([f.position[0] for f in rig.fixtures], dtype=float)
    xs /= max(float(np.abs(xs).max()), 1e-6)
    steering, head_audits = [], []
    for index, fixture in enumerate(rig.fixtures):
        side = xs[index]
        distance = side-lateral
        balance = cfg.spatial_floor + (1-cfg.spatial_floor)*np.exp(-distance**2/(2*cfg.spatial_sigma**2))
        if fixture.kind == 'moving_head':
            params['dimmer'][:, index] = (.24+.76*bounce_level) * (.5+.5*balance)
            params['sat'][:, index] = .58+.16*_level(c['qom'])
            params['zoom'][:, index] = .18 + .50*spread
            # Aim at world coordinates below the truss. In the renderer's
            # convention pan rotates x/y, so blindly using +/-180 aims UP.
            # Fixture-specific anchors retain spatial breadth across the rig.
            target = np.column_stack((
                np.clip(fixture.position[0]*(.72+.20*(spread-.5)) + 2.8*lateral
                        + .45*side*(2*hand_spread-1), -5.4, 5.4),
                1.7+1.4*bounce+.15*side*lateral_balance,
                -3.2+1.5*bounce))
            pan, tilt = _angles(target-np.asarray(fixture.position))
            params['hue'][:, index] += .04*side*lateral_balance
            p = _limited_ticks(pan, fixture.pan_range_deg, cfg.pan_speed_deg_s, cfg.dmx_fps)
            t = _limited_ticks(tilt, fixture.tilt_range_deg, cfg.tilt_speed_deg_s, cfg.dmx_fps)
            params['pan'][:, index] = p / 65535
            params['tilt'][:, index] = t / 65535
            steering.append((fixture, p, t))
            head_audits.append({'fixture': fixture.name,
                'pan_speed_deg_s': float(np.max(np.abs(np.diff(p)), initial=0) * fixture.pan_range_deg * cfg.dmx_fps / 65535),
                'tilt_speed_deg_s': float(np.max(np.abs(np.diff(t)), initial=0) * fixture.tilt_range_deg * cfg.dmx_fps / 65535),
                'pan_range_deg': [float((p.min()/65535-.5)*fixture.pan_range_deg), float((p.max()/65535-.5)*fixture.pan_range_deg)],
                'tilt_range_deg': [float((t.min()/65535-.5)*fixture.tilt_range_deg), float((t.max()/65535-.5)*fixture.tilt_range_deg)]})
        elif fixture.kind == 'wash':
            params['dimmer'][:, index] = (.06 + .88*(.55*_level(c['head_speed'])+.45*section)*body)*(.55+.45*balance)
            params['zoom'][:, index] = .35+.6*spread
            params['hue'][:, index] += .06
        elif fixture.kind == 'blinder':
            params['dimmer'][:, index] = .9*foot_accent*balance
            params['sat'][:, index] = 0.
        elif fixture.kind == 'strobe':
            # The fixture runs as a continuous flood. Its flash-rate channel stays zero.
            params['dimmer'][:, index] = hand_accent*(.35+.65*_level(c['hand_speed']))
            params['sat'][:, index] = 0.
        elif fixture.kind == 'projector':
            params['dimmer'][:, index] = .08+.3*section*body
            params['zoom'][:, index] = .15+.4*spread
            # Move the projection across the back wall from learned body
            # position/bounce. All targets stay below the projector, avoiding
            # the pan-angle discontinuity at its vertical pole.
            target = np.column_stack((3.5*lateral, 1.65+.7*bounce, np.full(count, -6.)))
            pan, tilt = _angles(target-np.asarray(fixture.position))
            params['pan'][:, index] = .5 + pan / fixture.pan_range_deg
            params['tilt'][:, index] = .5 + tilt / fixture.tilt_range_deg
        else:
            raise ValueError('Unsupported fixture kind: ' + fixture.kind)
    frames = pack_universe(rig, params)
    # Exact integer writes preserve the declared angular speed cap through packing.
    for fixture, pan, tilt in steering:
        start = fixture.address - 1
        frames[:, start] = (pan >> 8).astype(np.uint8)
        frames[:, start+1] = (pan & 255).astype(np.uint8)
        frames[:, start+2] = (tilt >> 8).astype(np.uint8)
        frames[:, start+3] = (tilt & 255).astype(np.uint8)
    trace = {'dimmer': params['dimmer'], 'zoom': params['zoom'], 'body': body,
             'lateral': lateral, 'bounce': bounce, 'hue': hue,
             'foot_accent': foot_accent, 'hand_accent': hand_accent}
    report = {'mapping': 'world_targets_to_rig_v3', 'config': asdict(cfg),
              'input_channels_used': list(REQUIRED), 'target_names': names,
              'independent_temporal_generators': False, 'strobe_rate_all_zero': True,
              'per_track_scaling': False, 'heads': head_audits,
              'colour_policy': 'palette position follows predicted section energy and laterality; no time-driven colour cycle',
              'accent_policy': 'positive evidence from learned B event scores' if event_scores is not None else 'pose speed reference excursions',
              'aiming_policy': 'body-derived world targets below truss, fixture anchors preserve lateral breadth',
              'interpolation': 'linear_model_grid_to40Hz', 'actuator_filter': 'encoded angular speed cap only',
              'trained_fixture_head': False}
    return frames, trace, report


def isolate_heads(frames, rig=None):
    """Role-isolated audit: other fixtures cannot satisfy beam-motion gates."""
    result = np.asarray(frames).copy()
    for fixture in (rig or canonical_rig()).fixtures:
        if fixture.kind != 'moving_head':
            start = fixture.address-1
            result[:, start:start+fixture.n_channels] = 0
    return result


def render_gpu(frames, audio_path, video_path, duration_s, *, rig=None, dmx_fps=40., on_progress=None):
    """Render a full saved DMX sequence; caller owns the exclusive GPU lock."""
    from lightsem.sim.renderer import RenderConfig, StageRenderer
    frames = np.asarray(frames)
    if frames.ndim != 2 or frames.shape[1] != 512 or frames.dtype != np.uint8 or len(frames) < 1:
        raise ValueError('Expected uint8(T,512) DMX')
    out = Path(video_path)
    if out.exists():
        raise FileExistsError(out)
    cfg = RenderConfig(width=854, height=480, fps=30., haze=.25,
                       cam_pos=(0., 2.6, 11.), cam_target=(0., 3.2, -3.),
                       cam_orbit=0., intensity_gain=2.4)
    renderer = StageRenderer(rig or canonical_rig(), cfg)
    if 'nvidia' not in renderer.renderer_name.lower() or 'llvmpipe' in renderer.renderer_name.lower():
        raise RuntimeError('An NVIDIA GPU renderer is required')
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{cfg.width}x{cfg.height}', '-r', str(cfg.fps), '-i', 'pipe:0', '-i', str(audio_path),
               '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-t', str(duration_s),
               '-shortest', str(out)]
    begun = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total = math.ceil(duration_s * cfg.fps)
    try:
        for index in range(total):
            src = min(round(index * dmx_fps / cfg.fps), len(frames)-1)
            # Freeze camera, projector pattern and haze animation. Only DMX can change the light.
            frame = renderer.render_frame(frames[src], 0.)
            process.stdin.write(frame.tobytes())
            if on_progress and index % 600 == 0:
                on_progress({'frame': index, 'total_frames': total})
    finally:
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        result = process.wait()
    if result:
        raise RuntimeError('Video encoding failed: ' + error)
    media = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(out)], text=True))
    return {'renderer': renderer.renderer_name, 'render_config': asdict(cfg), 'shader_time': 0.,
            'elapsed_seconds': time.monotonic()-begun, 'media': media, 'ffmpeg_argv': command}
