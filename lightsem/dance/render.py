"""Actual emitted-DMX lit-body rendering of a native CLI run."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .baseline import sha
from .pipeline import accelerator_lock, write_json


def render_video(body, root, audio, output, lighting, *, start=0., duration=None):
    from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
    from lightsem.sim.renderer import RenderConfig
    from scripts.rendering.native_repair_traces import body_traces, motor_speed, pixel_envelope
    import soundfile as sf
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Video already exists; choose another --out')
    source = sf.info(audio); song_duration = source.frames / source.samplerate
    audio_digest = sha(audio)
    body_audio = body.metadata.get('audio', {}).get('audio_sha256')
    if body_audio != audio_digest:
        raise ValueError('saved body belongs to a different waveform')
    generation_path, lighting_path = root / 'GENERATION.json', root / 'LIGHTING.json'
    if not generation_path.is_file() or not lighting_path.is_file():
        raise ValueError('render requires generation and lighting provenance reports')
    generation = json.loads(generation_path.read_text())
    lighting_report = json.loads(lighting_path.read_text())
    if generation.get('status') in {'failure_diagnostic', 'unqualified'}:
        raise ValueError(f"refusing to render generation with status {generation.get('status')}")
    if generation.get('body_sha256') != sha(root / 'body.npz') or lighting_report.get('body_sha256') != sha(root / 'body.npz'):
        raise ValueError('saved body hash does not match generation/lighting provenance')
    if lighting_report.get('lights_sha256') != sha(root / 'lights.npz'):
        raise ValueError('saved lights hash does not match lighting provenance')
    if lighting_report.get('patch_sha256') != sha(root / 'patch.json'):
        raise ValueError('saved fixture patch hash does not match lighting provenance')
    if not np.isfinite(start) or start < 0 or abs(start * 30 - round(start * 30)) > 1e-7:
        raise ValueError('--start must be a nonnegative native30Hz frame boundary')
    duration = song_duration - start if duration is None else float(duration)
    if not np.isfinite(duration) or duration <= 0 or start + duration > song_duration + 1e-9:
        raise ValueError('Requested render interval must be within the original song')
    frames = int(math.ceil(duration * 30 - 1e-9))
    world = np.asarray(body.world)
    body_time = np.asarray(body.time)
    if world.ndim != 3 or world.shape[0] != len(body_time) or world.shape[1:] != (22, 3):
        raise ValueError(f'body must have world shape (T, 22, 3) matching time; got {world.shape}')
    if not np.all(np.isfinite(body_time)) or not np.all(np.isfinite(world)) or len(body_time) < 2 or np.any(np.diff(body_time) <= 0):
        raise ValueError('body time/world must be finite with strictly increasing time')
    expected_body_time = np.arange(len(body_time), dtype=np.float64) / 30.
    if not np.allclose(body_time, expected_body_time, rtol=0., atol=1e-8):
        raise ValueError('saved body must use the exact zero-based native 30 Hz clock')
    first_index = round(start * 30)
    last_index = first_index + frames - 1
    if first_index < 0 or last_index >= len(body_time) or body_time[first_index] > start + 1e-6 or body_time[last_index] < start + (frames - 1) / 30 - 1e-6:
        raise ValueError('requested render interval exceeds saved native body coverage')
    indexes = first_index + np.arange(frames)
    times, patch, controls, _ = lighting
    times = np.asarray(times)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('lighting times must be finite and strictly increasing')
    if not np.allclose(times, np.arange(len(times), dtype=np.float64) / 40., rtol=0., atol=1e-8):
        raise ValueError('emitted lights must use the exact zero-based 40 Hz clock')
    if not (0 <= np.rint((start + np.arange(frames) / 30) * 40).astype(int)[-1] < len(times)):
        raise ValueError('requested render interval exceeds emitted lighting coverage')
    ticks = np.rint((start + np.arange(frames) / 30) * 40).astype(int)
    if not all(k in controls for k in ('directions', 'rgb', 'dimmer')):
        raise ValueError('lighting controls are missing required emitted DMX streams')
    fixture_count = len(patch)
    if not isinstance(patch, (list, tuple)) or fixture_count < 1:
        raise ValueError('fixture patch must be a non-empty list')
    moving_count = sum(f.get('kind') != 'bulb' for f in patch)
    bulb_count = sum(f.get('kind') == 'bulb' for f in patch)
    if moving_count > 16 or bulb_count > 128:
        raise ValueError('fixture count exceeds the lit-body renderer limit (16 moving, 128 bulbs)')
    for key in ('directions', 'rgb', 'dimmer'):
        arr = np.asarray(controls[key])
        expected_shape = (len(times), fixture_count, 3) if key in ('directions', 'rgb') else (len(times), fixture_count)
        if arr.shape != expected_shape or not np.all(np.isfinite(arr)):
            raise ValueError(f'lighting control {key} has invalid shape or non-finite values')
        if key == 'rgb' and np.any((arr < 0) | (arr > 1)):
            raise ValueError('emitted RGB values must be within [0, 1]')
        if key == 'dimmer' and np.any((arr < 0) | (arr > 1)):
            raise ValueError('emitted dimmer values must be within [0, 1]')
    raw = body_traces(body.world, body.time)
    movers = np.array([f['kind'] != 'bulb' for f in patch]); beams = np.array([f['kind'] == 'beam' for f in patch])
    rows = [(key, raw[key], label) for key, label in [('left_arm', 'Left arm (m/s)'), ('right_arm', 'Right arm (m/s)'),
                                                   ('lower', 'Lower body (m/s)'), ('body_jerk', 'Raw world jerk (m/s³)'),
                                                   ('root_y', 'Root Y (m)')]]
    if movers.any():
        rows.append(('emitted', motor_speed(controls['angles_deg'][:, movers], times), 'Emitted motors (°/s)'))
    if beams.any():
        rows.append(('dimmer', (times, controls['dimmer'][:, beams].mean(1)), 'Actual beam dimmer'))
    with np.load(root / 'plan.npz', allow_pickle=False) as planned:
        rows.append(('rate', (planned['time'], planned['effort'].mean(1)), 'Recorded modifier rate (×)'))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    strip = Image.new('RGB', (1280, 304), (12, 18, 27)); draw = ImageDraw.Draw(strip)
    draw.text((10, 4), 'RAW saved body / actual emitted DMX | per-pixel min/max retains peaks | no smoothing', font=font, fill='#d1e1ed')
    exports = {}
    for i, (key, (time, values), label) in enumerate(rows):
        mask = (time >= start) & (time <= start + duration); selected = values[mask]
        if not len(selected):
            raise ValueError('No samples in trace interval')
        lo, hi = float(selected.min()), float(selected.max())
        if key == 'root_y':
            margin = max((hi - lo) * .05, .005); lo -= margin; hi += margin
        else:
            lo = 0.; hi = max(hi, 1e-8)
        y = 32 + i * 30
        draw.text((10, y), label, font=font, fill='#c1d2e2')
        draw.text((215, y), f'{lo:.2g}..{hi:.2g}', font=font, fill='#9bb5cc')
        low, high = pixel_envelope(time, values, left=start, right=start + duration, pixels=996)
        for x in np.flatnonzero(np.isfinite(low)):
            draw.line((278 + int(x), y + 24 - (low[x] - lo) / (hi - lo) * 22,
                       278 + int(x), y + 24 - (high[x] - lo) / (hi - lo) * 22), fill='#4bd6e6')
        exports[key + '_time'] = time[mask]; exports[key + '_raw'] = selected
    with (root / 'RAW_TRACE.npz').open('xb') as target:
        np.savez_compressed(target, **exports)
    video_duration = frames / 30
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x1024', '-r', '30', '-i', 'pipe:0', '-ss', str(start), '-i', str(audio),
               '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(frames), '-c:v', 'libx264',
               '-preset', 'veryfast', '-threads', '2', '-b:v', '1800k', '-pix_fmt', 'yuv420p',
               '-af', f'apad=whole_dur={video_duration},atrim=end={video_duration},asetpts=PTS-STARTPTS',
               '-c:a', 'aac', '-b:a', '128k', '-t', str(video_duration), '-movflags', '+faststart', str(output)]
    cameras = [dict(cam_pos=(2.8, 2.15, 3.8), cam_target=(0., 1., -3.5), camera_scale=.20),
               dict(cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)]
    renderers, configs = [], []
    with accelerator_lock():
        try:
            for camera, fill in zip(cameras, (.50, .12)):
                cfg = RenderConfig(width=640, height=600, fps=30, cam_orbit=0., haze=.25,
                                   intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                                   fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                                   beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                                   require_gpu=True, **camera)
                renderers.append(LitBodyStageRenderer(patch, cfg, body_offset=(0., 0., -3.5), body_scale=1., neutral_fill=fill))
                configs.append(asdict(cfg))
            # Actual source-body visibility and frozen-pixel repeat before encoding.
            pose = body.world[indexes[0]].copy(); pose -= pose[:1] * np.array([1., 0., 1.])
            test = tuple(controls[key][ticks[0]] for key in ('directions', 'rgb', 'dimmer'))
            for renderer in renderers:
                diagnostic = renderer.render_diagnostics(*test, pose)
                if not diagnostic['metrics']['body_visible_pixels'] or not np.array_equal(diagnostic['actual'], renderer.render_controls(*test, pose)):
                    raise ValueError('Actual body/DMX smoke check failed')
            with (root / 'encode.log').open('xb') as log:
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                try:
                    for frame, (index, tick) in enumerate(zip(indexes, ticks)):
                        pose = body.world[index].copy(); pose -= pose[:1] * np.array([1., 0., 1.])
                        lighting_values = tuple(controls[key][tick] for key in ('directions', 'rgb', 'dimmer'))
                        canvas = Image.new('RGB', (1280, 1024), (7, 10, 16)); text = ImageDraw.Draw(canvas)
                        modifiers = (' + D4-max' if body.metadata['d4max_enabled'] else '') + (' + D6' if body.metadata['d6_enabled'] else '')
                        display_title = str(body.metadata.get('display_title', 'Fixed C'))
                        text.text((16, 8), display_title + modifiers, font=large, fill='white')
                        text.text((16, 40), 'P2 | historical brightness | actual emitted DMX | fixture motion limits OFF', font=font, fill='#c1d2e2')
                        text.text((16, 65), 'OWN-SONG 3D BODY', font=font, fill='#c1d2e2')
                        text.text((656, 65), 'ACTUAL EMITTED LIGHTS', font=font, fill='#c1d2e2')
                        for x, renderer in zip((0, 640), renderers):
                            canvas.paste(Image.fromarray(renderer.render_controls(*lighting_values, pose)), (x, 84))
                        text.text((16, 690), f'Source {start + frame / 30:.2f}s | original audio speed | rootXZ view transform; neutral fill .50/.12 separate from DMX', font=font, fill='#bacedd')
                        picture = strip.copy(); line = ImageDraw.Draw(picture)
                        cursor = round(278 + (frame / 30) / duration * 995)
                        line.line((cursor, 30, cursor, 290), fill='#fff2bb', width=2)
                        canvas.paste(picture, (0, 720)); encoder.stdin.write(canvas.tobytes())
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError('Video encode failed; partial output retained')
                except BaseException:
                    if encoder.poll() is None:
                        encoder.kill(); encoder.wait()
                    raise
        finally:
            for renderer in reversed(renderers):
                renderer.close()
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-i', str(output), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(output)]))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    if int(stream['nb_read_frames']) != frames or stream['r_frame_rate'] != '30/1':
        raise ValueError('Rendered frame count or clock mismatch')
    write_json(root / 'RENDER.json', {'path': str(output), 'sha256': sha(output), 'frames': frames,
                                     'source_start_s': start, 'source_duration_s': duration,
                                     'video_duration_s': video_duration, 'audio_speed': 1.,
                                     'full_software_decode': True, 'body_sha256': sha(root / 'body.npz'),
                                     'lights_sha256': sha(root / 'lights.npz'), 'renderer_configs': configs,
                                     'posthoc_modifiers': {'d4max': body.metadata['d4max_enabled'], 'd6': body.metadata['d6_enabled']}})
    return output
