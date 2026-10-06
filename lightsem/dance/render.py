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
    if not np.isfinite(start) or start < 0 or abs(start * 30 - round(start * 30)) > 1e-7:
        raise ValueError('--start must be a nonnegative native30Hz frame boundary')
    duration = song_duration - start if duration is None else float(duration)
    if not np.isfinite(duration) or duration <= 0 or start + duration > song_duration + 1e-9:
        raise ValueError('Requested render interval must be within the original song')
    frames = int(math.ceil(duration * 30 - 1e-9))
    indexes = np.minimum(round(start * 30) + np.arange(frames), len(body.time) - 1)
    times, patch, controls, _ = lighting
    ticks = np.minimum(np.rint((start + np.arange(frames) / 30) * 40).astype(int), len(times) - 1)
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
                        text.text((16, 8), 'Fixed C' + modifiers, font=large, fill='white')
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
