"""Render foundation controls in the native-style two-view stage layout.

This is deliberately a comparison visualizer, not a native DMX render.  It
uses only saved audio→light controls, maps the eight learned fixtures to the
native-style 8-beam/5-laser/50-bulb stage for presentation, and keeps a
neutral off-camera body so no old native body can leak into the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
from lightsem.sim.renderer import RenderConfig
from lightsem.dance.pipeline import accelerator_lock


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def plot_strip(values: dict[str, np.ndarray], duration: float, *, width=1280, height=304):
    strip = Image.new('RGB', (width, height), (12, 18, 27))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    draw.text((10, 4), 'RAW foundation controls | no native body or emitted DMX | no smoothing', font=font, fill='#d1e1ed')
    rows = [('aim_speed', 'Aim speed (/s)', '#4bd6e6'), ('aim_spread', 'Fixture aim spread', '#ffcf67'),
            ('brightness', 'Mean brightness', '#8ee28e'), ('event', 'Event probability', '#e28ee2')]
    for i, (key, label, color) in enumerate(rows):
        arr = np.asarray(values[key], dtype=np.float32)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        if key in ('aim_spread', 'brightness', 'event'):
            lo, hi = 0., max(hi, 1e-9)
        y = 34 + i * 60
        draw.text((10, y), label, font=font, fill='#c1d2e2')
        draw.text((160, y), f'{lo:.3g}..{hi:.3g}', font=font, fill='#9bb5cc')
        x0, x1 = 278, width - 20
        xs = np.linspace(x0, x1, len(arr)).astype(int)
        ys = y + 28 - (arr - lo) / (hi - lo) * 26
        points = [(int(x), int(yv)) for x, yv in zip(xs, ys) if np.isfinite(yv)]
        if len(points) > 1:
            draw.line(points, fill=color, width=1)
    return strip


def directions_from_aim(aim: np.ndarray) -> np.ndarray:
    pan = aim[..., 0] * math.pi
    tilt = aim[..., 1] * (math.pi / 2.)
    x = np.sin(pan) * np.cos(tilt)
    y = -np.cos(pan) * np.cos(tilt)
    z = np.sin(tilt)
    out = np.stack((x, y, z), axis=-1)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--patch', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/native_first_video_20261015_v1/shpongle/full_run/patch.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', default='Shpongle | foundation audio→light | native-style comparison')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with np.load(args.predictions, allow_pickle=False) as z:
        pred = {k: np.asarray(z[k]) for k in ('aim', 'rgb', 'brightness', 'speed', 'event')}
        times = np.asarray(z['time'], dtype=np.float32)
    n = len(times)
    if n < 2 or not np.all(np.isfinite(times)) or not np.allclose(times, np.arange(n) / 30., atol=1e-6):
        raise ValueError('predictions must use a native 30 Hz clock')
    if not all(np.isfinite(v).all() for v in pred.values()):
        raise ValueError('nonfinite foundation control')
    patch = json.loads(args.patch.read_text())
    if len(patch) != 63 or sum(x['kind'] == 'beam' for x in patch) != 8 or sum(x['kind'] == 'laser' for x in patch) != 5 or sum(x['kind'] == 'bulb' for x in patch) != 50:
        raise ValueError('comparison patch must be the 8/5/50 native-style layout')

    # Learned fixture slots drive the eight beams. Lasers and bulbs are an
    # explicit presentation duplication of those slots; they add no new
    # information and are labelled diagnostic-only in the receipt.
    moving_map = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 5, 7], dtype=np.int64)
    bulb_map = np.arange(50, dtype=np.int64) % 8
    dirs8 = directions_from_aim(pred['aim'])
    directions = np.zeros((n, 63, 3), dtype=np.float32)
    directions[:, :13] = dirs8[:, moving_map]
    directions[:, 13:] = np.array([0., -1., 0.], dtype=np.float32)
    rgb = np.zeros((n, 63, 3), dtype=np.float32)
    rgb[:, :13] = pred['rgb'][:, moving_map]
    rgb[:, 13:] = pred['rgb'][:, bulb_map]
    dimmer = np.zeros((n, 63), dtype=np.float32)
    dimmer[:, :13] = pred['brightness'][:, moving_map, 0]
    dimmer[:, 13:] = pred['brightness'][:, bulb_map, 0]
    pair = np.linalg.norm(pred['aim'][:, :, None, :] - pred['aim'][:, None, :, :], axis=-1)
    ii, jj = np.triu_indices(8, 1)
    aim_spread = pair[:, ii, jj].mean(1)
    aim_speed = np.linalg.norm(np.diff(pred['aim'], axis=0), axis=-1).mean(1) * 30.
    event = 1. / (1. + np.exp(-pred['event'][:, 0]))
    np.savez_compressed(args.out.with_suffix('.CONTROLS.npz'), time=times, directions=directions, rgb=rgb,
                        dimmer=dimmer, aim_spread=aim_spread, aim_speed=aim_speed,
                        brightness=pred['brightness'][:, :, 0].mean(1), event=event)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True)
    (args.out / 'frames').mkdir()
    # Keep the mannequin outside both camera frusta. It is not a source body.
    neutral_body = np.full((22, 3), 1000., dtype=np.float32)
    cfgs = [RenderConfig(width=640, height=600, fps=30, cam_orbit=0., haze=.25,
                         intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                         fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                         beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                         require_gpu=True, cam_pos=(2.8, 2.15, 3.8), cam_target=(0., 1., -3.5), camera_scale=.20),
            RenderConfig(width=640, height=600, fps=30, cam_orbit=0., haze=.25,
                         intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                         fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                         beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                         require_gpu=True, cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    output = args.out / 'demo.mp4'
    video_duration = n / 30.
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x1024', '-r', '30', '-i', 'pipe:0', '-i', str(args.audio),
               '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(n), '-c:v', 'libx264',
               '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
               '-t', str(video_duration), '-movflags', '+faststart', str(output)]
    with accelerator_lock():
        renderers = [LitBodyStageRenderer(patch, cfg, body_offset=(0., 0., 0.), body_scale=1., neutral_fill=0.) for cfg in cfgs]
        try:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            for i in range(n):
                views = [renderer.render_controls(directions[i], rgb[i], dimmer[i], neutral_body) for renderer in renderers]
                canvas = Image.new('RGB', (1280, 1024), (7, 10, 16)); draw = ImageDraw.Draw(canvas)
                draw.text((16, 8), args.label, font=font, fill='white')
                draw.text((16, 40), '30 Hz learned controls | duplicated 8→5/50 presentation outputs | no native body/DMX', font=small, fill='#c1d2e2')
                draw.text((16, 65), 'NO BODY INPUT', font=small, fill='#c1d2e2')
                draw.text((656, 65), 'FOUNDATION LIGHT OUTPUT', font=small, fill='#c1d2e2')
                canvas.paste(Image.fromarray(views[0]), (0, 84)); canvas.paste(Image.fromarray(views[1]), (640, 84))
                draw.text((16, 690), f'Sh pongle  t={i/30.:05.2f}s | audio-only inference | display is diagnostic comparison', font=small, fill='#bacedd')
                strip = plot_strip({'aim_speed': aim_speed[:i+1], 'aim_spread': aim_spread[:i+1],
                                    'brightness': pred['brightness'][:i+1, :, 0].mean(1), 'event': event[:i+1]}, video_duration)
                cursor = round(278 + (i / max(n - 1, 1)) * 982)
                ImageDraw.Draw(strip).line((cursor, 30, cursor, 290), fill='#fff2bb', width=2)
                canvas.paste(strip, (0, 720)); encoder.stdin.write(canvas.tobytes())
                if i in (0, n // 2, n - 1): canvas.save(args.out / 'frames' / f'frame_{i:04d}.png')
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('ffmpeg failed')
        finally:
            for renderer in reversed(renderers): renderer.close()
    montage = Image.new('RGB', (640 * 3, 360))
    for i, frame in enumerate([0, n // 2, n - 1]):
        montage.paste(Image.open(args.out / 'frames' / f'frame_{frame:04d}.png').resize((640, 360)), (640 * i, 0))
    montage.save(args.out / 'frames' / 'montage.png')
    receipt = {'schema_version': 1, 'status': 'diagnostic_only', 'song': 'Shpongle', 'duration_s': video_duration,
               'fps': 30, 'video': str(output), 'video_sha256': sha(output), 'predictions': str(args.predictions),
               'predictions_sha256': sha(args.predictions), 'audio': str(args.audio), 'audio_sha256': sha(args.audio),
               'patch': str(args.patch), 'patch_sha256': sha(args.patch), 'learned_beams': 8,
               'presentation_lasers': 5, 'presentation_bulbs': 50, 'presentation_duplication': 'explicit 8→5/50; no new independent information',
               'native_body': False, 'native_dmx': False, 'state_reset_per_chunk': True,
               'collapse_gate': {'min_per_frame_mean_pairwise_aim': float(aim_spread.min()), 'fraction_below_0.2': float(np.mean(aim_spread < .2)), 'passed': bool(np.all(aim_spread >= .2))},
               'frames_inspected': [str(args.out / 'frames' / f'frame_{i:04d}.png') for i in (0, n // 2, n - 1)],
               'montage': str(args.out / 'frames' / 'montage.png'), 'promotion': 'forbidden'}
    (args.out / 'COMPARISON.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
