"""Render the existing two-song body-transfer timelines in the old dark stage."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.conductor.body_transfer import decode
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.sim.body_stage_renderer import BodyStageRenderer
from scripts.motion.run_rhythm_companion import render_config, encode_command, audit_movie
from scripts.rendering.render_body_transfer import PARENTS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def check(plan):
    for path, expected in plan['files'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if min(shutil.disk_usage('/root').free, shutil.disk_usage('/mnt').free) < 2**30:
        raise RuntimeError('Disk reserve')
    root = Path(plan['output'])
    if root.exists() and sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > plan['max_bytes']:
        raise RuntimeError('Output storage cap')


def config():
    cfg = render_config()
    # Same viewpoint, wider lens to include x=+-8m garland at z=3m.
    cfg.camera_scale = .62
    return cfg


def decorate(rgb, pose, title, seconds):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 16)
    draw.text((30, 22), title + ' · P2', font=font, fill=(221, 229, 241))
    draw.text((1700, 25), f'{seconds:06.2f} с', font=small, fill=(159, 174, 194))
    # Compact source inset keeps the old stage as the main image.
    draw.rounded_rectangle((22, 726, 270, 1055), radius=10, fill=(5, 8, 12), outline=(27, 34, 44))
    draw.text((40, 740), 'Исходное тело', font=small, fill=(151, 171, 191))
    pose = pose - pose[0] * np.array([1., 0., 1.])
    xy = np.column_stack((146 + (pose[:, 0] * .8 + pose[:, 2] * .6) * 112, 1030 - pose[:, 1] * 112))
    for j, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ink = (110, 164, 199) if j % 2 == 0 else (188, 140, 210)
        draw.line([tuple(xy[parent]), tuple(xy[j])], fill=ink, width=3)
    for x, y in xy:
        draw.ellipse((x-2, y-2, x+2, y+2), fill=(205, 215, 228))
    return np.asarray(image)


def load_track(track):
    with np.load(track['timeline'], allow_pickle=False) as z:
        dmx, body = z['universes'].copy(), z['body'].copy()
    patch = json.loads(Path(track['patch']).read_text())
    if len(patch) != 63 or dmx.shape != (7200, 1, 512):
        raise ValueError('Expected unchanged 63-output, 180s demonstration')
    return decode(dmx, patch), body, patch


def smoke(plan):
    check(plan)
    root = Path(plan['output']) / 'smoke'
    root.mkdir(parents=True, exist_ok=False)
    report = {'config': asdict(config()), 'tracks': [], 'controls': {}}
    with gpu_lock():
        for track in plan['tracks']:
            controls, body, patch = load_track(track)
            renderer = BodyStageRenderer(patch, config())
            started = time.monotonic()
            frames = []
            try:
                for tick in (2400, 3600, 4800, 7199):
                    rgb = renderer.render_controls(*(controls[k][tick] for k in ('directions', 'rgb', 'dimmer')))
                    assert rgb.shape == (1080, 1920, 3)
                    path = root / f'{track["id"]}_{tick:04d}.png'
                    Image.fromarray(decorate(rgb, body[tick], track['title'], tick/40)).save(path)
                    frames.append({'tick':tick, 'path':str(path), 'sha256':sha(path), 'mean_pixel':float(rgb.mean())})
                # Same controls produce identical pictures: shader time fixed.
                args = [controls[k][2400] for k in ('directions', 'rgb', 'dimmer')]
                a = renderer.render_controls(*args)
                b = renderer.render_controls(*args)
                assert np.array_equal(a, b)
                # The final bulb must not disappear behind the old16-light cap.
                zeros = np.zeros(63, dtype=np.float32)
                dark = renderer.render_controls(args[0], np.ones((63,3)), zeros)
                last = zeros.copy(); last[-1] = 1
                one = renderer.render_controls(args[0], np.ones((63,3)), last)
                diff = np.max(np.abs(one.astype(int)-dark.astype(int)), axis=2)
                assert (diff > 5).sum() > 10
                # Each thin laser is represented, including the fifth one.
                lasers = []
                for j, fixture in enumerate(patch):
                    if fixture['kind'] != 'laser':
                        continue
                    levels = zeros.copy(); levels[j] = 1
                    shot = renderer.render_controls(args[0], np.ones((63,3)), levels)
                    delta = np.max(np.abs(shot.astype(int)-dark.astype(int)), axis=2)
                    assert (delta > 5).sum() > 10
                    lasers.append({'name':fixture['name'], 'pixels_above_dark':int((delta>5).sum())})
                report['controls'][track['id']] = {'static_picture_exact':True, 'last_bulb_visible_pixels':int((diff>5).sum()), 'lasers':lasers}
                report['tracks'].append({'id':track['id'], 'renderer':renderer.renderer_name, 'elapsed_s':time.monotonic()-started, 'frames':frames})
            finally:
                renderer.ctx.release()
    check(plan)
    save(root / 'REPORT.json', report)
    print('SMOKE_COMPLETE', root, flush=True)


def render(plan):
    check(plan)
    root = Path(plan['output'])
    start, count = round(plan['start_s'] * 30), round(plan['duration_s'] * 30)
    indices = np.rint(np.arange(start, start+count) * 40/30).astype(int)
    movies = []
    for track in plan['tracks']:
        out = root / track['id']
        out.mkdir(exist_ok=False)
        movie = out / (track['id'] + '_body_P2_dark_stage_1080p_av1.mp4')
        controls, body, patch = load_track(track)
        command = encode_command(movie, {'audio':track['audio']}, 1920, 1080, start, count)
        started = time.monotonic()
        metrics = []
        with gpu_lock():
            renderer = BodyStageRenderer(patch, config())
            device = renderer.renderer_name
            print('GPU', device, flush=True)
            try:
                with (out / 'encode.log').open('xb') as err:
                    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=err)
                    try:
                        for frame, tick in enumerate(indices):
                            rgb = renderer.render_controls(*(controls[k][tick] for k in ('directions', 'rgb', 'dimmer')))
                            sample = rgb[::4, ::4]
                            metrics.append([float(sample.mean()), float(np.mean(sample.max(2)>=250))])
                            rgb = decorate(rgb, body[tick], track['title'], tick/40)
                            encoder.stdin.write(rgb.tobytes())
                            if frame in (0, 900, 1800, 2700, count-1):
                                Image.fromarray(rgb).save(out / f'frame_{frame:04d}.png')
                            if frame % 300 == 0:
                                check(plan)
                                print('DARK_STAGE',track['id'],frame,count,round(time.monotonic()-started,1),flush=True)
                        encoder.stdin.close()
                        if encoder.wait() != 0:
                            raise RuntimeError((out / 'encode.log').read_text())
                    except BaseException:
                        if encoder.poll() is None:
                            encoder.kill(); encoder.wait()
                        raise
            finally:
                renderer.ctx.release()
        assert movie.stat().st_size <= 50_000_000
        np.savez_compressed(out / 'sampled_pixels.npz', values=np.asarray(metrics))
        report = {'path':str(movie), 'sha256':sha(movie), 'frames':count, 'start_s':plan['start_s'],
            'duration_s':plan['duration_s'], 'timeline_sha256':sha(track['timeline']), 'patch_sha256':sha(track['patch']),
            'audio_sha256':sha(track['audio']), 'config':asdict(config()), 'command':command, 'renderer':device,
            'elapsed_s':time.monotonic()-started, 'scope':'Same saved body/P2/DMX as geometric demo; presentation-only old dark stage with expanded fixture support.'}
        save(movie.with_name(movie.stem+'_render.json'), report)
        audit = audit_movie(out, movie)
        pcm = subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',track['audio_reference'],
            '-ac','2','-ar','44100','-f','f32le','-'])
        assert hashlib.sha256(pcm).hexdigest() == audit['pcm_sha256']
        assert audit['decoded_frames']==count and audit['pts_uniform'] and audit['full_software_AV_decode']
        movies.append({'id':track['id'],'path':str(movie),'sha256':sha(movie),'bytes':movie.stat().st_size,
            'audio_matches_source_crop':True,'frames':count,'timeline_sha256':sha(track['timeline']),
            'audit':str(movie.with_name(movie.stem+'_audit.json'))})
        check(plan)
    save(root / 'DELIVERY.json', {'status':'two_dark_stage_demos_complete','movies':movies})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('smoke','render'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    {'smoke':smoke,'render':render}[args.stage](plan)
