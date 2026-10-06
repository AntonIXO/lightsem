"""Native 1080p geometric preview of a body and actual decoded virtual DMX.

This deliberately shows the transfer mechanism, not a photometric venue render.
Moving rays use decoded pan/tilt, bulbs use decoded brightness, both use P2 RGB.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lightsem.conductor.body_transfer import decode

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def project(points):
    p = np.asarray(points)
    return np.stack((1210 + 62 * (p[..., 0] + .36 * p[..., 2]),
                     825 - 87 * (p[..., 1] - .22 * p[..., 2])), -1)


def ray_hits(positions, directions):
    """First intersection with fixed stage floor/back/side/front/ceiling planes."""
    directions = np.asarray(directions)
    origin = np.broadcast_to(positions, directions.shape)
    distance = np.full(directions.shape[:-1], np.inf)
    for axis, value in ((0, -8.), (0, 8.), (1, 0.), (1, 8.), (2, -6.), (2, 6.)):
        denominator = directions[..., axis]
        with np.errstate(divide='ignore', invalid='ignore'):
            step = (value - origin[..., axis]) / denominator
        valid = np.isfinite(step) & (step > 1e-6)
        distance = np.minimum(distance, np.where(valid, step, np.inf))
    if not np.isfinite(distance).all():
        raise ValueError('Unbounded virtual beam')
    return origin + directions * distance[..., None]


def render_one(out, track, plan):
    out = Path(out)
    movie = out / (track['id'] + '_body_to_63_P2_1080p_av1.mp4')
    if movie.exists():
        raise FileExistsError(movie)
    with np.load(out / 'timeline.npz', allow_pickle=False) as data:
        dmx, body = data['universes'], data['body']
    patch = json.loads((out / 'patch.json').read_text())
    decoded = decode(dmx, patch)
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    positions = np.array([p['position'] for p in patch])
    hits = ray_hits(positions[moving], decoded['directions'][:, moving])
    endpoints = project(hits)
    origins = project(positions)
    first = round(plan['review_start_s'] * 30)
    count = round(plan['review_duration_s'] * 30)
    indices = np.rint(np.arange(first, first + count) * 40 / 30).astype(int)
    assert indices[-1] < len(dmx)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    if not Path(font_path).exists():
        font_path = '/usr/share/fonts/TTF/DejaVuSans.ttf'
    regular = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 17)
    title = ImageFont.truetype(font_path, 29)
    canvas = Image.new('RGB', (1920, 1080), (8, 12, 19))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), track['title'], font=title, fill=(232, 238, 248))
    draw.text((30, 63), 'Тело → свет · P2 · геометрическая диагностика переноса', font=regular, fill=(173, 190, 210))
    draw.line((515, 120, 515, 996), fill=(48, 61, 79), width=2)
    draw.text((30, 120), 'Исходное тело этой песни', font=regular, fill=(217, 228, 245))
    draw.text((550, 120), '8 лучей + 5 виртуальных лазеров + 50 лампочек', font=regular, fill=(217, 228, 245))
    draw.text((30, 163), 'Спереди', font=small, fill=(145, 163, 184))
    draw.text((30, 557), 'Под углом', font=small, fill=(145, 163, 184))
    draw.text((550, 940), 'Направления и яркость показаны после ограничений и квантования DMX.', font=regular, fill=(173, 190, 210))
    draw.text((550, 976), 'Движение задаёт тело. Отдельных ритмических траекторий нет.', font=regular, fill=(173, 190, 210))
    draw.text((30, 1032), 'Неподвижное тело → неподвижный рисунок. Пространственные преобразования не добавляют новых движений.', font=small, fill=(140, 159, 183))
    # Fixed room wireframe shows the floor/wall chart without camera movement.
    for z in (-6., -3., 0., 3., 6.):
        points = project(np.array([[-8, 0, z], [8, 0, z]]))
        draw.line([tuple(v) for v in points], fill=(29, 43, 59), width=1)
    for x in (-8., -4., 0., 4., 8.):
        points = project(np.array([[x, 0, -6], [x, 0, 6]]))
        draw.line([tuple(v) for v in points], fill=(29, 43, 59), width=1)
    for y in (0., 2., 4., 6.):
        points = project(np.array([[-8, y, -6], [8, y, -6]]))
        draw.line([tuple(v) for v in points], fill=(27, 38, 51), width=1)
    bulbs = np.flatnonzero(~moving)
    draw.line([tuple(origins[j]) for j in bulbs], fill=(56, 58, 67), width=2)
    for j, item in enumerate(patch):
        if item['kind'] != 'bulb':
            x, y = origins[j]
            draw.text((x - 12, y - 29), ('Л' if item['kind'] == 'laser' else 'Г') + item['name'].split('_')[-1].lstrip('0'),
                      font=small, fill=(130, 145, 164))
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1920x1080', '-r', '30', '-i', 'pipe:0', '-ss', str(plan['review_start_s']), '-i', track['audio'],
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'av1_nvenc', '-preset', 'p7', '-tune', 'uhq',
        '-b:v', '1100k', '-maxrate', '1200k', '-bufsize', '2000k', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', '-t', str(plan['review_duration_s']), str(movie)]
    selected = set(np.linspace(0, count - 1, 6).round().astype(int))
    peak = 0
    started = time.monotonic()
    log_path = out / 'encode.log'
    with log_path.open('xb') as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame, tick in enumerate(indices):
                picture = canvas.copy()
                draw = ImageDraw.Draw(picture)
                pose = body[tick].copy()
                # Camera follows horizontal root for readability; it does not
                # edit source coordinates or feed the transfer calculation.
                pose -= pose[0] * np.array([1., 0., 1.])
                for view, floor in ((0, 510), (1, 914)):
                    x = pose[:, 0] if view == 0 else .8 * pose[:, 0] + .6 * pose[:, 2]
                    xy = np.stack((266 + x * 165, floor - pose[:, 1] * 165), 1)
                    draw.line((60, floor, 468, floor), fill=(52, 64, 80), width=1)
                    for joint, parent in enumerate(PARENTS):
                        if parent < 0:
                            continue
                        color = (113, 190, 226) if joint % 2 == 0 else (218, 153, 235)
                        draw.line([tuple(xy[parent]), tuple(xy[joint])], fill=color, width=5)
                    for xj, yj in xy:
                        draw.ellipse((xj - 3, yj - 3, xj + 3, yj + 3), fill=(227, 235, 246))
                moving_index = 0
                for j, item in enumerate(patch):
                    rgb = decoded['rgb'][tick, j]
                    level = decoded['dimmer'][tick, j]
                    ink = tuple(np.rint(rgb * level * 255).astype(int))
                    x, y = origins[j]
                    if item['kind'] == 'bulb':
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ink)
                        continue
                    end = endpoints[tick, moving_index]
                    moving_index += 1
                    width = 3 if item['kind'] == 'laser' else 8
                    dark = tuple(int(v * .20) for v in ink)
                    draw.line([tuple(origins[j]), tuple(end)], fill=dark, width=width + 9)
                    draw.line([tuple(origins[j]), tuple(end)], fill=ink, width=width)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=ink)
                    xe, ye = end
                    draw.ellipse((xe - 4, ye - 3, xe + 4, ye + 3), fill=ink)
                draw.text((1660, 69), f'{tick / 40:06.2f} с', font=regular, fill=(210, 220, 235))
                data = np.asarray(picture)
                peak = max(peak, int(data.max()))
                if frame in selected:
                    picture.save(out / f'frame_{frame:04d}.png')
                encoder.stdin.write(data.tobytes())
                if frame % 600 == 0:
                    print('RENDER_BODY_N', track['id'], frame, count, flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(log_path.read_text())
        except BaseException:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            raise
    if movie.stat().st_size > 50_000_000:
        raise RuntimeError('Video size cap')
    report = {'frames': count, 'source_start_s': plan['review_start_s'], 'duration_s': plan['review_duration_s'],
        'path': str(movie), 'sha256': digest(movie), 'bytes': movie.stat().st_size,
        'timeline_sha256': digest(out / 'timeline.npz'), 'patch_sha256': digest(out / 'patch.json'),
        'source_audio': track['audio'], 'source_audio_sha256': digest(track['audio']),
        'command': command, 'elapsed_s': time.monotonic() - started, 'pixel_max': peak,
        'scope': 'Fixed-camera geometric preview; all light directions and levels decoded from emitted virtual DMX. Body camera recenters for readability; transfer does not use this display transform.'}
    with movie.with_name(movie.stem + '_render.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return movie
