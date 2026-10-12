"""Auditable two-song body-transfer portfolio rendering and local review.

The CPU schematic route never imports an EGL renderer or invokes NVENC. GPU
work requires a parent-written release receipt and both campaign locks. These
presentation stages only consume saved, decoded DMX; they generate no motion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.conductor.body_transfer import decode
from scripts.rendering.render_body_transfer import PARENTS, project, ray_hits

WIDTH, HEIGHT, FPS = 1920, 1080, 30
HOST_LOCK = Path('/tmp/lightsem-audio-understanding-gpu.lock')
LEGACY_LOCK = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock')
TEMPLATE = Path(__file__).resolve().parents[1] / 'light_layers/body_portfolio_review.html'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_files(plan):
    if not plan.get('files'):
        raise ValueError('PLAN.files must pin source inputs and implementation')
    for filename, expected in plan['files'].items():
        if sha(filename) != expected:
            raise RuntimeError('Frozen input changed: ' + filename)


def guard(root, plan):
    if shutil.disk_usage(root).free < plan.get('minimum_free_bytes', 20 * 2 ** 30):
        raise RuntimeError('Free-space reserve reached')
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size > plan.get('max_bytes', 2 * 2 ** 30):
        raise RuntimeError('Campaign storage budget exceeded')


def require_release(root):
    path = root / 'records/GPU_RELEASE.json'
    receipt = read(path)
    if receipt.get('released') is not True or receipt.get('basis') not in (
            'owner_confirmed_window', 'verified_owner_queue_complete'):
        raise RuntimeError('A parent-authenticated GPU release is required')
    if not receipt.get('evidence'):
        raise RuntimeError('GPU release must retain owner/queue evidence')
    return {'path': str(path), 'sha256': sha(path), 'basis': receipt['basis']}


@contextmanager
def gpu_admission(root):
    """Fixed lock order; never truncate, steal, or inspect another job's lock."""
    receipt = require_release(root)
    streams = []
    try:
        for path in (HOST_LOCK, LEGACY_LOCK):
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a+')
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('GPU busy; deferred without interrupting owner: ' + str(path)) from error
        lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
            '--format=csv,noheader'], text=True, timeout=15).splitlines()
        # Chromium's compositor is reported by some drivers in the compute
        # table even though it is a graphics process and does not own CUDA
        # work. Keep it visible in the release evidence, but do not treat it as
        # a competing ML workload. Any other foreign process still blocks the
        # campaign without interruption.
        foreign = [line for line in lines
                    if line.strip() and line.split(',')[0].strip() != str(os.getpid())
                    and not ('chrome' in line.lower() and '--type=gpu-process' in line)]
        if foreign:
            raise RuntimeError('Foreign GPU workload; deferred: ' + '; '.join(foreign))
        if sha(root / 'records/GPU_RELEASE.json') != receipt['sha256']:
            raise RuntimeError('GPU release changed during admission')
        yield receipt
    finally:
        for stream in reversed(streams):
            stream.close()


def pairs(plan, algorithm=None, track=None):
    algorithms = plan['algorithms']
    tracks = plan['tracks']
    if algorithm is not None and not any(a['id'] == algorithm for a in algorithms):
        raise ValueError('Unknown algorithm: ' + algorithm)
    if track is not None and not any(t['id'] == track for t in tracks):
        raise ValueError('Unknown track: ' + track)
    for a in algorithms:
        for t in tracks:
            if (algorithm is None or a['id'] == algorithm) and (track is None or t['id'] == track):
                yield a, t


def load_pair(root, plan, algorithm, track):
    folder = root / 'timelines' / algorithm['id'] / track['id']
    report = read(folder / 'report.json')
    if report.get('status') not in ('engineering_valid', 'diagnostic'):
        raise RuntimeError('Pair has not passed numeric admission: ' + str(folder))
    pins = {'timeline_sha256': sha(folder / 'timeline.npz'), 'patch_sha256': sha(folder / 'patch.json')}
    if any(report.get(key) != value for key, value in pins.items()):
        raise RuntimeError('Timeline/patch differs from pre-render numeric report')
    patch = read(folder / 'patch.json')
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        dmx, body, times = (data[k].copy() for k in ('universes', 'body', 'time'))
    n = round(plan['prefix_s'] * 40)
    if body.shape != (n, 22, 3) or dmx.shape[0] != n or times.shape != (n,):
        raise ValueError('Timeline length/body shape differs from frozen plan')
    if not np.allclose(times, np.arange(n) / 40, rtol=0, atol=1e-10):
        raise ValueError('Invalid source timeline clock')
    controls = decode(dmx, patch)
    if not np.isfinite(body).all() or not all(np.isfinite(v).all() for v in controls.values()):
        raise ValueError('Nonfinite source body/emitted controls')
    first = round(plan['review_start_s'] * FPS)
    count = round(plan['review_duration_s'] * FPS)
    indices = np.rint(np.arange(first, first + count) * 40 / FPS).astype(int)
    if not len(indices) or indices[0] < 0 or indices[-1] >= n:
        raise ValueError('Review crop outside the source timeline')
    return controls, body, patch, indices, report, {
        **pins, 'numeric_report_sha256': sha(folder / 'report.json'),
        'audio_sha256': sha(track['audio']), 'plan_sha256': sha(root / 'PLAN.json')}


@lru_cache(maxsize=1)
def fonts():
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return tuple(ImageFont.truetype(path, size) for size in (30, 23, 18))


def decorate(rgb, pose, algorithm, track, seconds, status, fixed=False):
    picture = Image.fromarray(rgb)
    draw = ImageDraw.Draw(picture)
    large, regular, small = fonts()
    draw.rectangle((0, 0, WIDTH, 112), fill=(7, 10, 16))
    draw.text((28, 16), algorithm['id'] + ' | ' + algorithm['name'], font=large, fill=(233, 239, 250))
    draw.text((28, 62), track['title'] + ' | P2 color | emitted DMX', font=regular, fill=(184, 201, 222))
    draw.text((1570, 22), f'Source {seconds:06.2f} s', font=regular, fill=(220, 229, 241))
    label = 'DIAGNOSTIC' if status == 'diagnostic' else 'ENGINEERING CHECKED'
    draw.text((1450, 68), label, font=small, fill=(242, 186, 112))
    draw.rounded_rectangle((20, 480, 414, 1058), radius=12, fill=(6, 9, 15), outline=(65, 80, 99), width=2)
    draw.text((40, 496), 'Source body: this song', font=regular, fill=(207, 221, 239))
    draw.text((40, 532), 'Same timing in body and light', font=small, fill=(146, 169, 195))
    pose = pose - pose[0] * np.array([1., 0., 1.])
    xy = np.column_stack((218 + (.8 * pose[:, 0] + .6 * pose[:, 2]) * 160, 1020 - pose[:, 1] * 160))
    for j, parent in enumerate(PARENTS):
        if parent >= 0:
            ink = (115, 190, 230) if j % 2 == 0 else (228, 151, 230)
            draw.line([tuple(xy[parent]), tuple(xy[j])], fill=ink, width=5)
    for x, y in xy:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(227, 235, 246))
    if fixed:
        draw.rectangle((445, 1016, WIDTH - 15, 1065), fill=(8, 12, 18))
        draw.text((465, 1026), 'Control: fixed white RGB / brightness 0.65; emitted directions unchanged',
                  font=regular, fill=(242, 203, 150))
    return np.asarray(picture)


class SchematicRenderer:
    """CPU-only fixed-camera geometry; all endpoints come from decoded DMX."""
    renderer_name = 'Pillow CPU geometric diagnostic'

    def __init__(self, patch):
        self.patch = patch
        self.positions = np.array([p['position'] for p in patch])
        self.moving = np.array([p['kind'] != 'bulb' for p in patch])
        self.origins = project(self.positions)
        self.canvas = Image.new('RGB', (WIDTH, HEIGHT), (8, 12, 19))
        draw = ImageDraw.Draw(self.canvas)
        for z in (-6., -3., 0., 3., 6.):
            draw.line([tuple(v) for v in project(np.array([[-8, 0, z], [8, 0, z]]))], fill=(29, 43, 59))
        for x in (-8., -4., 0., 4., 8.):
            draw.line([tuple(v) for v in project(np.array([[x, 0, -6], [x, 0, 6]]))], fill=(29, 43, 59))

    def render_controls(self, directions, rgb, dimmer):
        if not all(np.isfinite(x).all() for x in (directions, rgb, dimmer)):
            raise ValueError('Nonfinite emitted controls')
        hits = project(ray_hits(self.positions[self.moving], directions[self.moving]))
        picture = self.canvas.copy()
        draw = ImageDraw.Draw(picture)
        index = 0
        for j, fixture in enumerate(self.patch):
            ink = tuple(np.rint(np.clip(rgb[j] * dimmer[j], 0, 1) * 255).astype(int))
            x, y = self.origins[j]
            if fixture['kind'] != 'bulb':
                endpoint = hits[index]
                index += 1
                draw.line([tuple(self.origins[j]), tuple(endpoint)], fill=ink,
                          width=3 if fixture['kind'] == 'laser' else 8)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ink)
        return np.asarray(picture)

    def close(self):
        pass


@contextmanager
def renderer_for(root, mode, patch):
    if mode == 'schematic':
        yield SchematicRenderer(patch), None, None
        return
    with gpu_admission(root) as release:
        from lightsem.sim.body_stage_renderer import BodyStageRenderer
        from scripts.motion.run_rhythm_companion import render_config
        cfg = render_config()
        cfg.camera_scale = .62
        renderer = BodyStageRenderer(patch, cfg)
        try:
            yield renderer, asdict(cfg), release
        finally:
            renderer.ctx.release()


def command_for(movie, track, plan, codec):
    first, count = round(plan['review_start_s'] * FPS), round(plan['review_duration_s'] * FPS)
    if codec == 'av1_nvenc':
        from scripts.motion.run_rhythm_companion import encode_command
        return encode_command(movie, track, WIDTH, HEIGHT, first, count)
    return ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
        '-i', 'pipe:0', '-ss', str(first / FPS), '-i', track['audio'], '-map', '0:v:0', '-map', '1:a:0',
        '-frames:v', str(count), '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
        '-b:v', '1800k', '-maxrate', '2400k', '-bufsize', '4800k', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', '-af', f'atrim=duration={count / FPS},asetpts=PTS-STARTPTS',
        '-t', str(count / FPS), '-movflags', '+faststart', str(movie)]


def validate_mode(mode, codec):
    if mode == 'schematic' and codec != 'libx264':
        raise ValueError('CPU schematic mode requires explicit --codec libx264')


def smoke_pair(root, plan, algorithm, track, mode):
    controls, body, patch, indices, report, pins = load_pair(root, plan, algorithm, track)
    out = root / 'smoke' / algorithm['id'] / track['id'] / mode
    receipt_path = out / 'SMOKE.json'
    if receipt_path.exists():
        previous = read(receipt_path)
        if previous['inputs'] != pins or any(sha(p) != h for p, h in previous['files'].items()):
            raise RuntimeError('Existing smoke artifacts changed')
        return previous
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    saved = {}
    with renderer_for(root, mode, patch) as (renderer, cfg, release):
        for tick in indices[np.linspace(0, len(indices) - 1, 4).round().astype(int)]:
            args = [controls[k][tick] for k in ('directions', 'rgb', 'dimmer')]
            actual = renderer.render_controls(*args)
            repeated = renderer.render_controls(*args)
            if not np.array_equal(actual, repeated):
                raise RuntimeError('Frozen emitted controls produce changing pixels')
            fixed = renderer.render_controls(args[0], np.ones_like(args[1]), np.full_like(args[2], .65))
            for kind, image in (('actual', actual), ('fixed_brightness', fixed)):
                filename = out / f'{int(tick):06d}_{kind}.png'
                Image.fromarray(decorate(image, body[tick], algorithm, track, tick / 40,
                    report['status'], kind != 'actual')).save(filename)
                saved[str(filename)] = sha(filename)
            rows.append({'tick': int(tick), 'time_s': float(tick / 40), 'static_pixels_exact': True,
                         'mean_pixel': float(actual.mean()), 'saturated_fraction': float(np.mean(actual.max(2) >= 250))})
        receipt = {'inputs': pins, 'files': saved, 'samples': rows, 'mode': mode, 'renderer': renderer.renderer_name,
                   'renderer_config': cfg, 'gpu_release': release, 'finite_emitted_controls': True}
    save(receipt_path, receipt)
    return receipt


def render_pair(root, plan, algorithm, track, mode, codec):
    controls, body, patch, indices, report, pins = load_pair(root, plan, algorithm, track)
    smoke = root / 'smoke' / algorithm['id'] / track['id'] / mode / 'SMOKE.json'
    if not smoke.exists() or read(smoke)['inputs'] != pins:
        raise RuntimeError('Run and inspect smoke outputs before full rendering')
    out = root / 'media' / algorithm['id'] / track['id']
    out.mkdir(parents=True, exist_ok=True)
    movie = out / (mode + '.mp4')
    sidecar = out / (mode + '_render.json')
    if sidecar.exists():
        existing = read(sidecar)
        if existing['inputs'] != pins or existing['codec'] != codec or sha(movie) != existing['sha256']:
            raise RuntimeError('Existing completed render differs from frozen inputs')
        return existing
    if movie.exists():
        raise RuntimeError('Partial movie retained; use a new campaign revision: ' + str(movie))
    command = command_for(movie, track, plan, codec)
    started = time.monotonic()
    samples = []
    with renderer_for(root, mode, patch) as (renderer, cfg, release):
        with (out / (mode + '_encode.log')).open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(indices):
                    rgb = renderer.render_controls(*(controls[k][tick] for k in ('directions', 'rgb', 'dimmer')))
                    pixels = rgb[::4, ::4]
                    samples.append([float(pixels.mean()), float(np.mean(pixels.max(2) >= 250))])
                    decorated = decorate(rgb, body[tick], algorithm, track, tick / 40, report['status'])
                    encoder.stdin.write(decorated.tobytes())
                    if frame % 300 == 0:
                        guard(root, plan)
                        print('RENDER', algorithm['id'], track['id'], mode, frame, len(indices), flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encoder failed; inspect ' + str(out / (mode + '_encode.log')))
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
        if movie.stat().st_size > 50_000_000:
            raise RuntimeError('Video exceeds 50 MB; preserve failed output')
        result = {'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
            'inputs': pins, 'frames': len(indices), 'duration_s': len(indices) / FPS,
            'start_s': plan['review_start_s'], 'codec': codec, 'mode': mode,
            'renderer': renderer.renderer_name, 'renderer_config': cfg, 'gpu_release': release,
            'command': command, 'elapsed_s': time.monotonic() - started, 'numeric_status': report['status'],
            'scope': 'Saved body and actual decoded DMX; no motion generation or choreography acceptance'}
    np.savez_compressed(out / (mode + '_sampled_pixels.npz'), values=np.asarray(samples))
    save(sidecar, result)
    return result


def audit_h264(movie, result):
    """Software-codec equivalent of the historical AV1-only audit helper."""
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(movie)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    if (video['codec_name'], video['width'], video['height'], video['r_frame_rate']) != ('h264', WIDTH, HEIGHT, '30/1'):
        raise RuntimeError('Unexpected software delivery format')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)], text=True))['frames']
    times = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    if len(times) != result['frames'] or not np.allclose(times, np.arange(len(times)) / FPS, atol=1e-5):
        raise RuntimeError('Decoded frame count/timestamps mismatch')
    if abs(float(audio['duration']) - result['duration_s']) > .06:
        raise RuntimeError('Audio duration mismatch')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-threads', '2',
        '-i', str(movie), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
    pcm = decoded_audio(movie)
    return {'path': str(movie), 'video': video, 'audio': audio, 'decoded_frames': len(times),
        'pts_uniform': True, 'full_software_AV_decode': True, 'pcm_sha256': hashlib.sha256(pcm).hexdigest()}


def decoded_audio(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', str(path),
        '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100', '-f', 'f32le', '-'])


def audit_pair(root, plan, algorithm, track, mode):
    _, _, _, _, _, pins = load_pair(root, plan, algorithm, track)
    out = root / 'media' / algorithm['id'] / track['id']
    movie = out / (mode + '.mp4')
    result = read(out / (mode + '_render.json'))
    if result['inputs'] != pins or sha(movie) != result['sha256']:
        raise RuntimeError('Render provenance/hash changed')
    path = out / (mode + '_audit.json')
    if path.exists():
        audit = read(path)
    elif result['codec'] == 'av1_nvenc':
        from scripts.motion.run_rhythm_companion import audit_movie
        audit = audit_movie(out, movie)
    else:
        audit = audit_h264(movie, result)
        save(path, audit)
    reference = Path(track['audio_reference'])
    if str(reference) not in plan['files']:
        raise RuntimeError('Audio reference must be pinned in PLAN.files')
    expected_pcm = hashlib.sha256(decoded_audio(reference)).hexdigest()
    if audit['pcm_sha256'] != expected_pcm or audit['decoded_frames'] != result['frames']:
        raise RuntimeError('Delivered audio/frame count differs from authenticated source crop')
    if not audit['pts_uniform'] or not audit['full_software_AV_decode']:
        raise RuntimeError('Incomplete media audit')
    receipt = {'video_sha256': result['sha256'], 'audit_sha256': sha(path),
        'source_reference_sha256': sha(reference), 'audio_matches_source_crop': True, 'inputs': pins}
    receipt_path = out / (mode + '_verified.json')
    if receipt_path.exists():
        if read(receipt_path) != receipt:
            raise RuntimeError('Existing media verification differs')
    else:
        save(receipt_path, receipt)
    return receipt


def review(root, plan, mode):
    variants = []
    for index, (algorithm, track) in enumerate(pairs(plan), 1):
        out = root / 'media' / algorithm['id'] / track['id']
        sidecar = out / (mode + '_render.json')
        verification = out / (mode + '_verified.json')
        row = {'id': f'{index:02d}', 'algorithm_id': algorithm['id'], 'track_id': track['id'],
               'title': algorithm['name'] + ' / ' + track['title'], 'status': 'planned', 'mode': mode,
               'papers': algorithm.get('papers', [])}
        report_path = root / 'timelines' / algorithm['id'] / track['id'] / 'report.json'
        if report_path.exists():
            report = read(report_path)
            row.update(numeric_status=report.get('status'), diagnostics=report.get('diagnostics', {}))
        if sidecar.exists() and verification.exists():
            result, receipt = read(sidecar), read(verification)
            _, _, _, _, _, pins = load_pair(root, plan, algorithm, track)
            if (result['inputs'] != pins or receipt['inputs'] != pins
                    or sha(result['path']) != result['sha256'] or receipt['video_sha256'] != result['sha256']
                    or sha(out / (mode + '_audit.json')) != receipt['audit_sha256']):
                raise RuntimeError('Review video changed')
            row.update(status='complete', path=result['path'], sha256=result['sha256'], bytes=result['bytes'],
                       download_name=f"{algorithm['id']}_{track['id']}_{mode}.mp4")
        variants.append(row)
    for index, track in enumerate(plan['tracks'], 31):
        old = Path('/root/lightsem/runs/body_transfer_dark_stage_20260930_v1') / track['id']
        movie = old / (track['id'] + '_body_P2_dark_stage_1080p_av1.mp4')
        if movie.exists():
            variants.append({'id': f'{index:02d}', 'algorithm_id': 'baseline', 'track_id': track['id'],
                'title': 'Previous body-transfer baseline / ' + track['title'], 'status': 'complete',
                'numeric_status': 'diagnostic', 'mode': 'dark', 'path': str(movie), 'sha256': sha(movie),
                'bytes': movie.stat().st_size, 'download_name': movie.name, 'papers': []})
    manifest = {'campaign': root.name, 'mode': mode, 'tracks': plan['tracks'], 'algorithms': plan['algorithms'],
        'review_start_s': plan['review_start_s'], 'review_duration_s': plan['review_duration_s'], 'variants': variants,
        'plan_sha256': sha(root / 'PLAN.json'), 'scope': 'Engineering candidates; viewer judges gesture transfer.'}
    dest = root / 'review'
    dest.mkdir(exist_ok=True)
    # This is the live review index; immutable media/provenance files stay intact.
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    shutil.copyfile(TEMPLATE, dest / 'index.html')
    print('REVIEW_READY', dest, sum(v['status'] == 'complete' for v in variants), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'render', 'review', 'audit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--algorithm')
    parser.add_argument('--track')
    parser.add_argument('--mode', choices=('dark', 'schematic'), default='dark')
    parser.add_argument('--codec', choices=('av1_nvenc', 'libx264'), default='av1_nvenc')
    args = parser.parse_args()
    validate_mode(args.mode, args.codec)
    root = args.root.resolve()
    plan = read(root / 'PLAN.json')
    verify_files(plan)
    guard(root, plan)
    if args.stage == 'review':
        review(root, plan, args.mode)
        return
    for algorithm, track in pairs(plan, args.algorithm, args.track):
        verify_files(plan)
        if args.stage == 'smoke':
            smoke_pair(root, plan, algorithm, track, args.mode)
        elif args.stage == 'render':
            render_pair(root, plan, algorithm, track, args.mode, args.codec)
        else:
            audit_pair(root, plan, algorithm, track, args.mode)
        guard(root, plan)


if __name__ == '__main__':
    main()
