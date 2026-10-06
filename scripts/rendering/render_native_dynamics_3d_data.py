"""Guarded 90–150 s human review of saved native bodies and emitted DMX.

Camera/optics are the existing audio-representation lit-body presentation.
Only smoke/render create GPU contexts; the enclosing campaign guard owns locks.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.rendering import render_native_dynamics as native
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from scripts.rendering.native_dynamics_traces import Strip

WIDTH, HEIGHT, FPS = 1280, 1024, 30
VIEW_WIDTH, VIEW_HEIGHT, VIEW_TOP = 640, 600, 84
START, DURATION, FRAMES = 90, 60, 1800
PLAN_SHA = '48696e41fdbb1700d117043cc9a4acdd612a3cb3e0fb3baba9d673efd841b36b'
METHODS = {'D7_reference': 'A | Matched continued training',
           'C3': 'B | Native dynamics losses', 'C4': 'C | Joint native dynamics planner'}
CAMERAS = {
    'body': dict(cam_pos=(2.8, 2.15, 3.8), cam_target=(0., 1., -3.5), camera_scale=.20),
    'stage': dict(cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02),
}
FILL_BY_VIEW = {'body': .50, 'stage': .12}
TEMPLATE = Path(__file__).with_name('native_dynamics_3d_data_review.html')
sha, read, save = native.sha, native.read, native.save


def display_pose(pose):
    """Existing mapper-local presentation: remove root XZ, retain height/facing."""
    pose = np.asarray(pose)
    if pose.shape[-2:] != (22, 3) or not np.isfinite(pose).all():
        raise ValueError('Finite 22-joint body required')
    return pose - pose[..., 0:1, :] * np.array([1., 0., 1.])


def frame_indices():
    body = np.arange(START * FPS, (START + DURATION) * FPS)
    return body, np.rint(body * 40 / FPS).astype(int)


def campaign(root, song):
    if sha(root / 'PLAN.json') != PLAN_SHA:
        raise ValueError('Campaign PLAN changed')
    plan = read(root / 'PLAN.json')
    track = next(t for t in plan['tracks'] if t['id'] == song)
    return plan, track


def audio_paths(root, track):
    out = root / 'preview_3d_data_v1' / 'audio' / track['id']
    return out / 'reference.m4a', out / 'REFERENCE.json'


def audio_pins(root, plan, track):
    reference, record = audio_paths(root, track)
    value = read(record)
    expected = {'plan_sha256': PLAN_SHA, 'source_audio': track['audio'],
        'source_audio_sha256': native.verify_source(plan, track),
        'reference': str(reference), 'reference_sha256': sha(reference),
        'source_start_s': START, 'duration_s': DURATION, 'playback_rate': 1.}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Prepared audio binding changed')
    return {'audio_record_sha256': sha(record),
        'audio_reference_sha256': expected['reference_sha256'],
        'source_audio_sha256': expected['source_audio_sha256'],
        'decoded_pcm_sha256': value['decoded_pcm_sha256'],
        'source_start_s': START, 'duration_s': DURATION, 'playback_rate': 1.}


def prepare_audio(root, song):
    """CPU only: one exact original-speed 90–150 s AAC reference per song."""
    plan, track = campaign(root, song)
    source_sha = native.verify_source(plan, track)
    reference, record = audio_paths(root, track)
    if record.exists():
        audio_pins(root, plan, track)
        if hashlib.sha256(native.base.decoded_audio(reference)).hexdigest() != read(record)['decoded_pcm_sha256']:
            raise ValueError('Prepared reference PCM changed')
        return read(record)
    if reference.exists():
        raise ValueError('Unreceipted reference retained')
    native.resource_guard(root, plan)
    reference.parent.mkdir(parents=True, exist_ok=True)
    partial = reference.with_name('reference.partial.m4a')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-i', track['audio'], '-map', '0:a:0', '-vn',
        '-af', 'atrim=start=90:end=150,asetpts=PTS-STARTPTS',
        '-c:a', 'aac', '-b:a', '96k', '-ac', '2', '-ar', '44100',
        '-t', '60', '-movflags', '+faststart', str(partial)]
    subprocess.run(command, check=True)
    os.link(partial, reference)
    partial.unlink()
    value = {'plan_sha256': PLAN_SHA, 'source_audio': track['audio'],
        'source_audio_sha256': source_sha, 'reference': str(reference),
        'reference_sha256': sha(reference), 'source_start_s': START,
        'duration_s': DURATION, 'playback_rate': 1., 'command': command,
        'decoded_pcm_sha256': hashlib.sha256(native.base.decoded_audio(reference)).hexdigest()}
    save(record, value)
    return value


def load(root, method, song):
    plan, track = campaign(root, song)
    pair = native.load_pair(root, plan, method, track)
    source = DanceResult.load(root / 'bodies' / method / song / 'body.npz')
    body_indices, ticks = frame_indices()
    body = display_pose(source.world[body_indices])
    if not np.isfinite(body).all() or ticks[-1] >= len(pair.times):
        raise ValueError('Invalid review body/clock')
    pins = {'plan_sha256': PLAN_SHA, **pair.pins,
            'audio': audio_pins(root, plan, track),
            'renderer_sha256': sha(Path(__file__)), 'template_sha256': sha(TEMPLATE)}
    strip=Strip(root,method,song)
    pins['trace_inputs']=strip.data['inputs']
    pins['trace_code_sha256']=sha(Path(__file__).with_name('native_dynamics_traces.py'))
    return plan, track, pair, body, ticks, pins


def folder(root, method, song):
    return root / 'preview_3d_data_v1' / method / song


@contextmanager
def renderers(root, patch, guarded):
    if not guarded:
        raise ValueError('GPU rendering requires --guarded and the sequential curator')
    from scripts.motion.native_dynamics_guard import authenticate_guarded_job
    receipt = authenticate_guarded_job(__file__)
    command = receipt['identity']['command']
    if '--root' not in command or Path(command[command.index('--root') + 1]).resolve() != root.resolve():
        raise ValueError('Guard command does not bind campaign root')
    from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
    from lightsem.sim.renderer import RenderConfig
    created, configs = [], []
    try:
        for name, camera in CAMERAS.items():
            cfg = RenderConfig(width=VIEW_WIDTH, height=VIEW_HEIGHT, fps=FPS, cam_orbit=0.,
                haze=.25, intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                require_gpu=True, **camera)
            created.append(LitBodyStageRenderer(patch, cfg, body_offset=(0., 0., -3.5),
                                               body_scale=1., neutral_fill=FILL_BY_VIEW[name]))
            configs.append(asdict(cfg))
        yield created, {'configs': configs, 'fill_by_view': dict(FILL_BY_VIEW),
            'presentation_by_view': {name: renderer.presentation_config()
                                     for name, renderer in zip(CAMERAS, created)},
            'pose_transform': 'root XZ removed for viewing; native pose/height/timing retained',
            'max_native_body_to_nearest_dmx_offset_s': float(np.max(np.abs(
                frame_indices()[0] / FPS - frame_indices()[1] / 40))),
            'gpu_receipt': receipt}
    finally:
        for renderer in reversed(created):
            renderer.close()


@lru_cache(maxsize=1)
def fonts():
    return tuple(ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', n) for n in (22, 15, 12))


def compose(views, method, track, stamp, strip):
    picture = Image.new('RGB', (WIDTH, HEIGHT), (7, 10, 16))
    for x, view in zip((0, VIEW_WIDTH), views):
        if view.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
            raise ValueError('Unexpected native view dimensions')
        picture.paste(Image.fromarray(view), (x, VIEW_TOP))
    draw = ImageDraw.Draw(picture)
    large, regular, small = fonts()
    draw.text((16, 8), METHODS[method], font=large, fill=(236, 241, 249))
    draw.text((16, 39), track['title'] + ' | P2 color | actual emitted DMX | restored brightness',
              font=regular, fill=(185, 205, 226))
    draw.text((1030, 12), f'Source {stamp:06.2f} s', font=regular, fill=(214, 226, 241))
    draw.text((1020, 40), ('EXPERIMENTAL CONTROL' if method == 'D7_reference' else 'FAILURE DIAGNOSTIC'), font=small, fill=(242, 190, 119))
    draw.text((16, 66), 'LIT SOURCE BODY — THIS SONG', font=small, fill=(181, 201, 224))
    draw.text((656, 66), 'SAME EMITTED LIGHTS — WIDE STAGE', font=small, fill=(181, 201, 224))
    draw.line((639, VIEW_TOP, 639, VIEW_TOP + VIEW_HEIGHT), fill=(50, 65, 84), width=1)
    caption = ('Matched experimental control; no baseline promotion.' if method == 'D7_reference'
               else 'Failed physical / transfer gates; diagnostic only, no baseline promotion.')
    draw.text((16, 688), caption + ' Native generation; no timewarp. Original audio speed.',
              font=small, fill=(242, 190, 119))
    draw.text((16, 705), 'Root XZ removed for viewing; native pose/height/timing retained. Viewer fill: close .50 / wide .12; separate from DMX.',
              font=small, fill=(143, 164, 189))
    picture.paste(strip.frame(stamp),(0,720))
    return np.asarray(picture)


def controls_at(pair, tick):
    return tuple(pair.controls[k][tick] for k in ('directions', 'rgb', 'dimmer'))


def smoke(root, method, song, guarded):
    plan, track, pair, body, ticks, pins = load(root, method, song)
    out = folder(root, method, song) / 'smoke'
    if out.exists():
        raise ValueError('Preserve prior smoke; inspect existing evidence or use a new source revision')
    native.resource_guard(root, plan)
    strip=Strip(root,method,song)
    files, samples = {}, []
    with renderers(root, pair.patch, guarded) as (rs, config):
        out.mkdir(parents=True, exist_ok=False)
        for stamp in (90., 105., 120., 135., 149.):
            frame = round((stamp - START) * FPS)
            controls = controls_at(pair, ticks[frame])
            views, metrics = [], []
            for index, renderer in enumerate(rs):
                diagnostic = renderer.render_diagnostics(*controls, body[frame])
                repeated = renderer.render_controls(*controls, body[frame])
                if not np.array_equal(repeated, diagnostic['actual']):
                    raise ValueError('Frozen body/DMX produced autonomous pixels')
                if diagnostic['metrics']['body_visible_pixels'] == 0:
                    raise ValueError('Source body invisible in camera ' + str(index))
                views.append(diagnostic['actual'])
                metrics.append(diagnostic['metrics'])
                if index == 0:
                    for key in ('lights_off', 'body_mask', 'direct_body_light'):
                        path = out / f'{stamp:06.2f}_{key}.png'
                        Image.fromarray(diagnostic[key]).save(path)
                        files[str(path)] = sha(path)
            path = out / f'{stamp:06.2f}_actual.png'
            Image.fromarray(compose(views, method, track, stamp, strip)).save(path)
            files[str(path)] = sha(path)
            samples.append({'source_s': stamp, 'body_frame': int(START * FPS + frame),
                'dmx_tick': int(ticks[frame]), 'frozen_pixels_exact': True, 'views': metrics})
    strip.export(out/'TRACE.npz')
    files[str(out/'TRACE.npz')]=sha(out/'TRACE.npz')
    value = {'inputs': pins, 'files': files, 'samples': samples, 'traces':strip.evidence(), **config}
    save(out / 'SMOKE.json', value)
    return value


def require_smoke(out, pins):
    record = out / 'smoke' / 'SMOKE.json'
    smoke_value = read(record)
    inspection = read(record.with_name('INSPECTED.json'))
    if smoke_value['inputs'] != pins or inspection.get('inspected') is not True or inspection.get('smoke_sha256') != sha(record):
        raise ValueError('Root inspection of matching GPU smoke required')
    if any(sha(p) != h for p, h in smoke_value['files'].items()):
        raise ValueError('Smoke pixels changed')
    return sha(record)


def encode_command(path, audio):
    # Six 60 s clips: 1.1 Mb/s video + 96 kb/s audio is approximately 54 MB.
    return ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x1024', '-r', '30', '-i', 'pipe:0',
        '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
        '-frames:v', '1800', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
        '-b:v', '1100k', '-maxrate', '1250k', '-bufsize', '2500k', '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-c:a', 'copy', '-t', '60',
        '-movflags', '+faststart', str(path)]


def render(root, method, song, guarded):
    plan, track, pair, body, ticks, pins = load(root, method, song)
    out = folder(root, method, song)
    smoke_sha = require_smoke(out, pins)
    movie, record = out / 'preview.mp4', out / 'RENDER.json'
    if movie.exists() or record.exists():
        raise ValueError('Existing preview retained; audit it instead')
    native.resource_guard(root, plan)
    partial = out / 'preview.partial.mp4'
    reference, _ = audio_paths(root, track)
    command = encode_command(partial, reference)
    strip=Strip(root,method,song)
    started = time.monotonic()
    with renderers(root, pair.patch, guarded) as (rs, config):
        with (out / 'encode.log').open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(ticks):
                    controls = controls_at(pair, tick)
                    views = [r.render_controls(*controls, body[frame]) for r in rs]
                    encoder.stdin.write(compose(views, method, track, START + frame / FPS, strip).tobytes())
                    if frame % 300 == 0:
                        native.resource_guard(root, plan)
                        print('PREVIEW_3D', method, song, frame, FRAMES, flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encode failed; partial retained')
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
    if partial.stat().st_size > 11_000_000:
        raise ValueError('Per-clip delivery budget exceeded; partial retained')
    os.link(partial, movie)
    partial.unlink()
    value = {'inputs': pins, 'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
        'method': method, 'song': song, 'source_start_s': START, 'duration_s': DURATION,
        'frames': FRAMES, 'audio_speed': 1., 'body_fps': 30, 'dmx_fps': 40,
        'smoke_sha256': smoke_sha, 'status': ('experimental_control' if method == 'D7_reference' else 'failure_diagnostic'),
        'baseline_promotion': False,
        'command': command, 'elapsed_s': time.monotonic() - started, 'traces':strip.evidence(), **config}
    save(record, value)
    return value


def audit(root, method, song):
    plan, track, _, _, _, pins = load(root, method, song)
    out = folder(root, method, song)
    value = read(out / 'RENDER.json')
    movie = out / 'preview.mp4'
    if value['inputs'] != pins or value['sha256'] != sha(movie):
        raise ValueError('Preview or source authentication failed')
    require_smoke(out, pins)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(movie)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    if (video['width'], video['height'], video['r_frame_rate'], int(video['nb_read_frames'])) != (WIDTH, HEIGHT, '30/1', FRAMES):
        raise ValueError('Preview dimensions/clock/frame count failed')
    if abs(float(video['duration']) - DURATION) > 1/FPS or abs(float(audio['duration']) - DURATION) > .05:
        raise ValueError('Video/audio duration mismatch')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)], text=True))['frames']
    times = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    if len(times) != FRAMES or not np.allclose(times, np.arange(FRAMES)/FPS, rtol=0, atol=1e-5):
        raise ValueError('Incomplete or nonuniform decoded video timestamps')
    if abs(float(audio.get('start_time', 0))) > 1/44100:
        raise ValueError('Audio start synchronization failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-threads', '2',
        '-i', str(movie), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
    reference, reference_record = audio_paths(root, track)
    reference_pcm = hashlib.sha256(native.base.decoded_audio(reference)).hexdigest()
    delivered_pcm = hashlib.sha256(native.base.decoded_audio(movie)).hexdigest()
    if delivered_pcm != reference_pcm or delivered_pcm != read(reference_record)['decoded_pcm_sha256']:
        raise ValueError('Delivered PCM differs from authenticated 90–150 s reference')
    report = {'render_sha256': sha(out / 'RENDER.json'), 'movie_sha256': sha(movie), 'inputs': pins,
        'full_software_AV_decode': True, 'decoded_frames': len(times), 'pts_uniform': True,
        'pcm_sha256': delivered_pcm, 'audio_matches_prepared_reference': True,
        'source_start_s': START, 'duration_s': DURATION, 'audio_speed': 1., 'probe': probe}
    audit_path = out / 'AUDIT.json'
    if audit_path.exists():
        old = read(audit_path)
        if old != report:
            raise ValueError('Existing audit differs')
    else:
        save(audit_path, report)
    return report


def gallery(root):
    items = []
    for song in ('shpongle', 'cure_for_me'):
        for method, label in METHODS.items():
            out = folder(root, method, song)
            _, _, _, _, _, pins = load(root, method, song)
            render_value, checked = read(out / 'RENDER.json'), read(out / 'AUDIT.json')
            if (checked.get('inputs') != pins or checked.get('decoded_frames') != FRAMES
                    or any(checked.get(k) is not True for k in ('full_software_AV_decode', 'pts_uniform', 'audio_matches_prepared_reference'))
                    or checked.get('pcm_sha256') != pins['audio']['decoded_pcm_sha256']):
                raise ValueError('Gallery requires complete synchronized media audits')
            if checked['render_sha256'] != sha(out / 'RENDER.json') or checked['movie_sha256'] != sha(out / 'preview.mp4'):
                raise ValueError('Gallery requires current audits')
            items.append({'method': method, 'label': label, 'song': song,
                'src': str((out / 'preview.mp4').relative_to(root / 'preview_3d_data_v1')),
                'bytes': render_value['bytes']})
    if sum(x['bytes'] for x in items) > 65_000_000:
        raise ValueError('Six-clip preview budget exceeded')
    html = TEMPLATE.read_text().replace('__CLIPS_JSON__', json.dumps(items).replace('<', '\\u003c'))
    path = root / 'preview_3d_data_v1' / 'index.html'
    with path.open('x') as stream:
        stream.write(html)
    return {'path': str(path), 'clips': items, 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-audio', 'smoke', 'render', 'audit', 'gallery', 'check-inputs'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--song', choices=('shpongle', 'cure_for_me'))
    parser.add_argument('--guarded', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare-audio':
        if args.song is None:
            parser.error('--song required')
        value = prepare_audio(args.root, args.song)
    elif args.action == 'gallery':
        value = gallery(args.root)
    else:
        if args.method is None or args.song is None:
            parser.error('--method and --song required')
        if args.action == 'check-inputs':
            *_, pins = load(args.root, args.method, args.song)
            value = {'inputs': pins, 'native_frames': FRAMES, 'start_s': START, 'duration_s': DURATION}
        elif args.action == 'audit':
            value = audit(args.root, args.method, args.song)
        else:
            value = globals()[args.action](args.root, args.method, args.song, args.guarded)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
