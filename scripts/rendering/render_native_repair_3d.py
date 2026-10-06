"""Guarded full-song human review of saved native repair-comparison bodies and emitted DMX.

Camera/optics are the existing audio-representation lit-body presentation.
Only smoke/render create GPU contexts; the enclosing campaign guard owns locks.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.rendering import render_native_dynamics as native
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from scripts.rendering.native_repair_traces import Strip
from scripts.rendering.native_unlimited_emission import apply as apply_unlimited

WIDTH, HEIGHT, FPS = 1280, 1024, 30
VIEW_WIDTH, VIEW_HEIGHT, VIEW_TOP = 640, 600, 84
PLAN_SHA = None  # Set explicitly by --plan-sha256 or bind_plan; never inferred from available files.
METHODS = {'D7_reference': 'Old D7', 'C3': 'Broken native C', 'C4': 'Repaired native C'}
ROLES = {'D7_reference': 'old_D7', 'C3': 'broken_C', 'C4': 'repaired_C'}
CAMERAS = {
    'body': dict(cam_pos=(2.8, 2.15, 3.8), cam_target=(0., 1., -3.5), camera_scale=.20),
    'stage': dict(cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02),
}
FILL_BY_VIEW = {'body': .50, 'stage': .12}
TEMPLATE = Path(__file__).with_name('native_repair_review.html')
sha, read, save = native.sha, native.read, native.save


def display_pose(pose):
    """Existing mapper-local presentation: remove root XZ, retain height/facing."""
    pose = np.asarray(pose)
    if pose.shape[-2:] != (22, 3) or not np.isfinite(pose).all():
        raise ValueError('Finite 22-joint body required')
    return pose - pose[..., 0:1, :] * np.array([1., 0., 1.])


def window(track):
    """Complete original song; only the final fractional video frame is held."""
    duration = float(track['duration_s'])
    if not math.isfinite(duration) or duration <= 0: raise ValueError('Positive full-song duration required')
    frames, ticks = math.ceil(duration*FPS), math.ceil(duration*40)
    if track.get('video_frames') != frames or track.get('body_frames') != frames or track.get('lighting_frames') != ticks:
        raise ValueError('Full-song frame counts differ from the frozen track')
    return {'source_start_s': 0., 'source_duration_s': duration, 'duration_s': frames/FPS,
            'frames': frames, 'lighting_frames': ticks, 'end_padding_s': frames/FPS-duration,
            'playback_rate': 1.}


def frame_indices(track):
    review = window(track)
    body = np.arange(review['frames'])
    ticks = np.minimum(np.rint(body*40/FPS).astype(int), review['lighting_frames']-1)
    return body, ticks


def bind_plan(root, expected_sha256):
    """An external frozen command supplies identity before any source is loaded."""
    global PLAN_SHA
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or sha(Path(root)/'PLAN.json') != expected_sha256):
        raise ValueError('Externally supplied frozen PLAN digest differs')
    PLAN_SHA = expected_sha256
    return PLAN_SHA


def campaign(root, song):
    if PLAN_SHA is None or sha(root / 'PLAN.json') != PLAN_SHA:
        raise ValueError('Campaign PLAN changed')
    plan = read(root / 'PLAN.json')
    review = plan.get('review', {})
    if (review.get('window') != 'full_song' or review.get('fps', FPS) != FPS
            or review.get('width', WIDTH) != WIDTH or review.get('height', HEIGHT) != HEIGHT):
        raise ValueError('Repair renderer requires explicit full_song1280x1024p30 review mode')
    track = next(t for t in plan['tracks'] if t['id'] == song)
    return plan, track


def audio_paths(root, track):
    out = root / 'review_3d_v1' / 'audio' / track['id']
    return out / 'reference.m4a', out / 'REFERENCE.json'


def audio_pins(root, plan, track):
    reference, record = audio_paths(root, track); value = read(record); review = window(track)
    expected = {'plan_sha256': PLAN_SHA, 'source_audio': track['audio'],
        'source_audio_sha256': native.verify_source(plan, track), 'reference': str(reference),
        'reference_sha256': sha(reference), **{k: review[k] for k in
        ('source_start_s','source_duration_s','duration_s','end_padding_s','playback_rate')}}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('Prepared full-song audio binding changed')
    inherited = {}
    if value.get('inherited_reference'):
        old, old_record = Path(value['inherited_reference']), Path(value['inherited_receipt'])
        if sha(old) != value['inherited_reference_sha256'] or sha(old_record) != value['inherited_receipt_sha256']:
            raise ValueError('Inherited full-song reference changed')
        inherited = {k: value[k] for k in ('inherited_reference_sha256','inherited_receipt_sha256')}
    return {'audio_record_sha256': sha(record), 'audio_reference_sha256': expected['reference_sha256'],
            'source_audio_sha256': expected['source_audio_sha256'], 'decoded_pcm_sha256': value['decoded_pcm_sha256'],
            **review, **inherited}


def prepare_audio(root, song):
    """Reuse an exact full reference or encode the complete unchanged-speed song."""
    import soundfile as sf
    plan, track = campaign(root, song); review = window(track)
    source_sha = native.verify_source(plan, track); reference, record = audio_paths(root, track)
    if record.exists():
        audio_pins(root, plan, track)
        if hashlib.sha256(native.base.decoded_audio(reference)).hexdigest() != read(record)['decoded_pcm_sha256']:
            raise ValueError('Prepared reference PCM changed')
        return read(record)
    if reference.exists(): raise ValueError('Unreceipted reference retained')
    info = sf.info(track['audio'])
    if abs(info.frames/info.samplerate-review['source_duration_s']) > 1/info.samplerate:
        raise ValueError('Frozen duration does not cover the original complete waveform')
    native.resource_guard(root, plan); reference.parent.mkdir(parents=True, exist_ok=True)
    old_record = root.parent/'audio'/song/'REFERENCE.json'; old = read(old_record) if old_record.exists() else None
    compatible = bool(old and old.get('source_audio') == track['audio'] and old.get('source_audio_sha256') == source_sha
        and old.get('source_start_s') == 0 and old.get('playback_rate') == 1.
        and old.get('duration_s') == review['duration_s']
        and abs(old.get('source_duration_s', -1)-review['source_duration_s']) <= 1/info.samplerate
        and abs(old.get('end_padding_s', -1)-review['end_padding_s']) <= 1e-9)
    reuse = {}
    if compatible:
        original = Path(old['reference'])
        if sha(original) != old['reference_sha256'] or hashlib.sha256(native.base.decoded_audio(original)).hexdigest() != old['decoded_pcm_sha256']:
            raise ValueError('Compatible inherited reference has changed bytes or PCM')
        os.link(original, reference)
        reuse = {'creation': 'Hardlink exact full-song original-speed AAC reference; no reencoding.',
            'inherited_reference': str(original), 'inherited_reference_sha256': old['reference_sha256'],
            'inherited_receipt': str(old_record), 'inherited_receipt_sha256': sha(old_record),
            'original_encoding_command': old.get('command', old.get('original_encoding_command'))}
    else:
        partial = reference.with_name('reference.partial.m4a'); duration = review['duration_s']
        command = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',track['audio'],
            '-map','0:a:0','-vn','-c:a','aac','-b:a','128k','-ac','2','-ar','44100',
            '-af',f'apad=whole_dur={duration:.12f},atrim=end={duration:.12f},asetpts=PTS-STARTPTS',
            '-t',str(duration),'-movflags','+faststart',str(partial)]
        subprocess.run(command, check=True); os.link(partial, reference); partial.unlink()
        reuse = {'creation': 'Complete original-speed audio encoded once; final fractional-frame silence padding only.',
                 'command': command, 'compatible_inherited_full_reference_found': False}
    value = {'plan_sha256': PLAN_SHA, 'source_audio': track['audio'], 'source_audio_sha256': source_sha,
        'reference': str(reference), 'reference_sha256': sha(reference),
        'decoded_pcm_sha256': hashlib.sha256(native.base.decoded_audio(reference)).hexdigest(),
        **review, **reuse}
    save(record, value); return value


def load(root, method, song):
    plan, track = campaign(root, song)
    pair = apply_unlimited(root, method, song, native.load_pair(root, plan, method, track))
    source = DanceResult.load(root / 'bodies' / method / song / 'body.npz')
    body_indices, ticks = frame_indices(track)
    body = display_pose(source.world[body_indices])
    if not np.isfinite(body).all() or ticks[-1] >= len(pair.times):
        raise ValueError('Invalid review body/clock')
    pins = {'plan_sha256': PLAN_SHA, **pair.pins,
            'audio': audio_pins(root, plan, track),
            'renderer_sha256': sha(Path(__file__)), 'template_sha256': sha(TEMPLATE)}
    strip=Strip(root,method,song)
    pins['trace_inputs']=strip.data['inputs']
    pins['trace_code_sha256']=sha(Path(__file__).with_name('native_repair_traces.py'))
    return plan, track, pair, body, ticks, pins


def folder(root, method, song):
    return root / 'review_3d_v1' / method / song


@contextmanager
def renderers(root, patch, guarded, track=None):
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
            'pose_transform': 'root XZ removed only for viewing; saved unfiltered native pose and timestamps retained',
            'max_native_body_to_nearest_dmx_offset_s': float(np.max(np.abs(
                frame_indices(track)[0] / FPS - frame_indices(track)[1] / 40))),
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
    draw.text((1020, 40), 'UNLIMITED SIMULATION', font=small, fill=(242, 190, 119))
    draw.text((16, 66), 'LIT SOURCE BODY — THIS SONG', font=small, fill=(181, 201, 224))
    draw.text((656, 66), 'SAME EMITTED LIGHTS — WIDE STAGE', font=small, fill=(181, 201, 224))
    draw.line((639, VIEW_TOP, 639, VIEW_TOP + VIEW_HEIGHT), fill=(50, 65, 84), width=1)
    caption = strip.motion_label(stamp)
    draw.text((16, 688), caption,
              font=small, fill=(242, 190, 119))
    draw.text((16, 705), 'Original audio speed; fixture motion caps OFF. Root XZ removed for viewing. Constant viewer fill .50 / .12 is separate from DMX.',
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
    with renderers(root, pair.patch, guarded, track) as (rs, config):
        out.mkdir(parents=True, exist_ok=False)
        count = window(track)['frames']
        for frame in sorted(set([0, min(18*FPS,count-1), min(90*FPS,count-1), min(135*FPS,count-1), count-1])):
            stamp = frame/FPS
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
            samples.append({'source_s': stamp, 'body_frame': int(frame),
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


def encode_command(path, audio, track):
    review = window(track)
    # Six complete tracks: approximately220.6MB decimal at1.1Mb/s +128kb/s audio.
    return ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x1024', '-r', '30', '-i', 'pipe:0',
        '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
        '-frames:v', str(review['frames']), '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
        '-b:v', '1100k', '-maxrate', '1250k', '-bufsize', '2500k', '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-c:a', 'copy', '-t', str(review['duration_s']),
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
    review = window(track)
    command = encode_command(partial, reference, track)
    strip=Strip(root,method,song)
    started = time.monotonic()
    with renderers(root, pair.patch, guarded, track) as (rs, config):
        with (out / 'encode.log').open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(ticks):
                    controls = controls_at(pair, tick)
                    views = [r.render_controls(*controls, body[frame]) for r in rs]
                    encoder.stdin.write(compose(views, method, track, frame / FPS, strip).tobytes())
                    if frame % 300 == 0:
                        native.resource_guard(root, plan)
                        print('PREVIEW_3D', method, song, frame, review['frames'], flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encode failed; partial retained')
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
    clip_budget = math.ceil(review['duration_s']*(1_250_000+128_000)/8)+2**20
    if partial.stat().st_size > clip_budget:
        raise ValueError('Per-clip delivery budget exceeded; partial retained')
    os.link(partial, movie)
    partial.unlink()
    value = {'inputs': pins, 'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
        'method': method, 'song': song, **review, 'audio_speed': 1., 'body_fps': 30, 'dmx_fps': 40,
        'method_label': METHODS[method], 'role': ROLES[method], 'recomposed': False, 'retimed': False,
        'posthoc_motion_correction': False,
        'smoke_sha256': smoke_sha, 'status': 'experimental_native_review',
        'baseline_promotion': False,
        'command': command, 'elapsed_s': time.monotonic() - started, 'traces':strip.evidence(), **config}
    save(record, value)
    return value


def audit(root, method, song):
    plan, track, _, _, _, pins = load(root, method, song)
    out = folder(root, method, song)
    review = window(track)
    value = read(out / 'RENDER.json')
    movie = out / 'preview.mp4'
    if value['inputs'] != pins or value['sha256'] != sha(movie) or any(value.get(k) != v for k,v in review.items()):
        raise ValueError('Preview or source authentication failed')
    require_smoke(out, pins)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(movie)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    if (video['width'], video['height'], video['r_frame_rate'], int(video['nb_read_frames'])) != (WIDTH, HEIGHT, '30/1', review['frames']):
        raise ValueError('Preview dimensions/clock/frame count failed')
    if abs(float(video['duration']) - review['duration_s']) > 1/FPS or abs(float(audio['duration']) - review['duration_s']) > .05:
        raise ValueError('Video/audio duration mismatch')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)], text=True))['frames']
    times = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    if len(times) != review['frames'] or not np.allclose(times, np.arange(review['frames'])/FPS, rtol=0, atol=1e-5):
        raise ValueError('Incomplete or nonuniform decoded video timestamps')
    if abs(float(audio.get('start_time', 0))) > 1/44100:
        raise ValueError('Audio start synchronization failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-threads', '2',
        '-i', str(movie), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
    reference, reference_record = audio_paths(root, track)
    reference_pcm = hashlib.sha256(native.base.decoded_audio(reference)).hexdigest()
    delivered_pcm = hashlib.sha256(native.base.decoded_audio(movie)).hexdigest()
    if delivered_pcm != reference_pcm or delivered_pcm != read(reference_record)['decoded_pcm_sha256']:
        raise ValueError('Delivered PCM differs from authenticated complete-song reference')
    report = {'render_sha256': sha(out / 'RENDER.json'), 'movie_sha256': sha(movie), 'inputs': pins,
        'full_software_AV_decode': True, 'decoded_frames': len(times), 'pts_uniform': True,
        'pcm_sha256': delivered_pcm, 'audio_matches_prepared_reference': True,
        **review, 'audio_speed': 1., 'probe': probe}
    audit_path = out / 'AUDIT.json'
    if audit_path.exists():
        old = read(audit_path)
        if old != report:
            raise ValueError('Existing audit differs')
    else:
        save(audit_path, report)
    return report


def review_evidence(root, method, song, pins=None):
    """Display root's sealed verdict; never infer repair success from a trace."""
    path = root/'reports'/'review'/f'{method}_{song}.json'
    if not path.exists():
        return {'status': 'pending', 'message': 'Root evaluation evidence is not available yet.'}
    seal = path.with_name(path.name+'.sha256.json')
    if not seal.exists() or read(seal).get('sha256') != sha(path):
        return {'status': 'unverified', 'message': 'Evaluation exists but is not sealed; no acceptance claim.', 'path': str(path)}
    value = read(path)
    if pins is None:
        return {'status': 'pending', 'message': 'Evaluation cannot be tied to a verified displayed movie yet.', 'path': str(path)}
    expected = {'plan_sha256': PLAN_SHA, 'body_sha256': pins['body_sha256'], 'timeline_sha256': pins['timeline_sha256']}
    paths = {'plan_sha256': root/'PLAN.json', 'body_sha256': root/'bodies'/method/song/'body.npz',
             'timeline_sha256': root/'timelines'/method/song/'timeline.npz'}
    if any(value.get(key, value.get('inputs', {}).get(str(paths[key]))) != digest for key, digest in expected.items()):
        return {'status': 'stale', 'message': 'Evaluation does not identify these displayed inputs; no acceptance claim.', 'path': str(path)}
    return {'status': 'authenticated', 'path': str(path), 'sha256': sha(path),
            'summary': value.get('summary') if isinstance(value.get('summary'), str) else None,
            'report': value}


def diagnostic_graphs(root, items):
    """Optional CPU zoom of the same raw full-song data; never another movie."""
    result = []
    for row in items:
        if row['status'] not in ('verified', 'failure_diagnostic') or row['source_duration_s'] < 150:
            continue
        method, song = row['method'], row['song']
        strip = Strip(root, method, song, interval=(90., 150.))
        directory = root/'review_3d_v1'/'graphs'/method/song
        image_path, record = directory/'raw_90_150.png', directory/'GRAPH.json'
        if record.exists():
            saved = read(record)
            if saved['inputs'] != strip.data['inputs'] or sha(image_path) != saved['sha256']:
                raise ValueError('Existing diagnostic graph identity changed')
        else:
            directory.mkdir(parents=True, exist_ok=False)
            strip.image.save(image_path)
            save(record, {'inputs': strip.data['inputs'], 'sha256': sha(image_path),
                'interval_s': [90.,150.], 'scales': strip.data['scales'],
                'scale_scope': 'Same full-song shared scales as the video', 'evidence': strip.evidence()})
        result.append({'song': song, 'method': method, 'title': METHODS[method]+' — raw90–150s detail',
            'src': 'data:image/png;base64,'+base64.b64encode(image_path.read_bytes()).decode(),
            'path': str(image_path), 'sha256': sha(image_path), 'interval_s': [90.,150.]})
    return result


def gallery(root, revision):
    """Publish one stable page; preserve each pending/final publication separately."""
    if not revision or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in revision):
        raise ValueError('Safe explicit gallery revision required')
    plan, _ = campaign(root, 'shpongle')
    items, variants, total = [], [], 0
    for song in ('shpongle', 'cure_for_me'):
        track = next(t for t in plan['tracks'] if t['id'] == song); full_window = window(track)
        for method, label in METHODS.items():
            index = f'{len(items)+1:02d}'; out = folder(root, method, song)
            body = root/'bodies'/method/song/'body.npz'
            row = {'method': method, 'label': label, 'role': ROLES[method], 'song': song,
                   **full_window, 'video_frames': full_window['frames'],
                   'download_name': f'{song}_{ROLES[method]}_full.mp4',
                   'status': 'awaiting_generation' if not body.exists() else 'awaiting_verification',
                   'message': 'No verified video for this role yet. No substitute motion is shown.',
                   'evidence': review_evidence(root, method, song)}
            media = {'id': index, 'status': 'planned', 'path': str(out/'preview.mp4'),
                     'download_name': f'{song}_{ROLES[method]}_full.mp4'}
            render_path, audit_path = out/'RENDER.json', out/'AUDIT.json'
            if render_path.exists() and audit_path.exists():
                try:
                    _, _, _, _, _, pins = load(root, method, song)
                    render_value, checked = read(render_path), read(audit_path)
                    if (checked.get('inputs') != pins or render_value.get('inputs') != pins or checked.get('decoded_frames') != full_window['frames']
                            or any(checked.get(k) is not True for k in ('full_software_AV_decode','pts_uniform','audio_matches_prepared_reference'))
                            or checked.get('pcm_sha256') != pins['audio']['decoded_pcm_sha256']
                            or checked.get('render_sha256') != sha(render_path)
                            or any(checked.get(k) != v or render_value.get(k) != v for k,v in full_window.items())
                            or checked.get('movie_sha256') != sha(out/'preview.mp4')
                            or render_value.get('sha256') != checked['movie_sha256']):
                        raise ValueError('Complete synchronized media authentication failed')
                    evidence = review_evidence(root, method, song, pins)
                    report = evidence.get('report', {})
                    failed = report.get('failure_diagnostic') is True or report.get('status') == 'failure_diagnostic'
                    row.update(status='failure_diagnostic' if failed else 'verified',
                        src=f'/media/{index}.mp4?v={checked["movie_sha256"]}', sha256=checked['movie_sha256'],
                        bytes=render_value['bytes'], evidence=evidence,
                        message='Verified media; scientific and artistic qualification remains in the evidence.')
                    media.update(status='complete', sha256=checked['movie_sha256'], bytes=render_value['bytes'])
                    total += render_value['bytes']
                except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
                    row.update(status='broken', message='Saved delivery rejected: '+str(error))
            items.append(row); variants.append(media)
    if total > plan.get('review', {}).get('maximum_total_bytes', 260_000_000):
        raise ValueError('Full-song six-video delivery budget exceeded')
    prior = plan.get('prior_review_evidence', [])
    if not isinstance(prior, list): raise ValueError('Prior evidence must be a list')
    prior = [dict(entry) for entry in prior]
    for entry in prior:
        if 'path' in entry:
            path = Path(entry['path'])
            expected = entry.get('sha256', plan.get('files', {}).get(str(path)))
            if not expected or sha(path) != expected: raise ValueError('Prior evidence is not frozen by PLAN')
            entry['sha256'] = expected
    graphs = diagnostic_graphs(root, items)
    manifest = {'plan_sha256': PLAN_SHA, 'variants': variants, 'clips': items, 'prior_evidence': prior,
                'baseline_promoted': False, 'revision': revision, 'roles': dict(ROLES), 'tracks': plan['tracks'],
                'diagnostic_graphs': [{k:v for k,v in g.items() if k!='src'} for g in graphs]}
    html = TEMPLATE.read_text().replace('__CLIPS_JSON__', json.dumps(items).replace('<', '\\u003c'))
    html = html.replace('__PRIOR_EVIDENCE_JSON__', json.dumps(prior).replace('<', '\\u003c'))
    html = html.replace('__TRACKS_JSON__', json.dumps(plan['tracks']).replace('<', '\\u003c'))
    html = html.replace('__DIAGNOSTIC_GRAPHS_JSON__', json.dumps(graphs).replace('<', '\\u003c'))
    dest = root/'review_3d_v1'; archive = dest/'revisions'/revision
    archive.mkdir(parents=True, exist_ok=False)
    save(archive/'MANIFEST.json', manifest)
    with (archive/'index.html').open('x') as stream: stream.write(html)
    # Stable URL publication changes only the small current index/manifest;
    # earlier revisions and every video/audit/source remain immutable.
    for name in ('MANIFEST.json', 'index.html'):
        temporary = dest/(name+'.publishing')
        with temporary.open('xb') as stream: stream.write((archive/name).read_bytes())
        os.replace(temporary, dest/name)
    return {'path': str(dest/'index.html'), 'revision': revision, 'clips': items,
            'verified_media': sum(v['status']=='complete' for v in variants), 'sha256': sha(dest/'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-audio', 'smoke', 'render', 'audit', 'gallery', 'check-inputs'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--revision', default='initial')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--song', choices=('shpongle', 'cure_for_me'))
    parser.add_argument('--guarded', action='store_true')
    args = parser.parse_args()
    bind_plan(args.root, args.plan_sha256)
    if args.action == 'prepare-audio':
        if args.song is None:
            parser.error('--song required')
        value = prepare_audio(args.root, args.song)
    elif args.action == 'gallery':
        value = gallery(args.root, args.revision)
    else:
        if args.method is None or args.song is None:
            parser.error('--method and --song required')
        if args.action == 'check-inputs':
            *_, pins = load(args.root, args.method, args.song)
            _, track = campaign(args.root, args.song)
            value = {'inputs': pins, **window(track)}
        elif args.action == 'audit':
            value = audit(args.root, args.method, args.song)
        else:
            value = globals()[args.action](args.root, args.method, args.song, args.guarded)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
