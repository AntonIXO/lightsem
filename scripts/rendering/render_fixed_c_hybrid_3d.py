"""Guarded C4-only90–150s fixed-repaired-C+D4-max+D6 human-check previews.

prepare-audio, check-inputs and audit are CPU-only. Only smoke/render open GPU
contexts, after authenticating THIS executed script and the frozen command.
This renderer consumes saved bodies and emitted universes; it generates none.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from lightsem.experiments.dynamic_dance.contracts import DanceResult
from scripts.rendering import render_native_positive_3d as mechanics
from scripts.rendering import render_native_dynamics as native
from scripts.rendering.fixed_c_hybrid_traces import Strip, validate_plan, fixed_source
from scripts.rendering.native_unlimited_emission import apply as apply_unlimited

WIDTH, HEIGHT, FPS = 1280, 1024, 30
VIEW_WIDTH, VIEW_HEIGHT, VIEW_TOP = 640, 600, 84
START, DURATION, FRAMES = 90, 60, 1800
METHODS = {'C4': 'Fixed C + D4-max + D6 | HUMAN CHECK / DIAGNOSTIC'}
CAMERAS = mechanics.CAMERAS
FILL_BY_VIEW = mechanics.FILL_BY_VIEW
sha, read, save = native.sha, native.read, native.save
display_pose, frame_indices = mechanics.display_pose, mechanics.frame_indices
controls_at, fonts, encode_command = mechanics.controls_at, mechanics.fonts, mechanics.encode_command


def campaign(root, song, plan_sha256):
    return validate_plan(Path(root), song, plan_sha256)


def source_files():
    """Bind the real wrapper, reused mechanics, traces and rendered implementations."""
    workspace = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(),
             Path(mechanics.__file__).resolve(), Path(native.__file__).resolve(),
             Path(native.base.__file__).resolve(), Path(native.drawing.__file__).resolve(),
             workspace / 'scripts/rendering/fixed_c_hybrid_traces.py',
             workspace / 'scripts/rendering/native_repair_traces.py',
             workspace / 'scripts/rendering/native_positive_traces.py',
             workspace / 'scripts/rendering/native_unlimited_emission.py',
             workspace / 'lightsem/experiments/dynamic_dance/contracts.py',
             workspace / 'lightsem/conductor/body_transfer.py',
             workspace / 'lightsem/sim/lit_body_renderer.py',
             workspace / 'lightsem/sim/body_stage_renderer.py',
             workspace / 'lightsem/sim/renderer.py',
             workspace / 'lightsem/sim/shaders.py',
             workspace / 'lightsem/dmx/fixtures.py',
             workspace / 'scripts/motion/native_dynamics_guard.py',
             workspace / 'scripts/motion/dynamic_dance_guard.py']
    return {str(path): sha(path) for path in dict.fromkeys(paths)}


def audio_paths(root, track):
    out = Path(root) / 'preview_3d_v1' / 'audio' / track['id']
    return out / 'reference.m4a', out / 'REFERENCE.json'


def audio_pins(root, plan, track, plan_sha256):
    reference, record = audio_paths(root, track)
    value = read(record)
    expected = {'plan_sha256': plan_sha256, 'source_audio': track['audio'],
                'source_audio_sha256': native.verify_source(plan, track),
                'reference': str(reference), 'reference_sha256': sha(reference),
                'source_start_s': START, 'duration_s': DURATION, 'playback_rate': 1.}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('Prepared audio binding changed')
    inherited = Path(value['inherited_reference'])
    inherited_record = Path(value['inherited_receipt'])
    if (sha(inherited) != value['inherited_reference_sha256']
            or sha(inherited_record) != value['inherited_receipt_sha256']):
        raise ValueError('Inherited immutable audio binding changed')
    return {'audio_record_sha256': sha(record), 'audio_reference_sha256': sha(reference),
            'inherited_reference_sha256': value['inherited_reference_sha256'],
            'inherited_receipt_sha256': value['inherited_receipt_sha256'],
            'source_audio_sha256': expected['source_audio_sha256'],
            'decoded_pcm_sha256': value['decoded_pcm_sha256'],
            'source_start_s': START, 'duration_s': DURATION, 'playback_rate': 1.}


def prepare_audio(root, song, plan_sha256):
    """CPU-only hardlink of authenticated original-speed90–150s audio."""
    root = Path(root)
    plan, track = campaign(root, song, plan_sha256)
    source_sha = native.verify_source(plan, track)
    reference, record = audio_paths(root, track)
    if record.exists():
        audio_pins(root, plan, track, plan_sha256)
        if hashlib.sha256(native.base.decoded_audio(reference)).hexdigest() != read(record)['decoded_pcm_sha256']:
            raise ValueError('Prepared reference PCM changed')
        return read(record)
    if reference.exists():
        raise ValueError('Unreceipted reference retained')
    old_folder = root.parent / 'preview_3d_unlimited_v1' / 'audio' / song
    old_record = old_folder / 'REFERENCE.json'
    old = read(old_record)
    old_reference = Path(old['reference'])
    expected = {'source_audio': track['audio'], 'source_audio_sha256': source_sha,
                'source_start_s': START, 'duration_s': DURATION, 'playback_rate': 1.}
    if (any(old.get(key) != item for key, item in expected.items())
            or sha(old_reference) != old['reference_sha256']):
        raise ValueError('Reusable audio has a different source, crop, speed or bytes')
    if hashlib.sha256(native.base.decoded_audio(old_reference)).hexdigest() != old['decoded_pcm_sha256']:
        raise ValueError('Reusable reference PCM differs from its authenticated receipt')
    native.resource_guard(root, plan)
    reference.parent.mkdir(parents=True, exist_ok=True)
    os.link(old_reference, reference)
    value = {'plan_sha256': plan_sha256, **expected, 'reference': str(reference),
             'reference_sha256': sha(reference), 'decoded_pcm_sha256': old['decoded_pcm_sha256'],
             'creation': 'hardlink of immutable original-speed AAC reference; no reencoding',
             'inherited_reference': str(old_reference),
             'inherited_reference_sha256': old['reference_sha256'],
             'inherited_receipt': str(old_record), 'inherited_receipt_sha256': sha(old_record),
             'original_encoding_command': old.get('command', old.get('original_encoding_command'))}
    save(record, value)
    return value


def admission(root, song):
    path = Path(root) / 'reports' / f'{song}_ADMISSION.json'
    value = read(path)
    failures = value.get('failure_reasons')
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        raise ValueError('Admission must contain an explicit failure_reasons list')
    return {'path': str(path), 'sha256': sha(path), 'failure_reasons': failures,
            'status': 'FAILURE DIAGNOSTIC' if failures else 'HUMAN CHECK / DIAGNOSTIC',
            'baseline_promotion': False}


def load(root, method, song, plan_sha256):
    root = Path(root)
    if method != 'C4':
        raise ValueError('Only C4 is authorized for these two human-check clips')
    plan, track = campaign(root, song, plan_sha256)
    fixed_source(plan, song)
    pair = apply_unlimited(root, method, song, native.load_pair(root, plan, method, track))
    source = DanceResult.load(root / 'bodies' / method / song / 'body.npz')
    body_indices, ticks = frame_indices()
    body = display_pose(source.world[body_indices])
    if not np.isfinite(body).all() or ticks[-1] >= len(pair.times):
        raise ValueError('Invalid diagnostic body/clock')
    strip = Strip(root, method, song, plan_sha256)
    decision = admission(root, song)
    pins = {'plan_sha256': plan_sha256, **pair.pins,
            'audio': audio_pins(root, plan, track, plan_sha256),
            'source_code_sha256': source_files(),
            'renderer_sha256': sha(Path(__file__).resolve()),
            'trace_inputs': strip.data['inputs'], 'role_artifacts': strip.data['artifacts'],
            'admission': decision}
    return plan, track, pair, body, ticks, pins, strip


def folder(root, method, song):
    return Path(root) / 'preview_3d_v1' / method / song


@contextmanager
def renderers(root, patch, guarded, plan_sha256):
    if not guarded:
        raise ValueError('GPU rendering requires --guarded and the sequential curator')
    from scripts.motion.native_dynamics_guard import authenticate_guarded_job
    # Authenticate the actual entrypoint; do not call the imported renderer guard.
    receipt = authenticate_guarded_job(str(Path(__file__).resolve()))
    command = receipt['identity']['command']
    for option, expected in (('--root', str(Path(root).resolve())), ('--plan-sha256', plan_sha256),
                             ('--method', 'C4')):
        if option not in command or command[command.index(option)+1] != expected:
            raise ValueError('Guard command does not bind ' + option)
    from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
    from lightsem.sim.renderer import RenderConfig
    created, configs = [], []
    try:
        for name, camera in CAMERAS.items():
            config = RenderConfig(width=VIEW_WIDTH, height=VIEW_HEIGHT, fps=FPS, cam_orbit=0.,
                haze=.25, intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                require_gpu=True, **camera)
            created.append(LitBodyStageRenderer(patch, config, body_offset=(0., 0., -3.5),
                           body_scale=1., neutral_fill=FILL_BY_VIEW[name]))
            configs.append(asdict(config))
        yield created, {'configs': configs, 'fill_by_view': dict(FILL_BY_VIEW),
            'presentation_by_view': {name: renderer.presentation_config()
                                     for name, renderer in zip(CAMERAS, created)},
            'pose_transform': 'rootXZ removed only for viewing; raw saved world motion/time retained in measurements and artifacts',
            'max_native_body_to_nearest_dmx_offset_s': float(np.max(np.abs(
                 frame_indices()[0]/FPS-frame_indices()[1]/40))),
            'gpu_receipt': receipt}
    finally:
        for renderer in reversed(created):
            renderer.close()


def compose(views, method, track, stamp, strip, decision):
    picture = Image.new('RGB', (WIDTH, HEIGHT), (7, 10, 16))
    for x, view in zip((0, VIEW_WIDTH), views):
        if view.shape != (VIEW_HEIGHT, VIEW_WIDTH, 3):
            raise ValueError('Unexpected native view dimensions')
        picture.paste(Image.fromarray(view), (x, VIEW_TOP))
    draw = ImageDraw.Draw(picture)
    large, regular, small = fonts()
    draw.text((16, 6), METHODS[method], font=large, fill=(236, 241, 249))
    draw.text((16, 34), track['title'] + ' | P2 | actual emitted DMX | historical brightness | limits OFF',
              font=regular, fill=(185, 205, 226))
    draw.text((1100, 35), f'{stamp:06.2f} s', font=regular, fill=(214, 226, 241))
    failures = decision['failure_reasons']
    if failures:
        text = 'FAILURE DIAGNOSTIC: ' + '; '.join(failures)
        draw.rectangle((0, 56, WIDTH, 82), fill=(77, 20, 21))
        # Every reason remains in the JSON; long banners scroll by cycling lines.
        chunks, current = [], ''
        for word in text.split():
            trial = (current + ' ' + word).strip()
            if draw.textlength(trial, font=small) > WIDTH-32 and current:
                chunks.append(current); current = word
            else:
                current = trial
        if current:
            chunks.append(current)
        line = chunks[int((stamp-START)//4) % len(chunks)]
        draw.text((16, 61), line, font=small, fill=(255, 221, 210))
    else:
        draw.text((16, 64), 'OWN-SONG RECOMPOSED / RETIMED BODY — HUMAN CHECK', font=small, fill=(181, 201, 224))
        draw.text((656, 64), 'SAME EMITTED LIGHTS — WIDE STAGE', font=small, fill=(181, 201, 224))
    draw.line((639, VIEW_TOP, 639, VIEW_TOP+VIEW_HEIGHT), fill=(50, 65, 84))
    draw.text((16, 686), strip.pacing_label(stamp), font=small, fill=(242, 190, 119))
    draw.text((16, 704), 'Original audio speed. RootXZ view transform only. Constant viewer fill .50 / .12 is separate from DMX. No native-learning claim.',
              font=small, fill=(143, 164, 189))
    picture.paste(strip.frame(stamp), (0, 720))
    return np.asarray(picture)


def smoke(root, method, song, guarded, plan_sha256):
    plan, track, pair, body, ticks, pins, strip = load(root, method, song, plan_sha256)
    out = folder(root, method, song) / 'smoke'
    if out.exists():
        raise ValueError('Preserve prior smoke; inspect it or use a new source revision')
    native.resource_guard(root, plan)
    files, samples, actual_files = {}, [], []
    with renderers(root, pair.patch, guarded, plan_sha256) as (rs, config):
        out.mkdir(parents=True, exist_ok=False)
        for stamp in (90., 105., 120., 135., 149.):
            frame = round((stamp-START)*FPS)
            controls = controls_at(pair, ticks[frame])
            views, metrics = [], []
            for index, renderer in enumerate(rs):
                diagnostic = renderer.render_diagnostics(*controls, body[frame])
                repeated = renderer.render_controls(*controls, body[frame])
                if not np.array_equal(repeated, diagnostic['actual']):
                    raise ValueError('Frozen body/DMX produced autonomous pixels')
                if diagnostic['metrics']['body_visible_pixels'] == 0:
                    raise ValueError('Source body invisible in camera ' + str(index))
                views.append(diagnostic['actual']); metrics.append(diagnostic['metrics'])
                for key in ('lights_off', 'body_mask', 'direct_body_light'):
                    path = out / f'{stamp:06.2f}_view{index}_{key}.png'
                    Image.fromarray(diagnostic[key]).save(path)
                    files[str(path)] = sha(path)
            path = out / f'{stamp:06.2f}_actual.png'
            Image.fromarray(compose(views, method, track, stamp, strip, pins['admission'])).save(path)
            files[str(path)] = sha(path); actual_files.append(str(path))
            samples.append({'source_s': stamp, 'body_frame': int(START*FPS+frame),
                            'dmx_tick': int(ticks[frame]), 'frozen_pixels_exact': True, 'views': metrics})
    files.update(strip.export(out / 'TRACE.npz'))
    value = {'inputs': pins, 'files': files, 'actual_files_for_root_inspection': actual_files,
             'samples': samples, 'traces': strip.evidence(), 'diagnostic': True,
             'physical_failure_reasons': pins['admission']['failure_reasons'], **config}
    save(out / 'SMOKE.json', value)
    return value


def require_smoke(out, pins):
    record = Path(out) / 'smoke' / 'SMOKE.json'
    value = read(record)
    inspection = read(record.with_name('INSPECTED.json'))
    if (value['inputs'] != pins or inspection.get('inspected') is not True
            or inspection.get('inspected_by') != 'root'
            or inspection.get('smoke_sha256') != sha(record)):
        raise ValueError('Root inspection of matching actual GPU smoke required')
    inspected = inspection.get('actual_files', [])
    if set(inspected) != set(value['actual_files_for_root_inspection']):
        raise ValueError('Root must inspect every saved actual smoke image for this song')
    if any(sha(path) != digest for path, digest in value['files'].items()):
        raise ValueError('Smoke image or trace artifacts changed')
    return sha(record), sha(record.with_name('INSPECTED.json'))


def render(root, method, song, guarded, plan_sha256):
    plan, track, pair, body, ticks, pins, strip = load(root, method, song, plan_sha256)
    out = folder(root, method, song)
    smoke_sha, inspection_sha = require_smoke(out, pins)
    movie, record = out / 'preview.mp4', out / 'RENDER.json'
    if movie.exists() or record.exists():
        raise ValueError('Existing preview retained; audit it instead')
    native.resource_guard(root, plan)
    partial = out / 'preview.partial.mp4'
    reference, _ = audio_paths(root, track)
    command = encode_command(partial, reference)
    started = time.monotonic()
    with renderers(root, pair.patch, guarded, plan_sha256) as (rs, config):
        with (out / 'encode.log').open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(ticks):
                    controls = controls_at(pair, tick)
                    views = [r.render_controls(*controls, body[frame]) for r in rs]
                    encoder.stdin.write(compose(views, method, track, START+frame/FPS,
                                               strip, pins['admission']).tobytes())
                    if frame % 300 == 0:
                        native.resource_guard(root, plan)
                        print('FIXED_C_HYBRID_3D', method, song, frame, FRAMES, flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encode failed; partial retained')
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill(); encoder.wait()
                raise
    if partial.stat().st_size > 11_000_000:
        raise ValueError('Per-clip delivery budget exceeded; partial retained')
    os.link(partial, movie); partial.unlink()
    trace_files = strip.export(out / 'TRACE.npz')
    strip.image.save(out / 'RAW_PAIR_TRACES.png')
    trace_files[str(out / 'RAW_PAIR_TRACES.png')] = sha(out / 'RAW_PAIR_TRACES.png')
    value = {'inputs': pins, 'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
             'method': method, 'song': song, 'source_start_s': START, 'duration_s': DURATION,
             'frames': FRAMES, 'audio_speed': 1., 'body_fps': 30, 'dmx_fps': 40,
             'method_label': METHODS[method], 'recomposed': True, 'external_D6_retimed': True,
             'smoke_sha256': smoke_sha, 'root_inspection_sha256': inspection_sha,
             'status': pins['admission']['status'], 'physical_failure_reasons': pins['admission']['failure_reasons'],
             'baseline_promotion': False, 'native_learning_claim': False,
             'fixture_motion_caps': 'OFF', 'command': command,
             'elapsed_s': time.monotonic()-started, 'trace_files': trace_files,
             'traces': strip.evidence(), **config}
    save(record, value)
    return value


def audit(root, method, song, plan_sha256):
    import json
    plan, track, _, _, _, pins, _ = load(root, method, song, plan_sha256)
    out = folder(root, method, song)
    value = read(out / 'RENDER.json'); movie = out / 'preview.mp4'
    if value['inputs'] != pins or value['sha256'] != sha(movie):
        raise ValueError('Preview or source authentication failed')
    smoke_sha, inspection_sha = require_smoke(out, pins)
    if value['smoke_sha256'] != smoke_sha or value['root_inspection_sha256'] != inspection_sha:
        raise ValueError('Render smoke inspection binding changed')
    if any(sha(path) != digest for path, digest in value['trace_files'].items()):
        raise ValueError('Rendered trace evidence changed')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                      '-show_streams', '-show_format', '-of', 'json', str(movie)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    if (video['width'], video['height'], video['r_frame_rate'], int(video['nb_read_frames'])) != (WIDTH, HEIGHT, '30/1', FRAMES):
        raise ValueError('Preview dimensions, clock or frame count failed')
    if abs(float(video['duration'])-DURATION) > 1/FPS or abs(float(audio['duration'])-DURATION) > .05:
        raise ValueError('Video/audio duration mismatch')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                       '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
                       '-of', 'json', str(movie)], text=True))['frames']
    times = np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
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
        raise ValueError('Delivered PCM differs from authenticated original-speed90–150s reference')
    report = {'render_sha256': sha(out / 'RENDER.json'), 'movie_sha256': sha(movie), 'inputs': pins,
              'full_software_AV_decode': True, 'decoded_frames': len(times), 'pts_uniform': True,
              'pcm_sha256': delivered_pcm, 'audio_matches_prepared_reference': True,
              'source_start_s': START, 'duration_s': DURATION, 'audio_speed': 1.,
              'trace_files': value['trace_files'], 'physical_failure_reasons': value['physical_failure_reasons'],
              'diagnostic': True, 'baseline_promotion': False, 'probe': probe}
    path = out / 'AUDIT.json'
    if path.exists():
        if read(path) != report:
            raise ValueError('Existing audit differs')
    else:
        save(path, report)
    return report


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-audio', 'smoke', 'render', 'audit', 'check-inputs'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--song', choices=('shpongle', 'cure_for_me'), required=True)
    parser.add_argument('--guarded', action='store_true')
    args = parser.parse_args()
    if args.action in ('smoke', 'render') and not args.guarded:
        parser.error('GPU smoke/render require --guarded and the sequential curator')
    if args.action == 'prepare-audio':
        value = prepare_audio(args.root, args.song, args.plan_sha256)
    elif args.action == 'check-inputs':
        *_, pins, strip = load(args.root, args.method, args.song, args.plan_sha256)
        value = {'inputs': pins, 'native_frames': FRAMES, 'start_s': START, 'duration_s': DURATION,
                 'raw_trace_evidence': strip.evidence(), 'diagnostic': True, 'baseline_promotion': False}
    elif args.action == 'audit':
        value = audit(args.root, args.method, args.song, args.plan_sha256)
    else:
        value = globals()[args.action](args.root, args.method, args.song, args.guarded, args.plan_sha256)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
