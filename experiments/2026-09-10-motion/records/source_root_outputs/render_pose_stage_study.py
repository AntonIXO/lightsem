"""Frozen pose-stage media consumer. No freeze, mapping, model, or vault calls.
Run ONLY with bash run_torch.sh outputs/render_pose_stage_study.py --output FROZEN_DIR.
Completed immutable entries are verified and skipped. A partial entry blocks retry.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ARMS = ('baseline', 'candidate')
WIDTH, HEIGHT, FPS = 1920, 360, 30
SCOPE = 'Saved-body mapper diagnostic; no human acceptance, music-fit, hardware or pipeline qualification.'


# Engineering storage policy, not a registered scientific gate or CRF bound.
MIB = 2**20
STORAGE_RESERVE_BYTES = 1024*MIB
STUDY_ARTIFACT_LIMIT_BYTES = 1792*MIB  # Exact parent 1.75GiB policy.
BOOKKEEPING_MARGIN_BYTES = 4*MIB
STREAM_WRITE_HEADROOM_BYTES = 32*MIB
VIDEO_PLANNING_BYTES_PER_SECOND = 2*MIB  # Combined advisory ~16.78Mbit/s, NOT a CRF bound.
PNG_PLANNING_OVERHEAD_BYTES = 64*1024


class ResourceBlocked(RuntimeError):
    def __init__(self, path, phase, free_bytes, required_bytes, **details):
        self.observation = {'path': str(path), 'phase': phase, 'free_bytes': int(free_bytes),
                            'required_free_bytes': int(required_bytes), 'scientific_refutation': False, **details}
        super().__init__(f'Resource blocked at {phase}: free={free_bytes} bytes, required={required_bytes}, '
                         f'study bytes={details.get("study_artifact_bytes", "unknown")}, '
                         f'budget remaining={details.get("remaining_study_budget_bytes", "unknown")}; partial evidence retained')


def study_root(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent/'freeze.json').is_file():
            return parent
    raise ValueError('Storage guard requires the frozen study root')


def study_artifact_bytes(path):
    """Count ALL study files, including frozen sources/maps/partial movies/logs."""
    total = 0
    for directory, _, files in os.walk(study_root(path), followlinks=False):
        for name in files:
            total += (Path(directory)/name).stat(follow_symlinks=False).st_size
    return total


def storage_guard(path, phase, write_allowance=STREAM_WRITE_HEADROOM_BYTES):
    free = shutil.disk_usage(path).free
    current = study_artifact_bytes(path)
    allowance = int(write_allowance)+BOOKKEEPING_MARGIN_BYTES
    required = STORAGE_RESERVE_BYTES+allowance
    observation = {'free_bytes': int(free), 'required_free_bytes': required,
        'study_artifact_bytes': current, 'study_artifact_limit_bytes': STUDY_ARTIFACT_LIMIT_BYTES,
        'remaining_study_budget_bytes': STUDY_ARTIFACT_LIMIT_BYTES-current,
        'next_write_and_inflight_allowance_bytes': int(write_allowance),
        'bookkeeping_margin_bytes': BOOKKEEPING_MARGIN_BYTES}
    if free < required or current+allowance > STUDY_ARTIFACT_LIMIT_BYTES:
        raise ResourceBlocked(path, phase, free, required, **{k: v for k, v in observation.items()
            if k not in ('free_bytes', 'required_free_bytes')})
    return observation


def storage_preflight(out, plan, entries):
    """Conservative planning estimate, NOT an upper bound on CRF media size.
    The mandatory per-write runtime guard remains active after this check.
    """
    estimates = []
    for entry in entries:
        item = next(i for i in plan['items'] if i['id'] == entry['item_id'])
        schedule = samples(plan, item, entry)
        selected = set().union(*map(set, schedule.values()))
        page_count = sum((len(indexes)+11)//12 for indexes in schedule.values())
        decoded_png_allowance = len(selected)*(WIDTH*HEIGHT*3+PNG_PLANNING_OVERHEAD_BYTES)
        sheet_allowance = page_count*(1920*6*202*3+PNG_PLANNING_OVERHEAD_BYTES)
        video_allowance = int(np.ceil(entry['frames']/FPS*VIDEO_PLANNING_BYTES_PER_SECOND))
        estimated = video_allowance+decoded_png_allowance+sheet_allowance+32*MIB
        estimates.append({'entry_id': entry['id'], 'estimated_bytes': estimated,
            'video_planning_bytes': video_allowance,
            'retained_silent_video_planning_bytes': 0,
            'retained_muxed_video_planning_bytes': video_allowance,
            'video_components_scope': 'Combined allowance; this renderer retains only one directly muxed movie, no separate silent intermediate.',
            'decoded_png_allowance_bytes': decoded_png_allowance,
            'sheet_allowance_bytes': sheet_allowance, 'metadata_allowance_bytes': 32*MIB})
    total = sum(e['estimated_bytes'] for e in estimates)
    try:
        checked = storage_guard(out, 'storage_preflight', total)
    except ResourceBlocked as error:
        error.observation.update(estimated_entries=estimates, estimated_batch_bytes=total, estimate_is_crf_hard_bound=False)
        print('STORAGE PREFLIGHT BLOCKED', json.dumps(error.observation, sort_keys=True), flush=True)
        raise
    print('STORAGE PREFLIGHT', json.dumps({'estimated_batch_bytes': total, **checked}, sort_keys=True), flush=True)
    return {'entries': estimates, 'estimated_batch_bytes': total, **checked,
        'reserve_bytes': STORAGE_RESERVE_BYTES, 'study_artifact_limit_bytes': STUDY_ARTIFACT_LIMIT_BYTES,
        'runtime_write_headroom_bytes': STREAM_WRITE_HEADROOM_BYTES, 'bookkeeping_margin_bytes': BOOKKEEPING_MARGIN_BYTES,
        'video_planning_bytes_per_second': VIDEO_PLANNING_BYTES_PER_SECOND,
        'estimate_is_crf_hard_bound': False, 'runtime_reserve_guard_required': True,
        'scope': 'Infrastructure planning only; not a scientific gate.'}


def require(value, message):
    if not value:
        raise ValueError(message)


def load_evaluator():
    path = Path(__file__).with_name('eval_pose_stage_study.py')
    spec = importlib.util.spec_from_file_location('pose_stage_frozen_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def samples(plan, item, entry):
    """Chronological 1 Hz + last, with the declared fixed-fast source samples."""
    n, start = entry['frames'], entry['start']
    require(type(n) is int and n > 0 and type(start) is int and start >= 0, 'Invalid frame crop')
    regular = sorted(set(range(0, n, FPS)) | {n-1})
    fast_plan = plan['fixed_fast_samples']
    if item['key'] == 'shpongle':
        absolute = []
        for a, b in fast_plan['shpongle_intervals_s']:
            require(float(a*FPS).is_integer() and float(b*FPS).is_integer(), 'Non-frame fast interval')
            absolute.extend(range(int(a*FPS), int(b*FPS), fast_plan['stride_frames']))
    else:
        absolute = [item['start']+i for i in fast_plan['reference_local_frames']]
    fast = sorted({i-start for i in absolute if start <= i < start+n})
    return {'chronological_1hz_last': regular, 'fixed_fast': fast}


def validate_arrays(worlds, streams, item, entry):
    n = item['full_frames']
    world = worlds['world']
    require(world.shape == (n, 22, 3) and np.isfinite(world).all(), 'Full cached world required')
    require(np.array_equal(worlds['times'], np.arange(n)/30), 'World timeline changed')
    count = round(n/30*40)
    require(np.array_equal(streams['time'], np.arange(count)/40), 'Full DMX timeline changed')
    for arm in ARMS:
        d = streams[arm]
        require(d.shape == (count, 512) and d.dtype == np.uint8, 'Full byte DMX required: '+arm)
    require(entry['start']+entry['frames'] <= n, 'No pose padding permitted')
    return world


def raw_hash(rgb):
    return np.frombuffer(hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).digest(), dtype=np.uint8)


def executable(ctx, name):
    paths = [p for p in ctx['freeze']['executables'] if Path(p).name == name]
    require(len(paths) == 1, 'Exactly one frozen executable required: '+name)
    return paths[0]


def encode_command(ffmpeg, path, n, audio, offset):
    return [ffmpeg, '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0',
            '-ss', str(offset), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
            '-frames:v', str(n), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-threads', '4', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
            '-af', f'atrim=duration={n/FPS},asetpts=PTS-STARTPTS', '-t', str(n/FPS), str(path)]


@contextmanager
def encoder(command, log):
    with Path(log).open('xb') as err:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=err)
        try:
            yield proc.stdin
        except BaseException:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.terminate()
            proc.wait()
            raise
        else:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            code = proc.wait()
    require(code == 0, 'Software encoding failed; partial output retained: '+Path(log).read_text())


def read_exact(pipe, n):
    parts, left = [], n
    while left:
        part = pipe.read(left)
        if not part:
            break
        parts.append(part)
        left -= len(part)
    return b''.join(parts)


def decode_command(ffmpeg, video, audio=False):
    common = [ffmpeg, '-v', 'error', '-nostdin', '-xerror', '-err_detect', 'explode',
              '-hwaccel', 'none', '-threads', '4', '-i', str(video)]
    if audio:
        return common + ['-map', '0:a:0', '-vn', '-ac', '2', '-ar', '48000',
                         '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1']
    return common + ['-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
                     '-c:v', 'rawvideo', '-threads', '1', '-f', 'rawvideo', 'pipe:1']


def review_sheets(target, schedule, shots, entry, font, ev):
    from PIL import Image, ImageDraw
    result = {}
    for name, indexes in schedule.items():
        pages = []
        for page, begin in enumerate(range(0, len(indexes), 12)):
            group = indexes[begin:begin+12]
            sheet = Image.new('RGB', (1920, 6*202), (25, 25, 25))
            draw = ImageDraw.Draw(sheet)
            for slot, frame in enumerate(group):
                row, col = divmod(slot, 2)
                with Image.open(target/shots[frame]['path']) as image:
                    tile = image.resize((960, 180))
                x, y = col*960, row*202
                sheet.paste(tile, (x, y+22))
                draw.text((x+4, y+2), f'{entry["id"]} | frame {frame} | source {(entry["start"]+frame)/FPS:.3f}s', font=font, fill='white')
            path = target/f'{name}_{page:03d}.png'
            storage_guard(target, 'review_sheet_write')
            with path.open('xb') as f:
                sheet.save(f, format='PNG')
            pages.append({'path': path.name, 'sha256': ev.digest(path), 'frames': group})
        result[name] = pages
    return result


def audit_media(target, n, ctx, sample_schedule, entry, font, ev):
    """Decode every video frame and every audio sample using software only.
    Review pictures come from decoded delivery bytes, not pre-encode substitutes.
    """
    from PIL import Image
    video = target/'matched.mp4'
    ffmpeg, ffprobe = executable(ctx, 'ffmpeg'), executable(ctx, 'ffprobe')
    probe_cmd = [ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)]
    info = json.loads(subprocess.check_output(probe_cmd, text=True))
    videos = [s for s in info['streams'] if s['codec_type'] == 'video']
    audios = [s for s in info['streams'] if s['codec_type'] == 'audio']
    require(len(videos) == len(audios) == 1, 'Exactly one video and audio stream required')
    v, a = videos[0], audios[0]
    require((v['width'], v['height'], v['r_frame_rate'], v['codec_name']) == (WIDTH, HEIGHT, '30/1', 'h264'), 'Wrong video format')
    require(abs(float(v['duration'])-n/FPS) < .001, 'Video duration differs from exact frame duration')
    require(abs(float(a['duration'])-n/FPS) < .08, 'Missing or truncated muxed audio')
    selected = set().union(*map(set, sample_schedule.values()))
    shots, count, h = {}, 0, hashlib.sha256()
    cmd = decode_command(ffmpeg, video)
    with (target/'decode_video.log').open('xb') as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                storage_guard(target, 'software_video_decode_and_samples')
                frame = read_exact(proc.stdout, WIDTH*HEIGHT*3)
                if not frame:
                    break
                require(len(frame) == WIDTH*HEIGHT*3, 'Partial decoded video frame')
                require(count < n, 'Excess decoded video frames')
                h.update(frame)
                if count in selected:
                    path = target/f'decoded_{count:06d}.png'
                    with path.open('xb') as f:
                        Image.frombytes('RGB', (WIDTH, HEIGHT), frame).save(f, format='PNG')
                    shots[count] = {'frame': count, 'source_time_s': (entry['start']+count)/FPS,
                                    'path': path.name, 'sha256': ev.digest(path)}
                count += 1
        except BaseException:
            proc.terminate()
            raise
        finally:
            proc.stdout.close()
            code = proc.wait()
    require(code == 0 and count == n, f'Complete software video decode failed: frames={count}/{n}, code={code}: '+(target/'decode_video.log').read_text())
    require(set(shots) == selected, 'Missing declared decoded review frames')
    ah, byte_count = hashlib.sha256(), 0
    acmd = decode_command(ffmpeg, video, audio=True)
    with (target/'decode_audio.log').open('xb') as err:
        proc = subprocess.Popen(acmd, stdout=subprocess.PIPE, stderr=err)
        try:
            for block in iter(lambda: proc.stdout.read(2**20), b''):
                storage_guard(target, 'software_audio_decode')
                ah.update(block)
                byte_count += len(block)
        except BaseException:
            proc.terminate()
            raise
        finally:
            proc.stdout.close()
            code = proc.wait()
    require(code == 0 and byte_count > 0 and byte_count % 4 == 0, 'Complete software audio decode failed')
    samples_decoded = byte_count//4
    require(abs(samples_decoded/48000-n/FPS) <= .08, 'Decoded audio coverage mismatch')
    sheets = review_sheets(target, sample_schedule, shots, entry, font, ev)
    return {'ffprobe': info, 'probe_command': probe_cmd, 'video_decode_command': cmd,
            'audio_decode_command': acmd, 'software_video_frames': count,
            'decoded_rgb_sha256': h.hexdigest(), 'decoded_audio_pcm_sha256': ah.hexdigest(),
            'decoded_audio_samples_stereo_48000': samples_decoded,
            'samples': [shots[i] for i in sorted(shots)], 'review_sheets': sheets,
            'review_status': 'pending independent observation; machine decode is not human acceptance'}


def summarize(metrics):
    return {k: {'mean': float(np.mean(v)), 'max': float(np.max(v)), 'min': float(np.min(v))}
            for k, v in metrics.items()}


def pixel_gate(summary):
    clip = summary['outside_bulbs_anychannel_ge250']
    return clip['mean'] < .01 and clip['max'] < .05


def verified_render(out, entry, ctx, ev):
    target = out/'renders'/entry['id']
    receipt = ev.loadjson(target/'render.json')
    require(receipt['entry'] == entry, 'Approved render identity changed')
    require(receipt['render_authorization'] == ctx['render_authorization'], 'Stage receipt or approved schedule changed')
    for path, sha in receipt['render_authorization']['stage_receipts'].items():
        ev.checked(path, sha)
    require(receipt['freeze_sha256'] == ev.digest(out/'freeze.json'), 'Render freeze changed')
    item = next(i for i in ctx['plan']['items'] if i['id'] == entry['item_id'])
    expected_samples = samples(ctx['plan'], item, entry)
    require(receipt['sample_schedule'] == expected_samples, 'Incomplete declared sampling schedule')
    expected_frames = sorted(set().union(*map(set, expected_samples.values())))
    require([s['frame'] for s in receipt['media']['samples']] == expected_frames, 'Incomplete decoded samples')
    require(receipt['media']['software_video_frames'] == entry['frames'], 'Incomplete decoded video')
    required_files = {'matched.mp4', 'raw_pixels.npz', 'raw_hashes.npz', 'repeated_source_raw_hashes.npz',
                      'pixel_masks.npz', 'started.json', 'encode.log', 'decode_video.log', 'decode_audio.log'}
    for name, indexes in expected_samples.items():
        pages = receipt['media']['review_sheets'][name]
        require([i for page in pages for i in page['frames']] == indexes, 'Incomplete decoded review sheets')
        required_files.update(page['path'] for page in pages)
    required_files.update(s['path'] for s in receipt['media']['samples'])
    require(required_files <= set(receipt['files']), 'Render manifest omits required media')
    source, _ = ev.mapped(out, ctx, item)
    require(receipt['map_sha256'] == ev.digest(source/'map.json'), 'Rendered map changed')
    for rel, sha in receipt['files'].items():
        ev.checked(target/rel, sha)
    if receipt['brightness_reference'] is not None:
        ref = receipt['brightness_reference']
        ev.checked(ref['receipt_path'], ref['receipt_sha256'])
    return target, receipt


def matched_reference(out, ctx, entry, schedule, ev):
    require(entry['item_id'] == 'shpongle_prior_flow0' and entry['start'] == 0,
            'Global brightness control is full Shpongle only')
    peers = [e for e in schedule if not e['matched'] and e['item_id'] == entry['item_id']
             and e['start'] == 0 and e['frames'] == entry['frames']]
    require(len(peers) == 1, 'Exactly one full original brightness reference required')
    target, receipt = verified_render(out, peers[0], ctx, ev)
    means = [receipt['pixels'][a]['outside_bulbs_mean_luma_255']['mean'] for a in ARMS]
    scalars = ctx['helpers']['optics'].brightness_scalars(means)
    return target, receipt, np.asarray(scalars, dtype=float)


def render_one(out, ctx, entry, schedule, ev, renderer, font):
    from PIL import Image, ImageDraw
    require(schedule == ctx['render_authorization']['schedule'] and entry in schedule, 'Entry is not stage-authorized')
    plan, review, optics = ctx['plan'], ctx['helpers']['review'], ctx['helpers']['optics']
    item = next(i for i in plan['items'] if i['id'] == entry['item_id'])
    source, map_receipt = ev.mapped(out, ctx, item)
    worlds, streams = ev.arrays(source/'full_world.npz'), ev.arrays(source/'full_dmx.npz')
    world = validate_arrays(worlds, streams, item, entry)
    start, n = entry['start'], entry['frames']
    target = out/'renders'/entry['id']
    target.mkdir(parents=True, exist_ok=False)
    ev.savejson(target/'started.json', {'entry': entry, 'freeze_sha256': ev.digest(out/'freeze.json'),
                'map_sha256': ev.digest(source/'map.json'), 'render_authorization': ctx['render_authorization'],
                'storage_preflight': ctx.get('storage_preflight'), 'started_unix': time.time(), 'scope': SCOPE})
    phase = 'raw_render_and_encoding'
    assessed_gates = {'candidate_pixels': None, 'complete_software_media': None}
    try:
        storage_guard(target, 'render_entry_start')
        indexes = review.dmx_indices(start, n, len(streams['time']))
        prepared = {a: review.optical_dmx(streams[a], ctx['rig'], plan['renderer'])[indexes] for a in ARMS}
        masks, bulb_xy = optics.pixel_masks(ctx['rig'], ctx['cfg'])
        require(all(m.shape == (360, 640) and m.dtype == bool for m in masks.values()), 'Pixel mask shape changed')
        ev.savenpz(target/'pixel_masks.npz', **masks)
        scalars, reference, original_hashes = np.ones(2), None, None
        original = None
        if entry['matched']:
            require(n == item['full_frames'], 'Brightness may not use selected excerpts')
            ref_target, original, scalars = matched_reference(out, ctx, entry, schedule, ev)
            original_hashes = ev.arrays(ref_target/'raw_hashes.npz')
            reference = {'receipt_path': str(ref_target/'render.json'), 'receipt_sha256': ev.digest(ref_target/'render.json'),
                         'scalars_baseline_candidate': scalars.tolist(),
                         'method': 'One scalar per full-record display RGB stream; dim both to lower raw outside-bulb mean. No per-frame match.'}
        columns = None
        stats = {a: [] for a in ARMS}
        frame_hashes = {a: np.empty((n, 32), np.uint8) for a in ARMS}
        repeat_hashes = {a: np.empty((n, 32), np.uint8) for a in ARMS}
        raw = {a: hashlib.sha256() for a in ARMS}
        repeated = {a: hashlib.sha256() for a in ARMS}
        composite = hashlib.sha256()
        residual_luma = np.zeros((n, 2), dtype=float)
        max_quantization = np.zeros(2)
        command = encode_command(executable(ctx, 'ffmpeg'), target/'matched.mp4', n,
                                 item['audio_path'], item['audio_offset_s']+start/FPS)
        with encoder(command, target/'encode.log') as pipe:
            for i in range(n):
                storage_guard(target, 'streamed_video_frame')
                body = review.body_image(item['source_arm'], world, {}, start+i, font)
                require(body.size == (640, 360), 'Frozen body helper dimensions changed')
                image = Image.new('RGB', (WIDTH, HEIGHT))
                image.paste(body, (0, 0))
                for col, a in enumerate(ARMS):
                    renderer.cfg = ctx['cfg']
                    rgb = renderer.render_frame(prepared[a][i], 0.).copy()
                    require(rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8, 'Raw renderer output changed')
                    repeat_hashes[a][i] = raw_hash(rgb)
                    repeated[a].update(rgb.tobytes())
                    if original_hashes is not None:
                        require(np.array_equal(repeat_hashes[a][i], original_hashes[a][i]),
                                f'Repeated original raw hash differs: {a}, frame {i}; no brightness claim')
                    dimmed = optics.dim_rgb(rgb, float(scalars[col]))
                    if entry['matched']:
                        error = dimmed.astype(float)-rgb.astype(float)*scalars[col]
                        max_quantization[col] = max(max_quantization[col], float(np.abs(error).max()))
                        luma_error = error @ np.array([.2126, .7152, .0722])
                        residual_luma[i, col] = float(luma_error[masks['outside_bulbs']].mean())
                    frame_hashes[a][i] = raw_hash(dimmed)
                    raw[a].update(dimmed.tobytes())
                    values = optics.pixel_stats(dimmed, masks)
                    if columns is None:
                        columns = sorted(values)
                    require(sorted(values) == columns and np.isfinite(list(values.values())).all(), 'Invalid pixel statistics')
                    stats[a].append([values[k] for k in columns])
                    image.paste(Image.fromarray(dimmed), ((col+1)*640, 0))
                draw = ImageDraw.Draw(image)
                captions = [f'FIXED SAVED BODY | {item["source_arm"]}', 'BASELINE | matched stage02 articulated_stage', 'CANDIDATE | pose_stage']
                for col, caption in enumerate(captions):
                    x = col*640
                    draw.rectangle((x, 0, x+639, 17), fill=(12, 12, 12))
                    draw.text((x+4, 1), caption, fill='white', font=font)
                draw.text((645, 340), f'{entry["id"]} | source {(start+i)/FPS:.3f}s | dim={scalars[0]:.6f}/{scalars[1]:.6f}', fill='white', font=font)
                payload = image.tobytes()
                composite.update(payload)
                pipe.write(payload)
                if i % 300 == 0:
                    print('RENDER', entry['id'], i+1, '/', n, flush=True)
        storage_guard(target, 'raw_metric_and_hash_artifacts')
        metrics = {a: np.asarray(stats[a], dtype=float) for a in ARMS}
        ev.savenpz(target/'raw_pixels.npz', **metrics, residual_luma=residual_luma)
        ev.savenpz(target/'raw_hashes.npz', **frame_hashes)
        ev.savenpz(target/'repeated_source_raw_hashes.npz', **repeat_hashes)
        pixels = {a: summarize({k: metrics[a][:, j] for j, k in enumerate(columns)}) for a in ARMS}
        assessed_gates['candidate_pixels'] = pixel_gate(pixels['candidate'])
        phase = 'brightness_provenance'
        if reference is not None:
            require(all(repeated[a].hexdigest() == original['raw_rgb_sha256'][a] for a in ARMS), 'Repeated whole raw stream hash mismatch')
            require(np.all(max_quantization <= .5+1e-9), 'Unexpected RGB rounding residual')
            matched_means = [pixels[a]['outside_bulbs_mean_luma_255']['mean'] for a in ARMS]
            reference.update({'all_repeated_frame_rawhashes_equal': True,
                'original_raw_rgb_sha256': {a: repeated[a].hexdigest() for a in ARMS},
                'quantization_max_absolute_channel_error_255': max_quantization.tolist(),
                'quantization_outside_bulb_mean_luma_residual_255': residual_luma.mean(0).tolist(),
                'matched_outside_bulb_mean_luma_255': matched_means,
                'matched_mean_difference_255': float(matched_means[1]-matched_means[0])})
        sample_schedule = samples(plan, item, entry)
        phase = 'software_media_audit'
        assessed_gates['complete_software_media'] = False
        media = audit_media(target, n, ctx, sample_schedule, entry, font, ev)
        assessed_gates['complete_software_media'] = True
        phase = 'receipt_finalization'
        storage_guard(target, 'final_render_receipt')
        files = {str(p.relative_to(target)): ev.digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        for rel in files:
            (target/rel).chmod(0o444)
        receipt = {'entry': entry, 'item': item, 'render_authorization': ctx['render_authorization'],
            'freeze_sha256': ev.digest(out/'freeze.json'),
            'map_sha256': ev.digest(source/'map.json'), 'files': files,
            'storage_preflight': ctx.get('storage_preflight'),
            'storage_runtime_policy': {'reserve_bytes': STORAGE_RESERVE_BYTES, 'write_headroom_bytes': STREAM_WRITE_HEADROOM_BYTES,
                'study_artifact_limit_bytes': STUDY_ARTIFACT_LIMIT_BYTES, 'bookkeeping_margin_bytes': BOOKKEEPING_MARGIN_BYTES,
                'cadence': 'Every encoded/decoded video frame; every decoded audio1MiB block; every PNG sheet; before artifact groups and final receipt.',
                'scope': 'Infrastructure guard only; concurrent external writers can defeat any sampled free-space check.'},
            'source_world_sha256': ev.digest(source/'full_world.npz'), 'source_dmx_sha256': ev.digest(source/'full_dmx.npz'),
            'audio_path': item['audio_path'], 'audio_sha256': ev.digest(item['audio_path']),
            'audio_trim_start_s': item['audio_offset_s']+start/FPS, 'audio_duration_s': n/FPS,
            'audio_padding': False, 'pose_padding': False,
            'ffmpeg_command': command, 'renderer_device': renderer.renderer_name,
            'shared_context': True, 'fixed_renderer_time': 0., 'panel_pixels': [640, 360],
            'layout': ['one shared fixed-world body front+oblique', 'baseline', 'candidate'],
            'caption_provenance': {'body': item['source_arm'], 'baseline': 'articulated_stage',
                'candidate': 'pose_stage', 'metric_input': 'Raw uncaptioned native640x360 RGB before video encoding'},
            'pixel_columns': columns, 'pixels': pixels,
            'bulb_mask_definition': 'Frozen optics helper: 20px fixture circles; outside_caption excludes first28 rows',
            'bulb_centers_pixels': [[float(x) if np.isfinite(x) else None for x in row] for row in bulb_xy],
            'raw_rgb_sha256': {a: raw[a].hexdigest() for a in ARMS},
            'decorated_raw_rgb_sha256': composite.hexdigest(), 'brightness_reference': reference,
            'sample_schedule': sample_schedule, 'media': media,
            'gates': assessed_gates,
            'scope': SCOPE, 'human_music_fit_acceptance': None, 'pipeline_qualified': False, 'default_promotion': False}
        ev.savejson(target/'render.json', receipt)
        print('RENDER COMPLETE', entry['id'], receipt['gates'], flush=True)
        return receipt
    except BaseException as error:
        resource = isinstance(error, ResourceBlocked) or isinstance(error, getattr(ev, 'ResourceBlocked', ()))
        if resource and phase == 'software_media_audit':
            assessed_gates['complete_software_media'] = None  # Resource stop is not measured decode failure.
        try:
            ev.savejson(target/'failed.json', {'entry': entry, 'freeze_sha256': ev.digest(out/'freeze.json'),
                    'render_authorization': ctx['render_authorization'], 'phase': phase, 'gates': assessed_gates,
                    'outcome': 'resource_blocked' if resource else 'technical_failure',
                    'resource_observation': getattr(error, 'observation', {'frozen_evaluator_resource_guard': str(error)}) if resource else None,
                    'error_type': type(error).__name__, 'error': str(error),
                    'partial_artifacts_retained': True, 'retry_in_place': False,
                    'human_music_fit_acceptance': None, 'pipeline_qualified': False})
        except BaseException as receipt_error:
            error.add_note(f'Secondary failure saving failure receipt: {receipt_error}')
            print(f'Failure receipt could not be saved: {receipt_error}; ORIGINAL: {type(error).__name__}: {error}', file=sys.stderr)
        raise


def validate_context(ctx):
    plan = ctx['plan']
    require(plan['fps'] == {'mapping': 40, 'video': 30}, 'Unapproved rates')
    require(plan['renderer']['zoom'] == 'zero' and plan['renderer']['frozen_angles'] is False, 'Unapproved optical control')
    require(plan['body_display'] == {'scale_px_m': 60, 'origin_world': [0, 0, 0],
            'views': ['front', 'oblique'], 'no_recentering': True}, 'Body display changed')
    require((ctx['cfg'].width, ctx['cfg'].height, ctx['cfg'].fps, ctx['cfg'].cam_orbit) == (640, 360, 30., 0.), 'Fixed rendering geometry changed')
    require(set(ctx['helpers']) >= {'review', 'optics'}, 'Explicit frozen helper bindings required')


def verify_review(out, ctx, entry, ev):
    target, rendered = verified_render(out, entry, ctx, ev)
    path = out/'reviews'/(entry['id']+'.json')
    receipt = ev.loadjson(path)
    require(receipt['entry'] == entry and receipt['freeze_sha256'] == ev.digest(out/'freeze.json'), 'Review identity changed')
    require(receipt['render_sha256'] == ev.digest(target/'render.json'), 'Reviewed render changed')
    require(receipt['render_authorization'] == ctx['render_authorization'], 'Review stage authorization changed')
    return receipt


def record_review(out, ctx, schedule, observation, ev):
    require(observation is not None, '--observation required for review')
    pending = []
    for entry in schedule:
        path = out/'reviews'/(entry['id']+'.json')
        if path.exists():
            verify_review(out, ctx, entry, ev)
        else:
            pending.append(entry)
    require(pending, 'All approved entries already have review receipts')
    entry = pending[0]
    target, rendered = verified_render(out, entry, ctx, ev)
    observed = json.loads(Path(observation).read_text())
    require(observed.get('entry_id') == entry['id'], 'Review next missing approved entry only')
    for key in ('reviewer', 'notes'):
        require(isinstance(observed.get(key), str) and observed[key].strip(), 'Nonempty reviewer and notes required')
    require(observed.get('independent_of_renderer') is True, 'Explicit independent-review attestation required')
    require(observed.get('coverage') == 'all_frozen_decoded_samples', 'Declare complete sampled coverage, not unmeasured playback')
    expected_frames = sorted(set().union(*map(set, rendered['sample_schedule'].values())))
    require(observed.get('viewed_sample_frames') == expected_frames, 'Every chronological and fixed-fast sample is required')
    expected_sheets = [s['path'] for pages in rendered['media']['review_sheets'].values() for s in pages]
    require(sorted(observed.get('viewed_sheets', [])) == sorted(expected_sheets), 'Review every declared sheet')
    comparisons = observed.get('comparisons', {})
    require(set(comparisons) == {'body', 'baseline_vs_candidate', 'clipping_and_visibility'}, 'All prescribed comparisons required')
    require(all(isinstance(v, str) and v.strip() for v in comparisons.values()), 'Comparison observations cannot be empty')
    require(isinstance(observed.get('limitations'), list) and all(isinstance(v, str) and v.strip() for v in observed['limitations']), 'Explicit limitations list required')
    (out/'reviews').mkdir(exist_ok=True)
    ev.savejson(out/'reviews'/(entry['id']+'.json'), {'entry': entry,
        'freeze_sha256': ev.digest(out/'freeze.json'), 'render_sha256': ev.digest(target/'render.json'),
        'render_authorization': ctx['render_authorization'], 'observation': observed,
        'observation_file_sha256': ev.digest(observation), 'created_unix': time.time(),
        'coverage_and_independence_are_observer_attestations_not_machine_proof': True,
        'human_music_fit_acceptance': None, 'pipeline_qualified': False, 'scope': SCOPE})
    print('Independent sampled observation recorded:', entry['id'], flush=True)


def aggregate_gate(rows, name):
    values = [row['gates'][name] for row in rows]
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def report(out, ctx, schedule, ev):
    """Terminal snapshot, including negative and missing coverage. Never repairs."""
    rows = []
    for entry in schedule:
        target = out/'renders'/entry['id']
        row = {'entry': entry, 'status': 'unrun', 'render_sha256': None,
            'failure_sha256': None, 'review_sha256': None, 'reviewed': False,
            'gates': {'candidate_pixels': None, 'complete_software_media': None},
            'observed_artifact_hashes': {}, 'verification_error': None}
        if target.exists():
            row['status'] = 'incomplete'
            # Inventory is evidence of bytes present now, not of valid completion.
            row['observed_artifact_hashes'] = {str(p.relative_to(target)): ev.digest(p)
                for p in sorted(target.rglob('*')) if p.is_file()}
            row['artifact_inventory_scope'] = 'Current byte hashes; partial artifacts are not completed or decoded coverage.'
            try:
                if (target/'render.json').exists():
                    _, rendered = verified_render(out, entry, ctx, ev)
                    row.update(status='completed', render_sha256=ev.digest(target/'render.json'), gates=rendered['gates'])
                    if (out/'reviews'/(entry['id']+'.json')).exists():
                        try:
                            verify_review(out, ctx, entry, ev)
                            row.update(reviewed=True, review_sha256=ev.digest(out/'reviews'/(entry['id']+'.json')))
                        except (ValueError, KeyError, OSError, TypeError) as error:
                            row['review_verification_error'] = f'{type(error).__name__}: {error}'
                            # A bad observation never erases valid measured pixel/media failures.
                            row['reviewed'] = False
                else:
                    if (target/'started.json').exists():
                        started = ev.loadjson(target/'started.json')
                        require(started['entry'] == entry and started['freeze_sha256'] == ev.digest(out/'freeze.json'), 'Partial start identity changed')
                        require(started['render_authorization'] == ctx['render_authorization'], 'Partial stage authorization changed')
                        row['started_sha256'] = ev.digest(target/'started.json')
                    if (target/'failed.json').exists():
                        failed = ev.loadjson(target/'failed.json')
                        require(failed['entry'] == entry and failed['freeze_sha256'] == ev.digest(out/'freeze.json'), 'Failure identity changed')
                        require(failed['render_authorization'] == ctx['render_authorization'], 'Failure stage authorization changed')
                        row.update(status='resource_blocked' if failed.get('outcome') == 'resource_blocked' else 'failed',
                            failure_sha256=ev.digest(target/'failed.json'),
                            gates=failed['gates'], failure_phase=failed['phase'],
                            failure_error=failed['error'], failure_error_type=failed['error_type'],
                            resource_observation=failed.get('resource_observation'))
            except (ValueError, KeyError, OSError, TypeError) as error:
                row.update(status='failed', verification_error=f'{type(error).__name__}: {error}',
                    gates={'candidate_pixels': None, 'complete_software_media': None}, reviewed=False)
                row['failure_scope'] = 'Artifact verification failure; scientific pixel/media gates not inferred from unauthenticated bytes.'
        rows.append(row)
    full_expansion = any(e['matched'] for e in schedule)
    complete = bool(rows) and all(row['status'] == 'completed' for row in rows)
    ev.savejson(out/'render_report.json', {'rows': rows, 'freeze_sha256': ev.digest(out/'freeze.json'),
        'render_authorization': ctx['render_authorization'], 'all_approved_entries_rendered': complete,
        'coverage_counts': {status: sum(row['status'] == status for row in rows)
                            for status in ('completed', 'failed', 'resource_blocked', 'incomplete', 'unrun')},
        'terminal_snapshot': True, 'missing_coverage_is_not_a_pass': True,
        'numeric_full_expansion_authorized': full_expansion, 'diagnostic_only': not full_expansion,
        'all_candidate_pixel_gates_pass': aggregate_gate(rows, 'candidate_pixels'),
        'all_software_media_gates_pass': aggregate_gate(rows, 'complete_software_media'),
        'all_independent_sampled_observations_present': complete and all(row['reviewed'] for row in rows),
        'human_music_fit_acceptance': None, 'pipeline_qualified': False, 'default_promotion': False,
        'scope': SCOPE, 'limitations': ['Raw pixel masks are fixed projected fixture circles, not pixel segmentation.',
            'Decoded silent review sheets do not establish music fit or full-playback acceptance.',
            'No reference body for Shpongle. No comparable legacy incumbent3D lighting.']})
    print('Created immutable terminal render_report.json, including missing/failing coverage; no acceptance inferred.', flush=True)


def record_storage_block(out, ctx, entry, error, ev):
    """Best effort. Failed bookkeeping is independent of an ordinary start write."""
    try:
        target = out/'renders'/entry['id']
        target.mkdir(parents=True, exist_ok=False)
        item = next(i for i in ctx['plan']['items'] if i['id'] == entry['item_id'])
        source, _ = ev.mapped(out, ctx, item)
        identity = {'entry': entry, 'freeze_sha256': ev.digest(out/'freeze.json'),
                    'render_authorization': ctx['render_authorization'],
                    'map_sha256': ev.digest(source/'map.json')}
    except BaseException as identity_error:
        error.add_note(f'Secondary failure preparing storage-block identity: {identity_error}')
        print(f'Could not prepare storage-block identity: {identity_error}; ORIGINAL: {error}', file=sys.stderr)
        return
    recording_errors = []
    started_written = False
    try:
        ev.savejson(target/'started.json', {**identity, 'started_unix': time.time(), 'scope': SCOPE})
        started_written = True
    except BaseException as started_error:
        recording_errors.append(f'Ordinary started receipt not written: {type(started_error).__name__}: {started_error}')
        error.add_note(recording_errors[-1])
        print(recording_errors[-1], file=sys.stderr)
    # failed.json is evaluator bookkeeping: it can use the reserved margin even
    # when the ordinary started.json write above was rejected by the real lock.
    try:
        ev.savejson(target/'failed.json', {**identity, 'phase': 'storage_preflight',
            'gates': {'candidate_pixels': None, 'complete_software_media': None},
            'outcome': 'resource_blocked', 'resource_observation': error.observation,
            'error_type': type(error).__name__, 'error': str(error),
            'started_receipt_written': started_written, 'secondary_recording_errors': recording_errors,
            'partial_artifacts_retained': True, 'retry_in_place': False,
            'human_music_fit_acceptance': None, 'pipeline_qualified': False})
    except BaseException as receipt_error:
        error.add_note(f'Secondary failure recording storage block: {receipt_error}')
        print(f'Could not record storage block: {receipt_error}; ORIGINAL: {error}', file=sys.stderr)


def run(out, budget, ev, stage='render', observation=None):
    require(type(budget) is int and budget == 1, 'Parent policy requires exactly --budget 1; combined render batches are forbidden')
    with ev.lock(out):
        ctx = ev.context(out, activate=stage == 'render')
        ev.checked(__file__, ctx['freeze']['files']['render_pose_stage_study.py'])
        if stage == 'render':
            validate_context(ctx)
        schedule = ev.render_schedule(out, ctx)
        native = ev.stage_receipt(out, ctx, 'native')
        stage_names = ['native'] + (['references'] if native['pass'] else [])
        stage_receipts = {}
        for stage_name in stage_names:
            ev.stage_receipt(out, ctx, stage_name)
            path = out/stage_name/'stage.json'
            stage_receipts[str(path)] = ev.digest(path)
        ctx['render_authorization'] = {'schedule': schedule, 'stage_receipts': stage_receipts}
        require(len({e['id'] for e in schedule}) == len(schedule), 'Duplicate approved render ID')
        require(all(Path(e['id']).name == e['id'] and e['id'] not in ('.', '..') for e in schedule), 'Unsafe render ID')
        if stage == 'review':
            record_review(out, ctx, schedule, observation, ev)
            return
        if stage == 'report':
            report(out, ctx, schedule, ev)
            return
        pending = []
        for entry in schedule:
            require(Path(entry['id']).name == entry['id'] and entry['id'] not in ('.', '..'), 'Unsafe render ID')
            target = out/'renders'/entry['id']
            if (target/'render.json').exists():
                verified_render(out, entry, ctx, ev)
            else:
                require(not target.exists(), 'Incomplete render retained; no in-place retry: '+str(target))
                pending.append(entry)
        if not pending:
            print('All approved renders verified. Independent visual review is still required.', flush=True)
            return
        try:
            ctx['storage_preflight'] = storage_preflight(out, ctx['plan'], pending[:budget])
        except ResourceBlocked as error:
            record_storage_block(out, ctx, pending[0], error, ev)
            raise
        from PIL import ImageFont
        from lightsem.experiments.stage11_common import gpu_lock
        from lightsem.sim.renderer import StageRenderer
        font = ImageFont.truetype(ctx['freeze']['font_path'], 13)
        renderer = None
        with gpu_lock(Path(ctx['freeze']['native_gpu_lock_root'])):
            try:
                renderer = StageRenderer(ctx['rig'], ctx['cfg'])  # Exactly one EGL context per invocation.
                for entry in pending[:budget]:
                    render_one(out, ctx, entry, schedule, ev, renderer, font)
            finally:
                if renderer is not None:
                    renderer.ctx.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, choices=(1,), default=1, help='Parent policy: exactly one approved entry per invocation')
    parser.add_argument('--stage', choices=('render', 'review', 'report'), default='render')
    parser.add_argument('--observation', type=Path)
    args = parser.parse_args()
    require(args.budget == 1, 'Parent policy requires exactly --budget 1')
    ev = load_evaluator()
    try:
        run(args.output.resolve(), args.budget, ev, args.stage, args.observation)
    except (ResourceBlocked, ev.ResourceBlocked) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()
