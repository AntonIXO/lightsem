"""Full-song, source-authenticated experimental combination comparisons.

Displays D7 reference and C3/C4 saved bodies and actual decoded DMX. This module
never generates body motion or changes the accepted product baseline. Rendering
is guarded; prepare-audio, audit and gallery are CPU-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from lightsem.conductor.body_transfer import decode
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from scripts.rendering import render_body_portfolio as base
from scripts.rendering import render_dynamic_dance as drawing

WIDTH, HEIGHT, FPS = 1920, 1080, 30
STAGE_WIDTH, STAGE_HEIGHT, STAGE_Y, TRACE_Y = 960, 540, 324, 864
TEMPLATE = Path(__file__).with_name('dynamic_combo_review.html')
sha, read = base.sha, base.read


def save(path, value):
    """Publish complete JSON without replacing any existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def track_clock(track):
    duration = float(track['duration_s'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Track duration must be finite and positive')
    nv, nb, nl = (math.ceil(duration * hz) for hz in (30, 30, 40))
    for key, expected in (('video_frames', nv), ('body_frames', nb), ('lighting_frames', nl)):
        if track.get(key) != expected:
            raise ValueError('Full-song clock mismatch: ' + key)
    times = np.arange(nl, dtype=np.float64) / 40
    indices = np.minimum(np.rint(np.arange(nv) * 40 / FPS).astype(int), nl - 1)
    return times, indices


def verify_source(plan, track):
    path = Path(track['audio'])
    digest = sha(path)
    if plan.get('files', {}).get(str(path)) != digest:
        raise ValueError('Original audio must be pinned by PLAN.files')
    return digest


def resource_guard(root, plan):
    if shutil.disk_usage(root).free < plan.get('minimum_free_bytes', 20 * 2**30):
        raise RuntimeError('Campaign free-space reserve reached')


def audio_paths(root, track):
    folder = root / 'audio' / track['id']
    return folder / 'reference.m4a', folder / 'REFERENCE.json'


def audio_filter(track):
    duration = track['video_frames'] / FPS
    return f'apad=whole_dur={duration:.12f},atrim=end={duration:.12f},asetpts=PTS-STARTPTS'


def prepare_audio(root, plan, track):
    """Encode once at unchanged speed; pad only the fractional final video frame."""
    track_clock(track)
    source_sha = verify_source(plan, track)
    reference, record = audio_paths(root, track)
    if record.exists():
        value = read(record)
        authenticate_audio(root, plan, track)
        return value
    if reference.exists():
        raise RuntimeError('Unreceipted audio retained; inspect before a new revision')
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', track['audio']], text=True))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    actual_duration = float(stream.get('duration', probe['format']['duration']))
    if abs(actual_duration - track['duration_s']) > max(1 / int(stream['sample_rate']), 1e-5):
        raise ValueError('Frozen full-song duration differs from original waveform')
    reference.parent.mkdir(parents=True, exist_ok=True)
    partial = reference.with_name('reference.partial.m4a')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', track['audio'],
        '-map', '0:a:0', '-vn', '-c:a', 'aac', '-b:a', '128k', '-ac', '2', '-ar', '44100',
        '-af', audio_filter(track), '-t', str(track['video_frames'] / FPS), '-movflags', '+faststart', str(partial)]
    subprocess.run(command, check=True)
    os.link(partial, reference)
    partial.unlink()
    value = {'plan_sha256': sha(root / 'PLAN.json'), 'source_audio': track['audio'],
        'source_audio_sha256': source_sha, 'source_duration_s': actual_duration,
        'reference': str(reference), 'reference_sha256': sha(reference),
        'decoded_pcm_sha256': hashlib.sha256(base.decoded_audio(reference)).hexdigest(),
        'duration_s': track['video_frames'] / FPS,
        'end_padding_s': track['video_frames'] / FPS - track['duration_s'],
        'source_start_s': 0., 'playback_rate': 1., 'command': command,
        'scope': 'Full original audio, AAC encoded once, unchanged speed; final fractional-frame silence padding only.'}
    save(record, value)
    return value


def authenticate_audio(root, plan, track):
    reference, record = audio_paths(root, track)
    value = read(record)
    expected = {'plan_sha256': sha(root / 'PLAN.json'), 'source_audio': track['audio'],
        'source_audio_sha256': verify_source(plan, track), 'reference': str(reference),
        'reference_sha256': sha(reference), 'source_start_s': 0., 'playback_rate': 1.,
        'duration_s': track['video_frames'] / FPS}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Full-song audio binding changed')
    expected_padding = track['video_frames'] / FPS - track['duration_s']
    if (not 0 <= value.get('end_padding_s', -1) < 1 / FPS + 1e-10
            or abs(value['end_padding_s'] - expected_padding) > 1e-10):
        raise ValueError('Only a final fractional video frame may be padded')
    return {'audio_record_sha256': sha(record), 'audio_reference_sha256': sha(reference),
            'source_audio_sha256': expected['source_audio_sha256']}


@dataclass
class Pair:
    controls: dict
    body: np.ndarray
    patch: list
    times: np.ndarray
    indices: np.ndarray
    report: dict
    pins: dict
    method: str
    metadata: dict
    source_time: np.ndarray | None
    rate: np.ndarray | None
    interventions: np.ndarray
    comparison: dict | None = None
    brightness_envelope: np.ndarray | None = None


def load_pair(root, plan, method, track):
    times, indices = track_clock(track)
    folder = root / 'timelines' / method / track['id']
    report = read(folder / 'report.json')
    if report.get('status') not in ('engineering_valid', 'diagnostic'):
        raise ValueError('Finite numeric admission is required before rendering')
    pins = {key: sha(folder / name) for key, name in (
        ('timeline_sha256', 'timeline.npz'), ('patch_sha256', 'patch.json'), ('numeric_report_sha256', 'report.json'))}
    if any(report.get(k) != pins[k] for k in ('timeline_sha256', 'patch_sha256')):
        raise ValueError('Timeline/patch changed after numeric admission')
    body_path = root / 'bodies' / method / track['id'] / 'body.npz'
    source = DanceResult.load(body_path)
    if report.get('body_sha256') != sha(body_path):
        raise ValueError('Mapped body does not match authenticated DanceResult')
    if source.world.shape != (track['body_frames'], 22, 3) or not np.allclose(
            source.time, np.arange(track['body_frames']) / 30, rtol=0, atol=1e-8):
        raise ValueError('Body must cover the complete track at native 30 Hz')
    sidecar = body_path.with_suffix('.json')
    expected_meta = report.get('inputs', {}).get('body_metadata_sha256')
    if expected_meta is not None and expected_meta != sha(sidecar):
        raise ValueError('Body provenance metadata changed after mapping')
    pins.update(body_sha256=sha(body_path), body_metadata_sha256=sha(sidecar),
                body_metadata_seal_sha256=sha(sidecar.with_name(sidecar.name + '.sha256.json')))
    patch = read(folder / 'patch.json')
    counts = {k: sum(p['kind'] == k for p in patch) for k in ('beam', 'laser', 'bulb')}
    if counts != {'beam': 8, 'laser': 5, 'bulb': 50}:
        raise ValueError('Delivery rig requires 8 beams, 5 lasers and 50 bulbs')
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if arrays['body'].shape != (len(times), 22, 3) or arrays['universes'].shape[0] != len(times):
        raise ValueError('Full-song body/DMX lengths do not match PLAN')
    if not np.array_equal(arrays['time'], times):
        raise ValueError('Lighting timestamps must be exact native 40 Hz')
    if arrays['universes'].dtype != np.uint8:
        raise ValueError('Emitted universes must be uint8 DMX')
    controls = decode(arrays['universes'], patch)
    if not np.isfinite(arrays['body']).all() or not all(np.isfinite(v).all() for v in controls.values()):
        raise ValueError('Nonfinite actual body or emitted DMX')
    source_time, rate = arrays.get('source_time'), arrays.get('playback_rate')
    for key, values in (('source_time', source_time), ('playback_rate', rate)):
        if values is not None and (values.ndim not in (1, 2) or len(values) != len(times) or not np.isfinite(values).all()):
            raise ValueError('Invalid ' + key)
    # A recomposed source clock has jumps at fragment cuts. Its derivative is
    # not playback rate; require the actual local rate recorded by generation.
    if source_time is not None and rate is None:
        raise ValueError('Composed source times require recorded fragment-local playback_rate')
    interventions = arrays.get('intervention_time', np.empty(0))
    if interventions.ndim != 1 or not np.isfinite(interventions).all():
        raise ValueError('Invalid intervention timestamps')
    envelope = arrays.get('brightness_envelope', arrays.get('audio_brightness_envelope'))
    if envelope is not None and (envelope.shape != (len(times),) or not np.isfinite(envelope).all()):
        raise ValueError('Invalid brightness envelope')
    return Pair(controls, arrays['body'], patch, times, indices, report, pins, method,
                source.metadata, source_time, rate, interventions, brightness_envelope=envelope)


def load_comparison(root, plan, algorithm, track):
    reference = load_pair(root, plan, 'D7_reference', track)
    candidate = load_pair(root, plan, algorithm['id'], track)
    if candidate.patch != reference.patch or not np.array_equal(candidate.times, reference.times):
        raise ValueError('Reference and candidate must share patch and lighting clock')
    path = root / 'reports' / 'comparisons' / f"{algorithm['id']}_{track['id']}.json"
    comparison = read(path)
    if comparison.get('method') != algorithm['id'] or comparison.get('song') != track['id']:
        raise ValueError('Comparison report has a different method/song identity')
    required = {str(root / 'PLAN.json'): sha(root / 'PLAN.json')}
    for pair in (reference, candidate):
        for filename, key in (('timeline.npz', 'timeline_sha256'), ('patch.json', 'patch_sha256'),
                              ('report.json', 'numeric_report_sha256')):
            required[str(root / 'timelines' / pair.method / track['id'] / filename)] = pair.pins[key]
    if any(comparison.get('inputs', {}).get(k) != v for k, v in required.items()):
        raise ValueError('Comparison does not authenticate reference/candidate/PLAN')
    candidate.comparison = comparison
    music_path = Path(track['music_timeline'])
    music = MusicTimeline.load(music_path)
    if music.metadata.get('input_sha256') != verify_source(plan, track):
        raise ValueError('Musical traces must come from this track original audio')
    if music.time[0] > 1/30 + 1e-6 or music.time[-1] < track['duration_s'] - 1/20 - 1e-6:
        raise ValueError('Musical analysis does not cover the complete song')
    music_arrays = {'time': music.time, 'sources': music.sources, 'downbeats': music.downbeats,
                    'sections': music.sections, 'mixture': music.mixture}
    pins = {'reference': reference.pins, 'candidate': candidate.pins,
        'comparison_sha256': sha(path), 'plan_sha256': sha(root / 'PLAN.json'),
        'music_timeline_sha256': sha(music_path), 'music_metadata_sha256': sha(music_path.with_suffix('.json')),
        'renderer_sha256': sha(Path(__file__)), 'template_sha256': sha(TEMPLATE),
        **authenticate_audio(root, plan, track)}
    return reference, candidate, music_arrays, pins


def evidence_summary(comparison):
    physical = comparison.get('physical_engineering_status', 'unqualified')
    reasons = comparison.get('physical_failure_reasons', [])
    if physical == 'fail':
        return 'FAILURE DIAGNOSTIC | ' + '; '.join(str(s) for s in reasons)
    if (comparison.get('primary') or {}).get('magnitude_gate') == 'fail':
        return 'PROXY FAILURE DIAGNOSTIC | physical checks: ' + physical + ' | artistic review pending | no baseline promotion'
    return 'EXPERIMENTAL | physical checks: ' + physical + ' | artistic review pending | no baseline promotion'


def comparison_note(comparison, method):
    comparator = comparison.get('comparator', (comparison.get('primary') or {}).get('baseline_id'))
    if comparator is None:
        comparator = 'C3' if method == 'C4' else 'D7_reference'
    label = 'C3 (D6 off)' if comparator == 'C3' else comparator.replace('_', ' ')
    return f'Visual reference: D7. {method} numerical comparison: {label}.'


def trace_values(pair):
    values = drawing.trace_values(pair)
    beams = np.array([p['kind'] == 'beam' for p in pair.patch])
    values['brightness'] = pair.controls['dimmer'][:, beams].mean(axis=1)
    return values


class TimelineStrip:
    def __init__(self, reference, candidate, music, track):
        self.start, self.end = 0., track['video_frames'] / FPS
        self.left, self.right = 325, WIDTH - 20
        self.image = Image.new('RGB', (WIDTH, HEIGHT - TRACE_Y), (13, 19, 29))
        draw = ImageDraw.Draw(self.image)
        _, _, font = drawing.fonts()
        draw.text((12, 1), 'Actual body / DMX; gray D7 reference, cyan combination', font=font, fill=(201, 219, 237))
        draw.text((1000, 1), 'Sources: drums / bass / other / vocals (estimated)', font=font, fill=(169, 184, 204))
        ref_values, new_values = trace_values(reference), trace_values(candidate)
        self.scales = {}
        rows = [('sources', 23, 'Sources (own full-song p95)'), ('upper', 51, 'Upper-body speed (m/s)'),
                ('lower', 79, 'Lower-body speed (m/s)'), ('emitted', 107, 'Emitted speed (deg/s)'),
                ('brightness', 135, 'Actual beam dimmer (0–1)'), ('rate', 163, 'Fragment-local rate (x)')]
        for key, y, label in rows:
            draw.text((12, y + 2), label, font=font, fill=(177, 192, 213))
            draw.line((self.left, y + 25, self.right, y + 25), fill=(46, 56, 72))
            if key == 'sources':
                for index, color in enumerate(drawing.SOURCE_COLORS):
                    values = music['sources'][:, index, 0]
                    self.line(draw, music['time'], values, y, max(float(np.quantile(values, .95)), 1e-9), color)
                self.scales[key] = 'own full-song p95 RMS, clipped only for graph pixels'
            elif key == 'rate':
                self.scales[key] = 1.5
                if candidate.rate is None:
                    draw.text((self.left + 4, y), 'No time warp; newly generated motion', font=font, fill=drawing.CANDIDATE_COLOR)
                else:
                    rate = candidate.rate[:, None] if candidate.rate.ndim == 1 else candidate.rate
                    for index in range(rate.shape[1]):
                        self.line(draw, candidate.times, rate[:, index], y, 1.5,
                                  drawing.CANDIDATE_COLOR if rate.shape[1] == 1 else drawing.SOURCE_COLORS[index % 4])
            else:
                scale = 1. if key == 'brightness' else max(float(ref_values[key].max()), float(new_values[key].max()), 1e-9)
                self.scales[key] = scale
                draw.text((270, y + 2), f'{scale:.2g}', font=font, fill=(137, 155, 177))
                self.line(draw, reference.times, ref_values[key], y, scale, drawing.BASE_COLOR)
                self.line(draw, candidate.times, new_values[key], y, scale, drawing.CANDIDATE_COLOR)
        for section in music['sections']:
            px = self.x(section['start'])
            draw.line((px, 21, px, 188), fill=(57, 67, 86))
        for stamp in candidate.interventions:
            px = self.x(stamp)
            draw.polygon([(px-3, 190), (px+3, 190), (px, 196)], fill=(252, 199, 93))
        draw.text((12, 196), 'Section lines are estimates; yellow markers = recorded plan / motif boundaries', font=font, fill=(151, 169, 190))

    def x(self, seconds):
        return int(round(self.left + seconds / self.end * (self.right - self.left)))

    def line(self, draw, times, values, y, scale, color):
        xs = np.arange(self.left, self.right + 1)
        ts = (xs - self.left) / (self.right - self.left) * self.end
        yy = y + 24 - np.clip(np.interp(ts, times, values) / scale, 0, 1) * 22
        draw.line(list(zip(xs.tolist(), yy.tolist())), fill=color, width=1)

    def frame(self, seconds):
        picture = self.image.copy()
        draw = ImageDraw.Draw(picture)
        px = self.x(seconds)
        draw.line((px, 21, px, 193), fill=(249, 246, 215), width=2)
        return picture


def clock_label(candidate, tick):
    if candidate.rate is None:
        return 'New gestures; no playback warp'
    rate = np.asarray(candidate.rate[tick])
    label = f'Local rate {rate.mean():.2f}x' if rate.ndim == 0 else f'Local rate {rate.min():.2f}–{rate.max():.2f}x'
    if candidate.source_time is not None:
        source = np.asarray(candidate.source_time[tick])
        label += (f' | source {float(source):.1f}s' if source.ndim == 0 else f' | source {source.min():.1f}–{source.max():.1f}s')
    return label


def motif_label(candidate, tick):
    """Locate recorded fragments in their stated clock, never treat cuts as rate."""
    fragments = candidate.metadata.get('fragment_provenance', [])
    if not fragments:
        return 'newly generated own-song motion'
    clock = float(candidate.times[tick])
    domain = candidate.metadata.get('fragment_clock_domain', '')
    if candidate.metadata.get('d6_enabled') or domain.startswith('C3 source'):
        if candidate.source_time is None:
            return 'motif provenance retained; C3 source clock unavailable'
        clock = float(np.asarray(candidate.source_time[tick]).reshape(-1)[0])
    for index, fragment in enumerate(fragments):
        bounds = fragment.get('target_seconds', [])
        if len(bounds) != 2:
            continue
        if bounds[0] <= clock < bounds[1] or index == len(fragments)-1 and clock >= bounds[0]:
            original = fragment.get('source_seconds', [])
            label = f'motif {fragment.get("source_fragment", index)}'
            if len(original) == 2:
                label += f' | G78 source {original[0]:.1f}–{original[1]:.1f}s'
            if candidate.metadata.get('d6_enabled'):
                label += ' | via C3 root clock'
            return label
    return 'between recorded motif intervals'


def compose(stages, reference, candidate, tick, frame_time, algorithm, track, strip, diagnostic=None):
    if any(stage.shape != (STAGE_HEIGHT, STAGE_WIDTH, 3) for stage in stages):
        raise ValueError('Expected two 960x540 stage images')
    picture = Image.new('RGB', (WIDTH, HEIGHT), (8, 12, 19))
    for x, stage in zip((0, 960), stages):
        picture.paste(Image.fromarray(stage), (x, STAGE_Y))
    picture.paste(strip.frame(frame_time), (0, TRACE_Y))
    draw = ImageDraw.Draw(picture)
    title, normal, small = drawing.fonts()
    draw.text((16, 8), f"{algorithm['id']} | {algorithm['name']} | {track['title']}", font=title, fill=(233, 240, 249))
    draw.text((16, 42), (diagnostic or evidence_summary(candidate.comparison))[:155], font=small, fill=(241, 190, 114))
    draw.text((1480, 8), f'Song {frame_time:06.2f}s / {track["duration_s"]:.2f}s', font=normal, fill=(234, 241, 249))
    draw.text((1425, 42), clock_label(candidate, tick), font=small, fill=drawing.CANDIDATE_COLOR)
    for x, pair, label in ((0, reference, 'D7 REFERENCE | new own-song body; experimental'),
            (960, candidate, algorithm['id'] + ' COMBINATION | ' + motif_label(candidate, tick))):
        draw.text((x+16, 64), label, font=small, fill=(193, 211, 230))
        drawing.body_panel(draw, pair.body[tick], x)
        drawing.body_panel(draw, pair.body[tick], x+480, oblique=True)
        draw.text((x+14, STAGE_Y+12), 'ACTUAL DMX | restored audio brightness | P2 | 8 beams + 5 lasers + 50 bulbs', font=small, fill=(163, 188, 209))
    draw.line((959, 62, 959, TRACE_Y), fill=(59, 78, 101), width=2)
    draw.text((15, TRACE_Y-24), 'Always-visible body drawings; stages use emitted brightness. ' + comparison_note(candidate.comparison, algorithm['id']), font=small, fill=(163, 188, 209))
    return np.asarray(picture)


@contextmanager
def renderer_for(root, patch, guarded=False, schematic=False):
    if schematic:
        renderer = base.SchematicRenderer(patch)
        class Sized:
            renderer_name = renderer.renderer_name
            def render_controls(self, *controls):
                return np.asarray(Image.fromarray(renderer.render_controls(*controls)).resize((960, 540), Image.Resampling.LANCZOS))
        yield Sized(), None, None
        return
    if not guarded:
        raise ValueError('GPU rendering requires --guarded and the sequential curator')
    from scripts.motion.dynamic_combo_guard import authenticate_guarded_job
    receipt = authenticate_guarded_job(__file__)
    command = receipt['identity']['command']
    if '--root' not in command or Path(command[command.index('--root')+1]).resolve() != root.resolve():
        raise ValueError('Guard command does not bind this campaign root')
    from lightsem.sim.body_stage_renderer import BodyStageRenderer
    from scripts.motion.run_rhythm_companion import render_config
    cfg = render_config(960, 540)
    cfg.camera_scale = .62
    renderer = BodyStageRenderer(patch, cfg)
    try:
        yield renderer, asdict(cfg), receipt
    finally:
        renderer.ctx.release()


def render_stages(renderer, reference, candidate, tick):
    return tuple(renderer.render_controls(*(p.controls[k][tick] for k in ('directions', 'rgb', 'dimmer')))
                 for p in (reference, candidate))


def smoke_pair(root, plan, algorithm, track, guarded=False, schematic=False):
    reference, candidate, music, pins = load_comparison(root, plan, algorithm, track)
    out = root / 'smoke' / algorithm['id'] / track['id'] / ('schematic' if schematic else 'dark')
    record = out / 'SMOKE.json'
    if record.exists():
        old = read(record)
        if old['inputs'] != pins or any(sha(p) != h for p, h in old['files'].items()):
            raise ValueError('Previously saved smoke changed')
        return old
    out.mkdir(parents=True, exist_ok=False)
    strip = TimelineStrip(reference, candidate, music, track)
    samples, files = [], {}
    seconds = sorted(set([0., min(18., candidate.times[-1]), min(60., candidate.times[-1]),
                          min(120., candidate.times[-1]), candidate.times[-1]]))
    with renderer_for(root, candidate.patch, guarded, schematic) as (renderer, cfg, receipt):
        for stamp in seconds:
            tick = min(int(round(stamp*40)), len(candidate.times)-1)
            stages = render_stages(renderer, reference, candidate, tick)
            repeat = render_stages(renderer, reference, candidate, tick)
            if any(not np.array_equal(a, b) for a, b in zip(stages, repeat)):
                raise ValueError('Frozen DMX produced autonomous pixel motion')
            path = out / f'{tick:06d}_actual.png'
            Image.fromarray(compose(stages, reference, candidate, tick, stamp, algorithm, track, strip)).save(path)
            files[str(path)] = sha(path)
            samples.append({'time_s': stamp, 'tick': tick, 'frozen_pixels_exact': True,
                'reference_stage_mean': float(stages[0].mean()), 'candidate_stage_mean': float(stages[1].mean())})
        # Fixture-class visibility controls never enter the production movie.
        tick = len(candidate.times)//2
        direction, rgb = candidate.controls['directions'][tick], np.ones((len(candidate.patch), 3))
        dark = renderer.render_controls(direction, rgb, np.zeros(len(candidate.patch)))
        visibility = {}
        for kind in ('beam', 'laser', 'bulb'):
            frame = renderer.render_controls(direction, rgb, np.array([.65 if p['kind']==kind else 0. for p in candidate.patch]))
            changed = int(np.any(frame != dark, axis=-1).sum())
            if changed == 0:
                raise ValueError('Fixture class invisible: ' + kind)
            visibility[kind] = changed
            path = out / f'optical_{kind}_only.png'
            Image.fromarray(frame).save(path)
            files[str(path)] = sha(path)
        value = {'inputs': pins, 'files': files, 'samples': samples, 'renderer': renderer.renderer_name,
            'renderer_config': cfg, 'gpu_receipt': receipt, 'fixture_class_visibility': visibility,
            'optical_controls_excluded_from_production': True, 'trace_scales': strip.scales}
    save(record, value)
    return value


def command_for(movie, reference, track, codec):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', '30', '-i', 'pipe:0', '-i', str(reference),
        '-map', '0:v:0', '-map', '1:a:0', '-frames:v', str(track['video_frames']), '-c:v', codec]
    if codec == 'av1_nvenc':
        command += ['-gpu', '0', '-preset', 'p7', '-tune', 'uhq', '-rc', 'vbr', '-multipass', 'fullres',
            '-rc-lookahead', '32', '-b:v', '1400k', '-maxrate', '1800k', '-bufsize', '3600k',
            '-spatial-aq', '1', '-temporal-aq', '1']
    elif codec == 'libx264':
        command += ['-preset', 'veryfast', '-threads', '2', '-b:v', '2200k', '-maxrate', '2800k', '-bufsize', '5600k']
    else:
        raise ValueError('Unsupported delivery codec')
    return command + ['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-c:a', 'copy', '-t', str(track['video_frames']/FPS), '-movflags', '+faststart', str(movie)]


def render_pair(root, plan, algorithm, track, codec='av1_nvenc', guarded=False):
    reference, candidate, music, pins = load_comparison(root, plan, algorithm, track)
    smoke = root / 'smoke' / algorithm['id'] / track['id'] / 'dark' / 'SMOKE.json'
    inspected = smoke.with_name('INSPECTED.json')
    if not smoke.exists() or read(smoke).get('inputs') != pins:
        raise ValueError('Matching actual-DMX GPU smoke required before full render')
    if not inspected.exists() or read(inspected).get('inspected') is not True or read(inspected).get('smoke_sha256') != sha(smoke):
        raise ValueError('Root must inspect smoke frames and authenticate INSPECTED.json')
    if any(sha(p) != h for p, h in read(smoke)['files'].items()):
        raise ValueError('Smoke pixels changed after inspection')
    out = root / 'media' / algorithm['id'] / track['id']
    out.mkdir(parents=True, exist_ok=True)
    movie, sidecar = out / 'full.mp4', out / 'full_render.json'
    if sidecar.exists():
        old = read(sidecar)
        if old['inputs'] != pins or old['codec'] != codec or sha(movie) != old['sha256']:
            raise ValueError('Completed render no longer matches authenticated inputs')
        return old
    if movie.exists():
        raise ValueError('Unreceipted movie preserved; use a new revision')
    partial = out / 'full.partial.mp4'
    audio_reference, _ = audio_paths(root, track)
    command = command_for(partial, audio_reference, track, codec)
    started, pixel_samples = time.monotonic(), []
    strip = TimelineStrip(reference, candidate, music, track)
    with renderer_for(root, candidate.patch, guarded) as (renderer, cfg, receipt):
        with (out / 'full_encode.log').open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(candidate.indices):
                    stages = render_stages(renderer, reference, candidate, tick)
                    pixel_samples.append([float(s[::4, ::4].mean()) for s in stages])
                    encoder.stdin.write(compose(stages, reference, candidate, tick, frame/FPS, algorithm, track, strip).tobytes())
                    if frame % 300 == 0:
                        resource_guard(root, plan)
                        print('RENDER', algorithm['id'], track['id'], frame, len(candidate.indices), flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encoder failed; partial output and log retained')
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
        os.link(partial, movie)
        partial.unlink()
        result = {'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
            'inputs': pins, 'frames': track['video_frames'], 'duration_s': track['video_frames']/FPS,
            'source_duration_s': track['duration_s'], 'start_s': 0., 'audio_speed': 1.,
            'final_frame_padding_s': track['video_frames']/FPS-track['duration_s'],
            'lighting_sample_max_offset_s': float(np.max(np.abs(candidate.times[candidate.indices]-np.arange(track['video_frames'])/FPS))),
            'codec': codec, 'renderer': renderer.renderer_name, 'renderer_config': cfg,
            'gpu_receipt': receipt, 'command': command, 'elapsed_s': time.monotonic()-started,
            'trace_scales': strip.scales, 'smoke_sha256': sha(smoke), 'inspection_sha256': sha(inspected),
            'banner': evidence_summary(candidate.comparison), 'motion_origin': candidate.metadata,
            'scope': 'Experimental full-song composition; D7 reference is not a product baseline.'}
    np.savez_compressed(out / 'full_sampled_pixels.npz', values=np.asarray(pixel_samples))
    save(sidecar, result)
    return result


def audit_pair(root, plan, algorithm, track):
    _, _, _, pins = load_comparison(root, plan, algorithm, track)
    out = root / 'media' / algorithm['id'] / track['id']
    movie = out / 'full.mp4'
    result = read(out / 'full_render.json')
    if result['inputs'] != pins or result['sha256'] != sha(movie):
        raise ValueError('Delivered video source identity changed')
    path = out / 'full_audit.json'
    if path.exists():
        audit = read(path)
    else:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(movie)], text=True))
        video = next(s for s in probe['streams'] if s['codec_type']=='video')
        audio = next(s for s in probe['streams'] if s['codec_type']=='audio')
        expected_codec = {'av1_nvenc': 'av1', 'libx264': 'h264'}[result['codec']]
        if (video['codec_name'], video['width'], video['height'], video['r_frame_rate']) != (expected_codec, WIDTH, HEIGHT, '30/1'):
            raise ValueError('Wrong delivery format')
        frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)], text=True))['frames']
        times = np.array([float(f['best_effort_timestamp_time']) for f in frames])
        if len(times) != track['video_frames'] or not np.allclose(times, np.arange(len(times))/FPS, rtol=0, atol=1e-5):
            raise ValueError('Incomplete or nonuniform decoded frame timestamps')
        if abs(float(audio['duration'])-result['duration_s']) > .06 or abs(float(audio.get('start_time', 0))) > 1/44100:
            raise ValueError('Audio duration/start synchronization failure')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-threads', '2',
            '-i', str(movie), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
        digest = hashlib.sha256(base.decoded_audio(movie)).hexdigest()
        reference, record = audio_paths(root, track)
        expected_digest = hashlib.sha256(base.decoded_audio(reference)).hexdigest()
        if digest != expected_digest or digest != read(record)['decoded_pcm_sha256']:
            raise ValueError('Delivered audio PCM differs from authenticated full-song reference')
        # Save decoded delivery pixels for parent inspection, including the quiet
        # intro and the final source frame. These are read from the actual movie.
        sample_frames = sorted(set([0, min(18*FPS, len(times)-1), min(60*FPS, len(times)-1),
                                    min(120*FPS, len(times)-1), len(times)-1]))
        sample_dir = out / 'decoded_samples'
        sample_dir.mkdir(exist_ok=False)
        expression = '+'.join(f'eq(n,{frame})' for frame in sample_frames)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '2', '-i', str(movie),
            '-vf', 'select='+expression.replace(',', r'\,'), '-fps_mode', 'passthrough', '-frames:v', str(len(sample_frames)),
            str(sample_dir/'%02d.png')], check=True)
        sample_paths = sorted(sample_dir.glob('*.png'))
        if len(sample_paths) != len(sample_frames):
            raise ValueError('Decoded sample frame count differs')
        sheet = Image.new('RGB', (960, 270*len(sample_paths)), (0, 0, 0))
        samples = []
        for index, (frame, sample) in enumerate(zip(sample_frames, sample_paths)):
            with Image.open(sample) as picture:
                if picture.size != (WIDTH, HEIGHT):
                    raise ValueError('Decoded delivery frame dimensions differ')
                sheet.paste(picture.resize((480, 270), Image.Resampling.LANCZOS), (0, index*270))
                # Stage-only views avoid compressing all four body views into a
                # tiny sheet; full 1080p originals remain next to the sheet.
                sheet.paste(picture.crop((960, 324, 1920, 864)).resize((480, 270), Image.Resampling.LANCZOS),
                            (480, index*270))
            samples.append({'frame': frame, 'time_s': frame/FPS, 'path': str(sample), 'sha256': sha(sample)})
        sheet_path = sample_dir/'CONTACT_SHEET.jpg'
        sheet.save(sheet_path, quality=92)
        audit = {'video_sha256': sha(movie), 'inputs': pins, 'decoded_frames': len(times),
            'pts_uniform': True, 'full_software_AV_decode': True, 'pcm_sha256': digest,
            'audio_matches_full_reference': True, 'source_start_s': 0., 'audio_speed': 1.,
            'source_duration_s': track['duration_s'], 'video_duration_s': result['duration_s'],
            'final_padding_s': result['final_frame_padding_s'], 'video': video, 'audio': audio,
            'decoded_samples': samples, 'contact_sheet': str(sheet_path), 'contact_sheet_sha256': sha(sheet_path)}
        save(path, audit)
    if audit['inputs'] != pins or audit['video_sha256'] != sha(movie):
        raise ValueError('Saved media audit is stale')
    _, audio_record = audio_paths(root, track)
    if (audit.get('decoded_frames') != track['video_frames']
            or any(audit.get(key) is not True for key in ('pts_uniform', 'full_software_AV_decode', 'audio_matches_full_reference'))
            or audit.get('pcm_sha256') != read(audio_record)['decoded_pcm_sha256']):
        raise ValueError('Saved audit does not establish complete synchronized delivery')
    if any(sha(sample['path']) != sample['sha256'] for sample in audit['decoded_samples']) or sha(audit['contact_sheet']) != audit['contact_sheet_sha256']:
        raise ValueError('Decoded inspection pixels changed after audit')
    verification = {'video_sha256': sha(movie), 'audit_sha256': sha(path), 'inputs': pins,
        'complete_full_song': True, 'source_audio_unchanged_speed': True}
    verified = out / 'full_verified.json'
    if verified.exists():
        if read(verified) != verification:
            raise ValueError('Existing verification differs')
    else:
        save(verified, verification)
    return verification


def stream_fingerprint(root, method, track, keys):
    digest = hashlib.sha256()
    with np.load(root/'timelines'/method/track['id']/'timeline.npz', allow_pickle=False) as data:
        for key in keys:
            value = np.ascontiguousarray(data[key])
            digest.update(str((key, value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def gallery(root, plan):
    rows, seen = [], {}
    for index, (algorithm, track) in enumerate(base.pairs(plan), 1):
        reference, candidate, _, pins = load_comparison(root, plan, algorithm, track)
        folder = root/'media'/algorithm['id']/track['id']
        result, verified = read(folder/'full_render.json'), read(folder/'full_verified.json')
        if (result['inputs'] != pins or verified['inputs'] != pins or sha(result['path']) != result['sha256']
                or verified['video_sha256'] != result['sha256'] or verified['audit_sha256'] != sha(folder/'full_audit.json')):
            raise ValueError('Gallery only admits authenticated and decoded movies')
        fingerprint = stream_fingerprint(root, algorithm['id'], track, ('time', 'body', 'universes'))
        duplicate = seen.get((track['id'], fingerprint))
        seen.setdefault((track['id'], fingerprint), f'{index:02d}')
        rows.append({'id': f'{index:02d}', 'algorithm_id': algorithm['id'], 'track_id': track['id'],
            'title': algorithm['name']+' / '+track['title'], 'status': 'complete',
            'path': result['path'], 'sha256': result['sha256'], 'bytes': result['bytes'],
            'duration_s': result['duration_s'], 'source_duration_s': track['duration_s'],
            'download_name': f"{algorithm['id']}_{track['id']}_full_1080p.mp4", 'duplicate_of': duplicate,
            'visual_sha256': fingerprint,
            'light_stream_sha256': stream_fingerprint(root, algorithm['id'], track, ('time', 'universes')),
            'motion_origin': candidate.metadata, 'comparison': candidate.comparison,
            'banner': evidence_summary(candidate.comparison), 'inputs': pins})
    manifest = {'campaign': root.name, 'plan_sha256': sha(root/'PLAN.json'), 'tracks': plan['tracks'],
        'algorithms': plan['algorithms'], 'variants': rows, 'source_start_s': 0.,
        'reference': 'D7_reference', 'baseline_promoted': False,
        'scope': 'Four full-song experimental comparisons; original audio brightness restored; artistic review pending.'}
    dest = root/'review'
    dest.mkdir(exist_ok=True)
    if (dest/'MANIFEST.json').exists():
        if read(dest/'MANIFEST.json') != manifest or sha(dest/'index.html') != sha(TEMPLATE):
            raise ValueError('Existing gallery differs; preserve it and use a new revision')
    else:
        save(dest/'MANIFEST.json', manifest)
        with (dest/'index.html').open('xb') as stream:
            stream.write(TEMPLATE.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-audio', 'smoke', 'render', 'audit', 'gallery'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--algorithm', choices=('C3', 'C4'))
    parser.add_argument('--track')
    parser.add_argument('--codec', choices=('av1_nvenc', 'libx264'), default='av1_nvenc')
    parser.add_argument('--guarded', action='store_true')
    parser.add_argument('--schematic', action='store_true', help='CPU smoke only; not a production renderer')
    args = parser.parse_args()
    root = args.root.resolve()
    if sha(root/'PLAN.json') != args.plan_sha256:
        raise ValueError('PLAN digest differs from frozen command')
    plan = read(root/'PLAN.json')
    if args.schematic and args.command != 'smoke':
        raise ValueError('Schematic mode is only an explicit smoke diagnostic')
    if args.command == 'prepare-audio':
        tracks = [t for t in plan['tracks'] if args.track is None or t['id']==args.track]
        if not tracks:
            raise ValueError('Unknown track')
        for track in tracks:
            print(json.dumps(prepare_audio(root, plan, track)), flush=True)
    elif args.command == 'gallery':
        result = gallery(root, plan)
        print('GALLERY_READY', root/'review', len(result['variants']), flush=True)
    else:
        for algorithm, track in base.pairs(plan, args.algorithm, args.track):
            if args.command == 'smoke':
                result = smoke_pair(root, plan, algorithm, track, args.guarded, args.schematic)
            elif args.command == 'render':
                result = render_pair(root, plan, algorithm, track, args.codec, args.guarded)
            else:
                result = audit_pair(root, plan, algorithm, track)
            print('COMPLETE', args.command, algorithm['id'], track['id'], flush=True)


if __name__ == '__main__':
    main()
