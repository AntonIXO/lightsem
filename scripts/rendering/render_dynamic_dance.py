"""Source-authenticated body/DMX comparisons for the dynamic-dance campaign.

Rendering never synthesizes motion. Both skeletons and both fixed-exposure
stages consume the saved 40 Hz timeline. The skeleton panels are diagnostic
drawings, deliberately visible even when the actual stage lights are off.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.rendering import render_body_portfolio as base
from scripts.rendering.render_body_transfer import PARENTS
from lightsem.experiments.dynamic_dance import campaign
from lightsem.experiments.dynamic_dance.diagnostics import dynamics_series

WIDTH, HEIGHT, FPS = 1920, 1080, 30
STAGE_WIDTH, STAGE_HEIGHT, STAGE_Y, TRACE_Y = 960, 540, 324, 864
TEMPLATE = Path(__file__).with_name('dynamic_dance_review.html')
SOURCE_NAMES = ('drums', 'bass', 'other', 'vocals')
SOURCE_COLORS = ((237, 161, 80), (147, 185, 101), (145, 137, 223), (232, 127, 163))
BASE_COLOR, CANDIDATE_COLOR = (152, 163, 180), (90, 212, 236)
UPPER = np.array([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
LOWER = np.array([1, 2, 4, 5, 7, 8, 10, 11])
sha, read, save = base.sha, base.read, base.save


def resource_guard(root, plan):
    """Preserve campaign free-space reserve; impose no legacy total-size cap."""
    campaign.disk_guard(root)
    minimum = plan.get('minimum_free_bytes', 20 * 2 ** 30)
    if minimum > 20 * 2 ** 30 and shutil.disk_usage(root).free < minimum:
        raise RuntimeError('Explicit campaign free-space reserve reached')
    if 'max_bytes' in plan:
        maximum = plan['max_bytes']
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not np.isfinite(maximum) or maximum <= 0:
            raise ValueError('Explicit max_bytes must be a positive finite byte budget')
        size = sum(path.stat().st_size for path in Path(root).rglob('*') if path.is_file())
        if size > maximum:
            raise RuntimeError('Explicit campaign storage budget exceeded')


def variant_metadata(root, logical_method):
    """Expose an authenticated implementation revision without renaming the arm."""
    revision = campaign.variant_metadata(root, logical_method)
    return {'artifact_revision': revision['artifact_id'], 'repair_reason': revision.get('reason'),
            'revision_provenance': revision}


def motion_provenance(root, artifact, logical_method, track, report):
    """Describe the recorded motion origin; fresh-generation claims require proof."""
    metadata = {}
    pins = {}
    expected = report.get('inputs', {}).get('body_metadata_sha256')
    if expected is not None:
        path = root / 'bodies' / artifact / track['id'] / 'body.npz'
        sidecar = path.with_suffix('.json')
        seal = sidecar.with_name(sidecar.name + '.sha256.json')
        if sha(sidecar) != expected or read(seal).get('sha256') != expected:
            raise ValueError('Motion provenance metadata differs from mapped body')
        details = read(sidecar)
        if (details.get('kind') != 'DanceResult' or details.get('npz_sha256') != sha(path)
                or report.get('body_sha256') != details.get('npz_sha256')):
            raise ValueError('Motion provenance does not identify the mapped body')
        metadata = details.get('metadata', {})
        pins = {'body_source_sha256': details['npz_sha256'], 'body_metadata_sha256': expected,
                'body_metadata_seal_sha256': sha(seal)}
    if logical_method in ('D7', 'D8'):
        if metadata.get('newly_generated_motion') is not True or not metadata.get('checkpoint_sha256'):
            raise ValueError('Newly generated D7/D8 motion requires authenticated generation provenance')
        kind, label = 'newly_generated', 'newly generated native30Hz body'
    elif logical_method == 'D4':
        kind, label = 'recomposed_fragments', 'recomposed same-song body fragments'
    elif logical_method == 'D5':
        kind, label = 'refined_trajectory', 'trajectory-refined same-song body'
    elif logical_method == 'D6':
        kind, label = 'learned_retiming', 'learned retiming of same-song source body'
    elif logical_method == 'baseline':
        kind, label = 'saved_baseline', 'original per-song generated body'
    else:
        kind, label = 'retimed_source', 'retimed same-song source body'
    fields = ('arm', 'method', 'newly_generated_motion', 'motion_origin', 'source_clock', 'source_body',
              'source_body_sha256', 'checkpoint', 'checkpoint_sha256', 'seed', 'draw_seed', 'conditioning',
              'timeline_path', 'timeline_sha256', 'feature_schema_sha256', 'source_time_jumps_are_fragment_boundaries')
    boundaries = []
    if logical_method == 'D4':
        for fragment in metadata.get('fragment_provenance', []):
            interval = fragment.get('target_seconds', [])
            if len(interval) != 2 or not np.isfinite(interval).all():
                raise ValueError('Invalid recorded D4 fragment boundary')
            boundaries.append(float(interval[0]))
    return {'kind': kind, 'label': label, 'metadata': {k: metadata[k] for k in fields if k in metadata},
            'fragment_boundaries_s': sorted(set(boundaries)), 'pins': pins}


@dataclass
class Pair:
    controls: dict
    body: np.ndarray
    patch: list
    indices: np.ndarray
    report: dict
    pins: dict
    times: np.ndarray
    rate: np.ndarray | None
    source_time: np.ndarray | None
    interventions: np.ndarray
    comparison: dict | None = None
    algorithm_id: str = ''
    artifact_revision: str = ''
    repair_reason: str | None = None
    motion_origin: dict = field(default_factory=dict)


def load_pair(root, plan, algorithm, track):
    """Admit only the exact body and emitted DMX checked by numeric preflight."""
    artifact = campaign.resolve_variant(root, algorithm['id'])
    controls, body, patch, indices, report, pins = base.load_pair(root, plan, {**algorithm, 'id': artifact}, track)
    folder = root / 'timelines' / artifact / track['id']
    revision = variant_metadata(root, algorithm['id'])
    pins['artifact_revision'] = artifact
    pins['repair_reason'] = revision.get('repair_reason')
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        times = data['time'].copy()
        source_time = data['source_time'].copy() if 'source_time' in data else None
        rate = data['playback_rate'].copy() if 'playback_rate' in data else None
        interventions = data['intervention_time'].copy() if 'intervention_time' in data else np.empty(0)
    for name, values in (('source_time', source_time), ('playback_rate', rate)):
        if values is not None and (values.ndim not in (1, 2) or len(values) != len(times)
                                   or not np.isfinite(values).all()):
            raise ValueError('Invalid ' + name + ' metadata')
    if source_time is not None and rate is None:
        if algorithm['id'] == 'D4':
            raise ValueError('D4 requires recorded fragment-local playback_rate; source_time jumps are provenance')
        derivative = np.diff(source_time, axis=0) / np.diff(times).reshape((-1,) + (1,) * (source_time.ndim - 1))
        rate = np.concatenate((derivative, derivative[-1:]))
    if interventions.ndim != 1 or not np.isfinite(interventions).all():
        raise ValueError('Invalid intervention timestamps')
    counts = {kind: sum(p['kind'] == kind for p in patch) for kind in ('beam', 'laser', 'bulb')}
    if counts != {'beam': 8, 'laser': 5, 'bulb': 50}:
        raise ValueError('Delivery rig must contain 8 beams, 5 lasers and 50 bulbs')
    comparison_path = folder / 'comparison.json'
    comparison = read(comparison_path) if comparison_path.exists() else None
    pins['comparison_sha256'] = sha(comparison_path) if comparison is not None else None
    if comparison is not None:
        if comparison.get('method') != algorithm['id'] or comparison.get('song') != track['id']:
            raise ValueError('Comparison report identifies a different method or song')
        for name, expected in (('timeline.npz', pins['timeline_sha256']),
                               ('patch.json', pins['patch_sha256']), ('report.json', pins['numeric_report_sha256'])):
            if comparison.get('inputs', {}).get(str(folder / name)) != expected:
                raise ValueError('Comparison report does not authenticate displayed ' + name)
    origin = motion_provenance(root, artifact, algorithm['id'], track, report)
    pins.update(origin['pins'])
    if algorithm['id'] == 'D4' and origin['fragment_boundaries_s']:
        interventions = np.unique(np.r_[interventions, origin['fragment_boundaries_s']])
    return Pair(controls, body, patch, indices, report, pins, times, rate, source_time, interventions,
                comparison, algorithm['id'], artifact, revision.get('repair_reason'), origin)


def load_music(root, track):
    """Read measured stem envelopes and estimated musical boundaries, not labels of truth."""
    path = Path(track.get('music_timeline', root / 'features' / 'timelines' / (track['id'] + '.npz')))
    metadata_path = path.with_suffix('.json')
    with np.load(path, allow_pickle=False) as data:
        music = {k: data[k].copy() for k in ('time', 'sources', 'beats', 'downbeats')}
    metadata = read(metadata_path)
    music['sections'] = metadata.get('sections', [])
    t, sources = music['time'], music['sources']
    if (t.ndim != 1 or len(t) < 2 or sources.shape != (len(t), 4, 4)
            or not np.isfinite(t).all() or not np.isfinite(sources).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid measured MusicTimeline source arrays')
    for key in ('beats', 'downbeats'):
        if music[key].ndim != 1 or not np.isfinite(music[key]).all():
            raise ValueError('Invalid musical event timestamps')
    for section in music['sections']:
        if not np.isfinite([section['start'], section['end']]).all() or section['end'] < section['start']:
            raise ValueError('Invalid estimated section interval')
    return music, {'music_timeline_sha256': sha(path), 'music_metadata_sha256': sha(metadata_path)}


def authenticate_audio_binding(root, plan, track):
    """Bind render audio without rewriting the original scientific PLAN.

    RENDER_BINDINGS schema 1 records original PLAN digest/crop and a tracks map.
    Each track binds source/reference paths and hashes. ``legacy_plan_pin``
    evidence must itself pin both files and identify the exact song and crop.
    A directly frozen PLAN.files reference remains supported for older callers.
    """
    reference = Path(track['audio_reference'])
    reference_sha = sha(reference)
    binding_path = root / 'RENDER_BINDINGS.json'
    if not binding_path.exists():
        if plan.get('files', {}).get(str(reference)) != reference_sha:
            raise RuntimeError('Audio reference needs a PLAN.files pin or authenticated RENDER_BINDINGS.json')
        return {'render_bindings_sha256': None, 'audio_reference_sha256': reference_sha,
                'audio_reference_binding_basis': 'plan_files'}
    binding = read(binding_path)
    if binding.get('schema_version') != 1 or binding.get('plan_sha256') != sha(root / 'PLAN.json'):
        raise ValueError('Render audio binding has stale PLAN identity or unsupported schema')
    for key in ('review_start_s', 'review_duration_s'):
        if binding.get(key) != plan[key]:
            raise ValueError('Render audio binding crop differs from original PLAN')
    if not isinstance(binding.get('tracks'), dict) or set(binding['tracks']) != {t['id'] for t in plan['tracks']}:
        raise ValueError('Render audio binding must cover exactly the original PLAN tracks')
    for original in plan['tracks']:
        row = binding['tracks'][original['id']]
        for key in ('audio', 'audio_reference'):
            filename = original[key]
            if row.get(key) != filename or not Path(filename).is_absolute() or row.get(key + '_sha256') != sha(filename):
                raise ValueError('Render audio binding has changed source/reference: ' + key)
        if plan.get('files', {}).get(original['audio']) != row['audio_sha256']:
            raise ValueError('Render audio source is not pinned by the original PLAN')
        proof = row.get('verification', {})
        evidence_path = Path(proof.get('evidence_path', ''))
        if (proof.get('basis') != 'legacy_plan_pin' or not evidence_path.is_absolute()
                or not evidence_path.is_file() or proof.get('evidence_sha256') != sha(evidence_path)):
            raise ValueError('Invalid render audio legacy-plan evidence')
        legacy = read(evidence_path)
        if any(legacy.get(key) != binding[key] for key in ('review_start_s', 'review_duration_s')):
            raise ValueError('Legacy audio evidence uses a different source crop')
        matches = [t for t in legacy.get('tracks', []) if t.get('id') == original['id']]
        if (len(matches) != 1 or any(matches[0].get(key) != original[key] for key in ('audio', 'audio_reference'))
                or any(legacy.get('files', {}).get(original[key]) != row[key + '_sha256']
                       for key in ('audio', 'audio_reference'))):
            raise ValueError('Legacy plan does not pin this source/reference pair')
    return {'render_bindings_sha256': sha(binding_path), 'audio_reference_sha256': reference_sha,
            'audio_reference_binding_basis': 'legacy_plan_pin'}


def load_comparison(root, plan, algorithm, track):
    candidate = load_pair(root, plan, algorithm, track)
    baseline = load_pair(root, plan, {'id': 'baseline'}, track)
    if candidate.patch != baseline.patch or not np.array_equal(candidate.times, baseline.times):
        raise ValueError('Baseline/candidate patch and source clocks must be identical')
    if candidate.comparison is not None:
        inputs = candidate.comparison.get('inputs', {})
        expected = {str(root / 'PLAN.json'): sha(root / 'PLAN.json'), **campaign.variant_binding_pins(root)}
        for filename, key in (('timeline.npz', 'timeline_sha256'), ('patch.json', 'patch_sha256'),
                              ('report.json', 'numeric_report_sha256')):
            expected[str(root / 'timelines' / baseline.artifact_revision / track['id'] / filename)] = baseline.pins[key]
        if any(inputs.get(path) != value for path, value in expected.items()):
            raise ValueError('Comparison report does not authenticate displayed baseline/plan/variant bindings')
    music, music_pins = load_music(root, track)
    if music['time'][0] > 0 or music['time'][-1] < plan['prefix_s'] - 1 / 20:
        raise ValueError('Musical analysis does not cover the source interval')
    pins = {'candidate': candidate.pins, 'baseline': baseline.pins, **music_pins,
            **authenticate_audio_binding(root, plan, track),
            'variant_binding_files': campaign.variant_binding_pins(root),
            'renderer_sha256': sha(Path(__file__)), 'template_sha256': sha(TEMPLATE)}
    return baseline, candidate, music, pins


def trace_values(pair):
    """Physical derivatives of the displayed body and decoded actuator angles."""
    # Centered gradients erase alternating-frame jitter; use the registered
    # forward-interval instrument and repeat the final interval for plotting.
    speed = np.linalg.norm(dynamics_series(pair.body, pair.times)['relative_velocity_m_s'], axis=-1)
    moving = np.array([p['kind'] != 'bulb' for p in pair.patch])
    angles = pair.controls['angles_deg'][:, moving]
    angular = np.linalg.norm(np.diff(angles, axis=0) / np.diff(pair.times)[:, None, None], axis=-1).mean(axis=1)
    angular = np.r_[angular, angular[-1]]
    return {'upper': speed[:, UPPER].mean(axis=1), 'lower': speed[:, LOWER].mean(axis=1),
            'emitted': angular}


def evidence_summary(pair):
    """Keep engineering, proxy magnitude, missing evidence and artistic review distinct."""
    comparison = pair.comparison if isinstance(pair, Pair) else pair
    if comparison is None:
        return {'comparison_status': 'missing_comparison', 'physical_engineering_status': 'unqualified',
                'primary': None, 'missing_controls': [], 'missing_evidence': ['comparison.json'],
                'failure_notes': [], 'banner': 'UNQUALIFIED | comparison evidence missing; artistic review pending'}
    status = comparison.get('status', 'unqualified')
    physical = comparison.get('physical_engineering_status', 'unqualified')
    primary = comparison.get('primary') or {}
    reasons = list(comparison.get('physical_failure_reasons', []))
    method_postgate = comparison.get('method_postgate') or {}
    if method_postgate.get('passed') is False:
        reasons.extend(str(reason) for reason in method_postgate.get('infeasibility', []))
    if status.startswith('failed_'):
        reasons.append(status)
    for key in ('failure', 'error'):
        if comparison.get(key):
            reasons.append(str(comparison[key]))
    reasons = list(dict.fromkeys(reasons))
    missing_controls = sorted(comparison.get('missing_controls', {}))
    missing_evidence = list(primary.get('missing_evidence', []))
    magnitude = primary.get('magnitude_gate', 'unqualified')
    if physical == 'fail' or status in ('failure_diagnostic',) or status.startswith('failed_'):
        banner = 'FAILED ENGINEERING | ' + ', '.join(reasons or [status])
    elif magnitude == 'fail':
        banner = f'PROXY MAGNITUDE FAILED | physical engineering: {physical}; artistic review pending'
    elif physical != 'pass' or missing_controls or missing_evidence:
        banner = f'UNQUALIFIED | physical engineering: {physical}; {len(missing_controls)} missing controls; proxy: {magnitude}'
    else:
        banner = f'ENGINEERING {physical.upper()} | proxy magnitude {magnitude}; artistic review pending'
    return {'comparison_status': status, 'physical_engineering_status': physical, 'primary': primary,
            'missing_controls': missing_controls, 'missing_evidence': missing_evidence,
            'failure_notes': reasons, 'banner': banner}


@lru_cache(maxsize=1)
def fonts():
    return tuple(ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
                 for size in (25, 19, 15))


def body_panel(draw, pose, x, width=480, oblique=False):
    """Fixed orthographic camera; root horizontal location is shown separately."""
    _, normal, small = fonts()
    centered = pose - pose[0] * np.array([1., 0., 1.])
    horizontal = .8 * centered[:, 0] + .6 * centered[:, 2] if oblique else centered[:, 0]
    xy = np.column_stack((x + width / 2 + horizontal * 106, 306 - centered[:, 1] * 106))
    draw.rectangle((x, 86, x + width - 1, 322), fill=(238, 241, 243))
    draw.line((x, 307, x + width, 307), fill=(174, 182, 190), width=1)
    draw.text((x + 16, 91), 'oblique' if oblique else 'front', font=normal, fill=(48, 61, 75))
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            draw.line([tuple(xy[parent]), tuple(xy[joint])],
                      fill=(39, 114, 156) if joint % 2 == 0 else (117, 74, 145), width=3)
    for px, py in xy:
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(34, 67, 93))
    if not oblique:
        draw.text((x + 16, 299), 'root xyz ' + ' '.join(f'{v:+.2f}' for v in pose[0]),
                  font=small, fill=(44, 59, 72))


class TimelineStrip:
    """Static measured traces plus a moving inspection cursor; no generated controls."""
    def __init__(self, baseline, candidate, music, plan):
        self.start = float(plan['review_start_s'])
        self.end = self.start + float(plan['review_duration_s'])
        self.left, self.right = 340, WIDTH - 20
        self.image = Image.new('RGB', (WIDTH, HEIGHT - TRACE_Y), (13, 19, 29))
        draw = ImageDraw.Draw(self.image)
        _, _, font = fonts()
        draw.text((12, 4), 'Measured audio + actual body / emitted DMX', font=font, fill=(216, 225, 239))
        draw.text((590, 4), 'body: gray baseline / cyan candidate   |   sources: drums / bass / other / vocals',
                  font=font, fill=(166, 185, 207))
        base_values, candidate_values = trace_values(baseline), trace_values(candidate)
        self.scales = {}
        rate_title = 'Fragment-local rate (x)' if candidate.algorithm_id == 'D4' else 'Source playback rate (x)'
        rows = [('sources', 27, 'Source RMS (per-source p95)'), ('upper', 60, 'Upper relative speed (m/s)'),
                ('lower', 93, 'Lower relative speed (m/s)'), ('emitted', 126, 'Emitted angle speed (deg/s)'),
                ('rate', 159, rate_title)]
        for key, y, label in rows:
            draw.text((12, y + 2), label, font=font, fill=(177, 192, 213))
            draw.line((self.left, y + 28, self.right, y + 28), fill=(46, 56, 72))
            if key == 'sources':
                for index, color in enumerate(SOURCE_COLORS):
                    values = music['sources'][:, index, 0]
                    scale = max(float(np.quantile(values, .95)), 1e-9)
                    self.line(draw, music['time'], values, y, scale, color)
                self.scales[key] = 'each measured source normalized by own full-song p95 RMS; clipped for display'
            elif key == 'rate':
                self.scales[key] = 1.5
                if candidate.rate is None:
                    label = candidate.motion_origin.get('label', 'saved pose trajectory')
                    draw.text((self.left + 4, y + 2), 'No source playback clock: ' + label,
                              font=font, fill=CANDIDATE_COLOR)
                else:
                    rate = candidate.rate[:, None] if candidate.rate.ndim == 1 else candidate.rate
                    for index in range(rate.shape[1]):
                        self.line(draw, candidate.times, rate[:, index], y, 1.5,
                                  CANDIDATE_COLOR if rate.shape[1] == 1 else SOURCE_COLORS[index % 4])
                    self.line(draw, baseline.times, np.ones(len(baseline.times)), y, 1.5, BASE_COLOR)
            else:
                scale = max(float(np.max(base_values[key])), float(np.max(candidate_values[key])), 1e-9)
                self.scales[key] = scale
                draw.text((290, y + 2), f'{scale:.2g}', font=font, fill=(137, 155, 177))
                self.line(draw, baseline.times, base_values[key], y, scale, BASE_COLOR)
                self.line(draw, candidate.times, candidate_values[key], y, scale, CANDIDATE_COLOR)
        for t in music['downbeats']:
            if self.start <= t < self.end:
                px = self.x(t)
                draw.line((px, 24, px, 30), fill=(138, 159, 181))
        for section in music['sections']:
            if self.start <= section['start'] < self.end:
                px = self.x(section['start'])
                draw.line((px, 24, px, 190), fill=(64, 70, 89))
                draw.text((min(px + 3, self.right - 85), 195), str(section.get('label', 'section'))[:12],
                          font=font, fill=(191, 170, 208))
        for t in candidate.interventions:
            if self.start <= t < self.end:
                px = self.x(t)
                draw.polygon([(px - 4, 190), (px + 4, 190), (px, 196)], fill=(252, 199, 93))
        marker = 'fragment cuts' if candidate.algorithm_id == 'D4' else 'plan anchors'
        draw.text((12, 195), 'Sections estimated; yellow = ' + marker, font=font, fill=(140, 158, 181))

    def x(self, seconds):
        return int(round(self.left + (seconds - self.start) / (self.end - self.start) * (self.right - self.left)))

    def line(self, draw, times, values, y, scale, color):
        # Interpolation affects only graph pixels; full motion stays in the timeline.
        xs = np.arange(self.left, self.right + 1)
        ts = self.start + (xs - self.left) / (self.right - self.left) * (self.end - self.start)
        values = np.interp(ts, times, values)
        yy = y + 28 - np.clip(values / scale, 0, 1) * 25
        draw.line(list(zip(xs.tolist(), yy.tolist())), fill=color, width=1)

    def frame(self, seconds):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        px = self.x(seconds)
        draw.line((px, 24, px, 192), fill=(249, 246, 215), width=2)
        return image


def compose(baseline_stage, candidate_stage, baseline, candidate, tick, algorithm, track, strip, diagnostic=''):
    """Assemble exactly the supplied body frame and stage pixels at one timestamp."""
    if baseline_stage.shape != (STAGE_HEIGHT, STAGE_WIDTH, 3) or candidate_stage.shape != baseline_stage.shape:
        raise ValueError('Expected matched 960x540 stage images')
    picture = Image.new('RGB', (WIDTH, HEIGHT), (8, 12, 19))
    picture.paste(Image.fromarray(baseline_stage), (0, STAGE_Y))
    picture.paste(Image.fromarray(candidate_stage), (STAGE_WIDTH, STAGE_Y))
    picture.paste(strip.frame(float(candidate.times[tick])), (0, TRACE_Y))
    draw = ImageDraw.Draw(picture)
    title, normal, small = fonts()
    draw.text((16, 8), f"{algorithm['id']} | {algorithm['name']} | {track['title']}", font=title, fill=(233, 240, 249))
    status = evidence_summary(candidate)['banner']
    draw.text((16, 43), (diagnostic or status)[:172], font=small, fill=(241, 190, 114))
    draw.text((1425, 12), f"Song time {candidate.times[tick]:06.2f}s", font=normal, fill=(234, 241, 249))
    if candidate.rate is None:
        rate_label = 'Time warp: none'
    else:
        value = np.asarray(candidate.rate[tick])
        if candidate.algorithm_id == 'D4':
            rate_label = f'Fragment rate {value.mean():.2f}x'
            if candidate.source_time is not None:
                rate_label += f' | source {np.asarray(candidate.source_time[tick]).mean():.2f}s'
        else:
            rate_label = f'Playback {float(value):.2f}x' if value.ndim == 0 else f'Regional rate {value.min():.2f}–{value.max():.2f}x'
    draw.text((1425, 42), rate_label, font=small, fill=CANDIDATE_COLOR)
    candidate_label = 'CANDIDATE | ' + candidate.motion_origin.get('label', 'saved body')
    if candidate.artifact_revision != candidate.algorithm_id:
        candidate_label += ' | artifact ' + candidate.artifact_revision
    for x, label, pair in ((0, 'BASELINE | this song original body', baseline),
                           (960, candidate_label, candidate)):
        draw.text((x + 16, 64), label, font=small, fill=(193, 211, 230))
        body_panel(draw, pair.body[tick], x)
        body_panel(draw, pair.body[tick], x + 480, oblique=True)
        draw.text((x + 14, STAGE_Y + 12), 'ACTUAL EMITTED DMX | fixed P2 / exposure | 8 beams + 5 lasers + 50 bulbs',
                  font=small, fill=(163, 188, 209))
    draw.line((959, 62, 959, TRACE_Y), fill=(59, 78, 101), width=2)
    draw.text((15, TRACE_Y - 24), 'Skeleton panels are diagnostic drawings; stage panels use actual light levels.',
              font=small, fill=(163, 188, 209))
    return np.asarray(picture)


def require_guard(root):
    """An outer campaign guard owns GPU locks; verify its live parent receipt."""
    from scripts.motion.dynamic_dance_guard import authenticate_guarded_job
    receipt = authenticate_guarded_job(__file__)
    command = receipt['identity']['command']
    if '--root' not in command or Path(command[command.index('--root') + 1]).resolve() != root.resolve():
        raise ValueError('Guarded renderer command does not bind the campaign root')
    return receipt


@contextmanager
def renderer_for(root, mode, patch, guarded=False):
    if mode == 'schematic':
        renderer = base.SchematicRenderer(patch)
        class SizedSchematic:
            renderer_name = renderer.renderer_name
            def render_controls(self, *controls):
                return np.asarray(Image.fromarray(renderer.render_controls(*controls)).resize(
                    (STAGE_WIDTH, STAGE_HEIGHT), Image.Resampling.LANCZOS))
        yield SizedSchematic(), None, None
        return
    @contextmanager
    def admission():
        if guarded:
            yield require_guard(root)
        else:
            with base.gpu_admission(root) as release:
                yield release
    with admission() as release:
        from lightsem.sim.body_stage_renderer import BodyStageRenderer
        from scripts.motion.run_rhythm_companion import render_config
        cfg = render_config(STAGE_WIDTH, STAGE_HEIGHT)
        cfg.camera_scale = .62
        renderer = BodyStageRenderer(patch, cfg)
        try:
            yield renderer, asdict(cfg), release
        finally:
            renderer.ctx.release()


def render_stages(renderer, baseline, candidate, tick):
    return tuple(renderer.render_controls(*(pair.controls[k][tick] for k in ('directions', 'rgb', 'dimmer')))
                 for pair in (baseline, candidate))


def smoke_pair(root, plan, algorithm, track, mode, guarded=False):
    baseline, candidate, music, pins = load_comparison(root, plan, algorithm, track)
    out = root / 'smoke' / algorithm['id'] / track['id'] / mode
    receipt_path = out / 'SMOKE.json'
    if receipt_path.exists():
        old = read(receipt_path)
        if old['inputs'] != pins or any(sha(p) != h for p, h in old['files'].items()):
            raise RuntimeError('Existing smoke artifacts changed')
        return old
    out.mkdir(parents=True, exist_ok=False)
    strip = TimelineStrip(baseline, candidate, music, plan)
    files, samples = {}, []
    with renderer_for(root, mode, candidate.patch, guarded) as (renderer, cfg, release):
        for tick in candidate.indices[np.linspace(0, len(candidate.indices) - 1, 4).round().astype(int)]:
            stages = render_stages(renderer, baseline, candidate, tick)
            repeated = render_stages(renderer, baseline, candidate, tick)
            if any(not np.array_equal(a, b) for a, b in zip(stages, repeated)):
                raise RuntimeError('Frozen DMX changes renderer pixels')
            path = out / f'{int(tick):06d}_actual.png'
            Image.fromarray(compose(*stages, baseline, candidate, tick, algorithm, track, strip)).save(path)
            files[str(path)] = sha(path)
            samples.append({'tick': int(tick), 'time_s': float(candidate.times[tick]), 'frozen_pixels_exact': True,
                            'baseline_stage_mean': float(stages[0].mean()), 'candidate_stage_mean': float(stages[1].mean())})
        # Explicit optical smoke controls. Never reused in production video.
        tick = candidate.indices[len(candidate.indices) // 2]
        directions = candidate.controls['directions'][tick]
        rgb, off = np.ones((len(candidate.patch), 3)), np.zeros(len(candidate.patch))
        dark = renderer.render_controls(directions, rgb, off)
        visibility = {}
        for kind in ('beam', 'laser', 'bulb'):
            dimmer = np.array([.65 if p['kind'] == kind else 0. for p in candidate.patch])
            illuminated = renderer.render_controls(directions, rgb, dimmer)
            visibility[kind] = {'changed_pixels': int(np.any(illuminated != dark, axis=-1).sum()),
                                'absolute_pixel_difference': int(np.abs(illuminated.astype(int) - dark.astype(int)).sum())}
            if visibility[kind]['changed_pixels'] == 0:
                raise RuntimeError('Fixture class invisible in optical smoke: ' + kind)
            path = out / ('optical_' + kind + '_only.png')
            Image.fromarray(illuminated).save(path)
            files[str(path)] = sha(path)
        path = out / 'optical_all_lights_off_with_body.png'
        Image.fromarray(compose(dark, dark, baseline, candidate, tick, algorithm, track, strip,
            'OPTICAL CONTROL ONLY: all stage dimmers zero; diagnostic skeleton remains visible')).save(path)
        files[str(path)] = sha(path)
    receipt = {'inputs': pins, 'files': files, 'samples': samples, 'mode': mode, 'renderer': renderer.renderer_name,
               'renderer_config': cfg, 'gpu_release': release, 'fixture_class_visibility': visibility,
               'artifact_revision': candidate.artifact_revision, 'repair_reason': candidate.repair_reason,
               'motion_origin': candidate.motion_origin,
               'trace_scales': strip.scales, 'optical_controls_excluded_from_production': True}
    save(receipt_path, receipt)
    return receipt


def render_pair(root, plan, algorithm, track, mode, codec, guarded=False):
    baseline, candidate, music, pins = load_comparison(root, plan, algorithm, track)
    smoke = root / 'smoke' / algorithm['id'] / track['id'] / mode / 'SMOKE.json'
    inspected = smoke.with_name('INSPECTED.json')
    if not smoke.exists() or read(smoke)['inputs'] != pins:
        raise RuntimeError('Run smoke before full rendering')
    if (not inspected.exists() or read(inspected).get('smoke_sha256') != sha(smoke)
            or read(inspected).get('inspected') is not True):
        raise RuntimeError('Parent must record inspected=true and smoke_sha256 in INSPECTED.json after visual review')
    for path, expected in read(smoke)['files'].items():
        if sha(path) != expected:
            raise RuntimeError('Smoke pixels changed after inspection')
    out = root / 'media' / algorithm['id'] / track['id']
    out.mkdir(parents=True, exist_ok=True)
    movie, sidecar = out / (mode + '.mp4'), out / (mode + '_render.json')
    if sidecar.exists():
        old = read(sidecar)
        if old['inputs'] != pins or old['codec'] != codec or sha(movie) != old['sha256']:
            raise RuntimeError('Existing completed render differs from frozen inputs')
        return old
    if movie.exists():
        raise RuntimeError('Partial movie retained; use a new campaign revision: ' + str(movie))
    command = base.command_for(movie, track, plan, codec)
    started, pixels = time.monotonic(), []
    strip = TimelineStrip(baseline, candidate, music, plan)
    with renderer_for(root, mode, candidate.patch, guarded) as (renderer, cfg, release):
        with (out / (mode + '_encode.log')).open('xb') as log:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame, tick in enumerate(candidate.indices):
                    stages = render_stages(renderer, baseline, candidate, tick)
                    pixels.append([float(stage[::4, ::4].mean()) for stage in stages])
                    encoder.stdin.write(compose(*stages, baseline, candidate, tick, algorithm, track, strip).tobytes())
                    if frame % 300 == 0:
                        resource_guard(root, plan)
                        print('RENDER', algorithm['id'], track['id'], mode, frame, len(candidate.indices), flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encoder failed; inspect ' + str(log.name))
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
        result = {'path': str(movie), 'sha256': sha(movie), 'bytes': movie.stat().st_size,
                  'inputs': pins, 'frames': len(candidate.indices), 'duration_s': len(candidate.indices) / FPS,
                  'start_s': plan['review_start_s'], 'codec': codec, 'mode': mode, 'renderer': renderer.renderer_name,
                  'renderer_config': cfg, 'gpu_release': release, 'command': command,
                  'elapsed_s': time.monotonic() - started, 'numeric_status': candidate.report['status'],
                  'artifact_revision': candidate.artifact_revision, 'repair_reason': candidate.repair_reason,
                  'motion_origin': candidate.motion_origin,
                  'trace_scales': strip.scales, 'smoke_sha256': sha(smoke), 'inspection_sha256': sha(inspected),
                  'scope': 'Actual saved body and emitted DMX; source analysis estimates; artistic review pending'}
    np.savez_compressed(out / (mode + '_sampled_pixels.npz'), values=np.asarray(pixels))
    save(sidecar, result)
    return result


def audit_pair(root, plan, algorithm, track, mode):
    _, _, _, pins = load_comparison(root, plan, algorithm, track)
    out = root / 'media' / algorithm['id'] / track['id']
    movie, path = out / (mode + '.mp4'), out / (mode + '_audit.json')
    result = read(out / (mode + '_render.json'))
    if result['inputs'] != pins or sha(movie) != result['sha256']:
        raise RuntimeError('Render provenance/hash changed')
    if path.exists():
        audit = read(path)
    elif result['codec'] == 'av1_nvenc':
        from scripts.motion.run_rhythm_companion import audit_movie
        audit = audit_movie(out, movie)
    else:
        audit = base.audit_h264(movie, result)
        save(path, audit)
    reference = Path(track['audio_reference'])
    # load_comparison authenticates either the original PLAN pin or the separate
    # render binding, whose digest is retained in the immutable render receipt.
    if sha(reference) != pins['audio_reference_sha256']:
        raise RuntimeError('Authenticated audio crop changed')
    expected_pcm = hashlib.sha256(base.decoded_audio(reference)).hexdigest()
    if (audit['pcm_sha256'] != expected_pcm or audit['decoded_frames'] != result['frames']
            or not audit['pts_uniform'] or not audit['full_software_AV_decode']):
        raise RuntimeError('Incomplete decode or delivered audio differs from unchanged source crop')
    receipt = {'video_sha256': result['sha256'], 'audit_sha256': sha(path), 'inputs': pins,
               'source_reference_sha256': sha(reference), 'audio_matches_source_crop': True}
    receipt_path = out / (mode + '_verified.json')
    if receipt_path.exists():
        if read(receipt_path) != receipt:
            raise RuntimeError('Existing media verification differs')
    else:
        save(receipt_path, receipt)
    return receipt


def stream_fingerprint(root, algorithm, track, keys):
    """Exact selected stream identity, excluding nonvisual metadata."""
    artifact = campaign.resolve_variant(root, algorithm['id'])
    path = root / 'timelines' / artifact / track['id'] / 'timeline.npz'
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            array = np.ascontiguousarray(data[key])
            h.update(str((key, array.shape, array.dtype.str)).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def visual_fingerprint(root, algorithm, track):
    return stream_fingerprint(root, algorithm, track, ('body', 'universes', 'time'))


def lighting_fingerprint(root, algorithm, track):
    return stream_fingerprint(root, algorithm, track, ('universes', 'time'))


def review(root, plan, mode):
    variants, duplicates, light_duplicates = [], {}, {}
    for index, (algorithm, track) in enumerate(base.pairs(plan), 1):
        row = {'id': f'{index:02d}', 'algorithm_id': algorithm['id'], 'track_id': track['id'],
               'title': algorithm['name'] + ' / ' + track['title'], 'status': 'planned', 'mode': mode}
        row.update(variant_metadata(root, algorithm['id']))
        folder = root / 'timelines' / row['artifact_revision'] / track['id']
        report_path = folder / 'report.json'
        if report_path.exists():
            report = read(report_path)
            row.update(numeric_status=report.get('status'), diagnostics=report.get('diagnostics', report),
                       failure=report.get('failure') or report.get('error'))
            row['motion_origin'] = motion_provenance(root, row['artifact_revision'], algorithm['id'], track, report)
        comparison_path = folder / 'comparison.json'
        comparison = read(comparison_path) if comparison_path.exists() else None
        row.update(evidence_summary(comparison))
        row['comparison'] = comparison
        row['comparison_sha256'] = sha(comparison_path) if comparison is not None else None
        out = root / 'media' / algorithm['id'] / track['id']
        sidecar, verification = out / (mode + '_render.json'), out / (mode + '_verified.json')
        if sidecar.exists() and verification.exists():
            result, receipt = read(sidecar), read(verification)
            _, _, _, pins = load_comparison(root, plan, algorithm, track)
            if (result['inputs'] != pins or receipt['inputs'] != pins or sha(result['path']) != result['sha256']
                    or receipt['video_sha256'] != result['sha256'] or sha(out / (mode + '_audit.json')) != receipt['audit_sha256']):
                raise RuntimeError('Review media provenance changed')
            fingerprint = visual_fingerprint(root, algorithm, track)
            lighting = lighting_fingerprint(root, algorithm, track)
            baseline_lighting = lighting_fingerprint(root, {'id': 'baseline'}, track)
            baseline_visual = visual_fingerprint(root, {'id': 'baseline'}, track)
            key = (track['id'], fingerprint)
            duplicate_of = duplicates.get(key)
            if duplicate_of is None:
                duplicates[key] = row['id']
            light_key = (track['id'], lighting)
            light_duplicate_of = 'baseline' if lighting == baseline_lighting else light_duplicates.get(light_key)
            if light_key not in light_duplicates:
                light_duplicates[light_key] = row['id']
            row.update(status='complete', path=result['path'], sha256=result['sha256'], bytes=result['bytes'],
                       download_name=f"{algorithm['id']}_{track['id']}_{mode}.mp4",
                       visual_sha256=fingerprint, duplicate_of=duplicate_of, light_stream_sha256=lighting,
                       light_duplicate_of=light_duplicate_of,
                       full_visual_matches_baseline=fingerprint == baseline_visual,
                       light_only_duplicate=(light_duplicate_of is not None and duplicate_of is None
                                             and fingerprint != baseline_visual))
        variants.append(row)
    manifest = {'campaign': root.name, 'mode': mode, 'tracks': plan['tracks'], 'algorithms': plan['algorithms'],
                'review_start_s': plan['review_start_s'], 'review_duration_s': plan['review_duration_s'],
                'variants': variants, 'plan_sha256': sha(root / 'PLAN.json'),
                'variant_binding_files': campaign.variant_binding_pins(root),
                'scope': 'Eight distinct experimental approaches. Failed or uncertain approaches remain diagnostics; no artistic acceptance.'}
    dest = root / 'review'
    dest.mkdir(exist_ok=True)
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    shutil.copyfile(TEMPLATE, dest / 'index.html')
    print('REVIEW_READY', dest, sum(v['status'] == 'complete' for v in variants), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'render', 'audit', 'review'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--algorithm')
    parser.add_argument('--track')
    parser.add_argument('--mode', choices=('dark', 'schematic'), default='dark')
    parser.add_argument('--codec', choices=('av1_nvenc', 'libx264'), default='av1_nvenc')
    parser.add_argument('--guarded', action='store_true', help='Verify the live outer campaign GPU guard; do not relock')
    args = parser.parse_args()
    base.validate_mode(args.mode, args.codec)
    root = args.root.resolve()
    plan = read(root / 'PLAN.json')
    base.verify_files(plan)
    resource_guard(root, plan)
    if args.stage == 'review':
        review(root, plan, args.mode)
        return
    for algorithm, track in base.pairs(plan, args.algorithm, args.track):
        base.verify_files(plan)
        if args.stage == 'smoke':
            smoke_pair(root, plan, algorithm, track, args.mode, args.guarded)
        elif args.stage == 'render':
            render_pair(root, plan, algorithm, track, args.mode, args.codec, args.guarded)
        else:
            audit_pair(root, plan, algorithm, track, args.mode)
        resource_guard(root, plan)


if __name__ == '__main__':
    main()
