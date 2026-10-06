"""Unfiltered fixed-C / D4-max / D6 diagnostic traces, with explicit artifact roles.

Saved native body samples are differentiated before any rendering transform.
Pixel envelopes retain every raw sample, including fragment-boundary peaks.
The D6 row uses recorded playback_rate into D4, never source-clock derivatives.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from scripts.rendering import render_native_dynamics as base
from scripts.rendering.native_repair_traces import body_traces, motor_speed, pixel_envelope, sha
from scripts.rendering.native_unlimited_emission import apply as apply_unlimited, path as emission_path

METHODS = ('D7_reference', 'C3', 'C4')
ROLES = {'D7_reference': 'fixed_repaired_native_C', 'C3': 'fixed_C_D4_max_only',
         'C4': 'fixed_C_D4_max_external_D6_positive_pacing'}
ROWS = (('left_arm', 'Raw left arm (m/s)'), ('right_arm', 'Raw right arm (m/s)'),
        ('lower', 'Raw lower body (m/s)'), ('body_jerk', 'World RMS jerk (m/s³)'),
        ('root_y', 'Raw root Y (m)'), ('requested', 'Requested motor (°/s)'),
        ('emitted', 'Emitted motor (°/s)'), ('brightness', 'Actual beam dimmer'),
        ('rate', 'Recorded D6 rate (×)'))
WIDTH, HEIGHT = 1280, 304
START, END = 90., 150.
REFERENCE, CANDIDATE, DIFFERENCE = '#e8aa65', '#4bd6e6', '#bcc8dd'
SOURCE_COLORS = ('#f9b35a', '#a7df6c', '#71a9ff', '#de9edf')


def validate_plan(root, song, plan_sha256):
    root = Path(root)
    if (not isinstance(plan_sha256, str) or len(plan_sha256) != 64
            or any(c not in '0123456789abcdef' for c in plan_sha256)
            or sha(root / 'PLAN.json') != plan_sha256):
        raise ValueError('Externally supplied frozen PLAN digest differs')
    plan = base.read(root / 'PLAN.json')
    required = {'output_folder': 'preview_3d_v1', 'interval_s': [90, 150],
                'render_size': [1280, 1024], 'body_fps': 30,
                'rendered_methods': ['C4'], 'fixture_motion_limits': False}
    review = plan.get('preview_3d', {})
    if any(review.get(key) != value for key, value in required.items()):
        raise ValueError('PLAN.preview_3d must bind the C4-only raw30Hz90–150s diagnostic')
    if plan.get('transfer', {}).get('apply_motion_limits') is not False:
        raise ValueError('Simulation fixture motion limits must remain OFF')
    tracks = plan.get('tracks', [])
    if {t['id'] for t in tracks} != {'shpongle', 'cure_for_me'} or len(tracks) != 2:
        raise ValueError('Both independent source songs must be present')
    if set(review.get('fixed_native_sources', {})) != {'shpongle', 'cure_for_me'}:
        raise ValueError('Both repaired native source bodies must be frozen explicitly')
    track = next(t for t in tracks if t['id'] == song)
    if track['duration_s'] < END:
        raise ValueError('Track does not cover the complete90–150s diagnostic')
    base.track_clock(track)
    base.verify_source(plan, track)
    source_paths = [Path(review['fixed_native_sources'][s]['body']).resolve()
                    for s in ('shpongle', 'cure_for_me')]
    if source_paths[0] == source_paths[1]:
        raise ValueError('Each song requires its own repaired native body')
    return plan, track


def fixed_source(plan, song):
    """Authenticate the actual repaired source, including its metadata seal."""
    item = plan['preview_3d']['fixed_native_sources'][song]
    path = Path(item['body'])
    metadata = path.with_suffix('.json')
    if not path.is_absolute() or sha(path) != item['body_sha256'] or sha(metadata) != item['metadata_sha256']:
        raise ValueError('Frozen repaired-C source body/metadata differs')
    body = DanceResult.load(path)
    if (body.source_time is not None or body.metadata.get('retiming') is True
            or body.metadata.get('motif_recomposition') is True):
        raise ValueError('Fixed repaired C must be the unfiltered native source')
    declared_song = body.metadata.get('song')
    if declared_song is not None and declared_song != song:
        raise ValueError('Repaired source provenance belongs to another song')
    pins = {str(path): sha(path), str(metadata): sha(metadata),
            str(metadata.with_name(metadata.name + '.sha256.json')):
                sha(metadata.with_name(metadata.name + '.sha256.json'))}
    return body, pins


def saved_pacing(method, arrays, time):
    """D4-only rates are fragment fits; they are never labeled external D6."""
    time = np.asarray(time, float)
    cuts = np.asarray(arrays.get('intervention_time', np.empty(0)), float)
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(np.diff(cuts) < 0):
        raise ValueError('Invalid recorded motif-cut timestamps')
    if method != 'C4':
        return {'rate_channels': np.ones((len(time), 1)), 'source_time': None,
                'motif_cuts': cuts if method == 'C3' else np.empty(0),
                'clock_scope': ('native generated body; no D6' if method == 'D7_reference'
                                else 'D4-max-only composition; no external D6; unit D6 reference')}
    if 'playback_rate' not in arrays:
        raise ValueError('C4 requires actual saved playback_rate into D4')
    rates = np.asarray(arrays['playback_rate'], float)
    rates = rates[:, None] if rates.ndim == 1 else rates
    if (rates.ndim != 2 or len(rates) != len(time) or rates.shape[1] < 1
            or not np.isfinite(rates).all() or rates.min() < -1e-8):
        raise ValueError('Invalid recorded external D6 playback rates')
    if 'source_time' not in arrays or 'intervention_time' not in arrays:
        raise ValueError('C4 requires saved D4 source clock and motif-cut timestamps')
    clock = np.asarray(arrays['source_time'], float)
    clock = clock[:, None] if clock.ndim == 1 else clock
    if clock.ndim != 2 or len(clock) != len(time) or not np.isfinite(clock).all():
        raise ValueError('Invalid saved source clock into D4')
    return {'rate_channels': rates.copy(), 'source_time': clock.copy(),
            'motif_cuts': cuts.copy(),
            'clock_scope': 'recorded external D6 playback_rate into D4-max; not original-library clock derivative'}


def requested_angles(targets, patch, moving):
    vector = np.asarray(targets)[:, moving] - np.asarray([p['position'] for p in patch])[moving]
    length = np.linalg.norm(vector, axis=-1)
    if not np.isfinite(vector).all() or np.any(length <= 1e-12):
        raise ValueError('Invalid requested fixture target')
    return np.stack((np.rad2deg(np.arctan2(vector[..., 0], -vector[..., 1])),
                     np.rad2deg(np.arcsin(np.clip(vector[..., 2] / length, -1, 1)))), -1)


def pair_scale(reference, candidate, key):
    """Scope all scales to this role pair and this window; keep signed root Y."""
    selected = []
    for time, values in (reference, candidate):
        raw = np.asarray(values)[(time >= START) & (time <= END)]
        if not raw.size or not np.isfinite(raw).all():
            raise ValueError('Finite raw samples must cover every graph row')
        selected.append(raw.ravel())
    raw = np.concatenate(selected)
    low, high = float(raw.min()), float(raw.max())
    if key == 'root_y':
        margin = max((high-low)*.04, .005)
        return [low-margin, high+margin]
    if key == 'brightness':
        if low < -1e-8 or high > 1+1e-8:
            raise ValueError('Emitted dimmer lies outside0–1')
        return [0., 1.]
    return [min(0., low), max(high, 1e-8)]


def raw_difference(reference, candidate):
    rt, rv = reference; ct, cv = candidate
    if not np.array_equal(rt, ct) or np.asarray(rv).shape != np.asarray(cv).shape:
        raise ValueError('Signed difference requires matching native timestamps; no resampling')
    return ct.copy(), np.asarray(cv)-np.asarray(rv)


def song_traces(root, song, plan_sha256):
    root = Path(root)
    plan, track = validate_plan(root, song, plan_sha256)
    fixed, fixed_pins = fixed_source(plan, song)
    music_path = Path(track['music_timeline'])
    music = MusicTimeline.load(music_path)
    if music.metadata.get('input_sha256') != base.verify_source(plan, track):
        raise ValueError('Estimated source traces belong to another original song')
    pins = {str(root / 'PLAN.json'): plan_sha256, str(Path(track['audio'])): sha(track['audio']),
            **fixed_pins}
    for item in (music_path, music_path.with_suffix('.json'),
                 music_path.with_suffix('.json.sha256.json')):
        pins[str(item)] = sha(item)
    values, artifacts, common_time, common_patch = {}, {}, None, None
    for method in METHODS:
        pair = apply_unlimited(root, method, song, base.load_pair(root, plan, method, track))
        if common_time is not None and (not np.array_equal(pair.times, common_time) or pair.patch != common_patch):
            raise ValueError('Roles must share song timestamps and fixture patch')
        common_time, common_patch = pair.times, pair.patch
        body_path = root / 'bodies' / method / song / 'body.npz'
        body = DanceResult.load(body_path)
        if method == 'D7_reference' and (not np.array_equal(body.world, fixed.world)
                or not np.array_equal(body.time, fixed.time) or sha(body_path) != fixed_pins[plan['preview_3d']['fixed_native_sources'][song]['body']]):
            raise ValueError('D7_reference must be this song actual fixed repaired-C body')
        if body.metadata.get('song', song) != song:
            raise ValueError('Own-song role body required')
        rows = {k: v for k, v in body_traces(body.world, body.time).items()
                if k in dict(ROWS)}
        moving = np.asarray([p['kind'] != 'bulb' for p in pair.patch])
        beams = np.asarray([p['kind'] == 'beam' for p in pair.patch])
        timeline = root / 'timelines' / method / song / 'timeline.npz'
        with np.load(timeline, allow_pickle=False) as loaded:
            arrays = {key: loaded[key].copy() for key in
                      ('targets', 'playback_rate', 'source_time', 'intervention_time') if key in loaded}
        rows['requested'] = motor_speed(requested_angles(arrays['targets'], pair.patch, moving), pair.times)
        rows['emitted'] = motor_speed(pair.controls['angles_deg'][:, moving], pair.times)
        rows['brightness'] = pair.times.copy(), pair.controls['dimmer'][:, beams].mean(1)
        pacing = saved_pacing(method, arrays, pair.times)
        rows['rate'] = pair.times.copy(), pacing['rate_channels'].mean(1)
        if any(not np.isfinite(raw).all() for _, raw in rows.values()):
            raise ValueError('Nonfinite native raw or emitted trace')
        values[method] = {'rows': rows, 'pacing': pacing, 'metadata': body.metadata,
                          'time': pair.times.copy()}
        emission = emission_path(root, method, song)
        items = (body_path, body_path.with_suffix('.json'),
                 body_path.with_suffix('.json.sha256.json'), timeline,
                 timeline.parent / 'report.json', timeline.parent / 'patch.json',
                 emission, emission.with_suffix('.json'))
        artifacts[method] = {str(item): sha(item) for item in items}
        pins.update(artifacts[method])
    scales, difference_scales = {}, {}
    for method in METHODS:
        reference, current = values['D7_reference'], values[method]
        scales[method], difference_scales[method] = {}, {}
        for key, _ in ROWS:
            ref, cand = reference['rows'][key], current['rows'][key]
            if key == 'rate':
                ref = (reference['time'], reference['pacing']['rate_channels'])
                cand = (current['time'], current['pacing']['rate_channels'])
            scales[method][key] = pair_scale(ref, cand, key)
            time, delta = raw_difference(reference['rows'][key], current['rows'][key])
            selected = (time >= START) & (time <= END)
            extent = max(float(np.abs(delta[selected]).max()), 1e-8)
            difference_scales[method][key] = [-extent, extent]
        maximum = max(scales[method]['requested'][1], scales[method]['emitted'][1])
        scales[method]['requested'] = scales[method]['emitted'] = [0., maximum]
    p95 = np.maximum(np.quantile(music.sources[:, :, 0], .95, axis=0), 1e-9)
    sources = music.sources[:, :, 0] / p95
    selected = (music.time >= START) & (music.time <= END)
    return {'values': values, 'scales_by_method': scales, 'difference_scales_by_method': difference_scales,
            'source_time': music.time, 'sources': sources, 'source_p95': p95,
            'source_scale': max(1., float(sources[selected].max())), 'sections': music.sections,
            'inputs': pins, 'artifacts': artifacts, 'roles': dict(ROLES),
            'interval_s': [START, END], 'song': song}


class Strip:
    def __init__(self, root, method, song, plan_sha256):
        if method not in METHODS:
            raise ValueError('Unknown diagnostic role')
        self.data = song_traces(root, song, plan_sha256)
        self.method, self.left, self.right = method, 278, 1268
        self.start, self.end = START, END
        self.scales = self.data['scales_by_method'][method]
        self.difference_scales = self.data['difference_scales_by_method'][method]
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
        self.image = Image.new('RGB', (WIDTH, HEIGHT), (12, 18, 27))
        draw = ImageDraw.Draw(self.image)
        draw.text((10, 2), 'RAW | orange: fixed repaired C; cyan: this role | lower gray: signed role − C | pixel min/max, no smoothing',
                  font=self.font, fill='#d0dcea')
        draw.text((10, 20), 'Estimated source RMS / p95', font=self.font, fill='#bccadc')
        for column, (label, color) in enumerate(zip(('drums', 'bass', 'other', 'vocals'), SOURCE_COLORS)):
            draw.text((550+column*102, 19), label, font=self.font, fill=color)
            self.series(draw, self.data['source_time'], self.data['sources'][:, column],
                        31, 11, [0., self.data['source_scale']], color)
        for index, (key, label) in enumerate(ROWS):
            y = 45 + index*25
            scale, diff_scale = self.scales[key], self.difference_scales[key]
            draw.text((10, y), label, font=self.font, fill='#c2cede')
            draw.text((190, y), f'{scale[0]:.2g}..{scale[1]:.2g}', font=self.font, fill='#a5b2c5')
            draw.text((190, y+11), f'Δ ±{diff_scale[1]:.2g}', font=self.font, fill='#929fb4')
            for role, color in (('D7_reference', REFERENCE), (method, CANDIDATE)):
                time, raw = self.data['values'][role]['rows'][key]
                if key == 'rate':
                    for channel in self.data['values'][role]['pacing']['rate_channels'].T:
                        self.series(draw, time, channel, y, 12, scale, color)
                else:
                    self.series(draw, time, raw, y, 12, scale, color)
            draw.line((self.left, y+19, self.right, y+19), fill='#46526a')
            time, delta = raw_difference(self.data['values']['D7_reference']['rows'][key],
                                         self.data['values'][method]['rows'][key])
            self.series(draw, time, delta, y+15, 8, diff_scale, DIFFERENCE)
        for section in self.data['sections']:
            stamp = float(section['start'])
            if self.start <= stamp <= self.end:
                draw.line((self.x(stamp), 31, self.x(stamp), 42), fill='#dee5f0')
        for stamp in self.data['values'][method]['pacing']['motif_cuts']:
            if self.start <= stamp <= self.end:
                draw.line((self.x(stamp), 271, self.x(stamp), 275), fill='#de9edf')
        for stamp in np.linspace(self.start, self.end, 7):
            draw.text((min(self.x(stamp)-12, self.right-31), 276), f'{stamp:.0f}',
                      font=self.font, fill='#b7c5d7')
        draw.text((10, 291), '90–150 s | scales: this role + fixed C, same window; every raw peak retained | D6 channels recorded into D4 | no native-learning claim',
                  font=self.font, fill='#b7c5d7')

    def x(self, stamp):
        return round(self.left+(float(stamp)-self.start)/(self.end-self.start)*(self.right-self.left))

    def series(self, draw, time, raw, y, height, scale, color):
        low, high = pixel_envelope(time, raw, left=self.start, right=self.end,
                                  pixels=self.right-self.left+1)
        minimum, maximum = scale
        for index in np.flatnonzero(np.isfinite(low)):
            a = y+height-(low[index]-minimum)/(maximum-minimum)*(height-1)
            b = y+height-(high[index]-minimum)/(maximum-minimum)*(height-1)
            draw.line((self.left+int(index), a, self.left+int(index), b), fill=color)

    def pacing_label(self, stamp):
        if self.method == 'D7_reference':
            return 'Fixed repaired native C | own-song saved native30Hz; no posthoc D4 or D6'
        if self.method == 'C3':
            return 'POSTHOC D4-max recomposition of fixed C | external D6 OFF | diagnostic, no native learning claim'
        row = self.data['values'][self.method]
        rates = row['pacing']['rate_channels']
        actual = [float(np.interp(stamp, row['time'], rates[:, column])) for column in range(rates.shape[1])]
        text = 'POSTHOC recomposed / retimed | recorded D6 ' + '/'.join(f'{v:.2f}×' for v in actual)
        source = row['pacing']['source_time']
        lookup = float(np.interp(stamp, row['time'], source[:, 0]))
        text += f' | D4 clock {lookup:.2f}s'
        for fragment in row['metadata'].get('fragment_provenance', []):
            bounds = fragment.get('target_seconds', [])
            if len(bounds) == 2 and bounds[0] <= lookup < bounds[1]:
                text += f' | motif {fragment.get("motif_family", fragment.get("source_fragment", "?"))}'
                break
        return text + ' | no native-learning claim'

    def frame(self, stamp):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        draw.line((self.x(stamp), 31, self.x(stamp), 270), fill='#fff2bb', width=2)
        draw.text((1130, 2), f'Song {stamp:.2f}s', font=self.font, fill='#fff2bb')
        return image

    def evidence(self):
        return {'inputs': self.data['inputs'], 'roles': self.data['roles'],
                'role_artifact_sha256': self.data['artifacts'], 'scales': self.scales,
                'signed_difference_scales': self.difference_scales,
                'interval_s': [self.start, self.end],
                'scale_scope': 'Displayed fixed-C/current-role pair only,90–150s; all-three raw arrays exported separately.',
                'body_speed': 'Unfiltered own saved native30Hz forward differences. Left/right/lower joint speeds relative to pelvis XYZ; no40Hz interpolation.',
                'world_RMS_jerk': 'Three forward divided differences of saved world22-joint coordinates; interval-center timestamps; RMS vector magnitudes,m/s³.',
                'root_y': 'Signed saved world root height in metres; no rootXZ presentation transform applied to measurements.',
                'motor_speed': 'Raw forward pan/tilt velocity norms, mean over13movers; requested targets and actual unlimited decoded emitted DMX at40Hz.',
                'dimmer': 'Actual emitted beam dimmer, mean over8moving beams. Viewer fill is separate.',
                'pacing': {method: {'scope': row['pacing']['clock_scope'],
                                    'channels': row['pacing']['rate_channels'].shape[1],
                                    'motif_cuts': len(row['pacing']['motif_cuts'])}
                           for method, row in self.data['values'].items()},
                'estimated_sources': {'source_scale': self.data['source_scale'],
                                      'full_song_p95': self.data['source_p95'].tolist(),
                                      'scope': 'Own-song estimated stem RMS, no musical-correctness claim.'},
                'graph': 'No temporal filtering, clipping or resampling. Per-pixel minima/maxima retain every native sample; gray minirow is signed candidate-minus-fixed-C on matching native clocks. Rate channels are all drawn; rate signed difference compares their arithmetic mean.',
                'claim': 'HUMAN CHECK / DIAGNOSTIC; posthoc recomposition/retiming, no native learning, latent repair, product promotion or musical verdict.'}

    def export(self, path):
        """Export all roles at their own clocks plus display pair envelopes and JSON."""
        path = Path(path)
        values = {'graph_interval_s': np.asarray([self.start, self.end])}
        source_mask = (self.data['source_time'] >= self.start) & (self.data['source_time'] <= self.end)
        values['estimated_source_time'] = self.data['source_time'][source_mask]
        values['estimated_source_rms_over_p95'] = self.data['sources'][source_mask]
        for method, row in self.data['values'].items():
            for key, (time, raw) in row['rows'].items():
                selected = (time >= self.start) & (time <= self.end)
                values[f'{method}_{key}_time'] = time[selected]
                values[f'{method}_{key}_raw'] = raw[selected]
                low, high = pixel_envelope(time, raw, left=self.start, right=self.end,
                                          pixels=self.right-self.left+1)
                valid = np.isfinite(low)
                values[f'{method}_{key}_pixel_valid'] = valid
                values[f'{method}_{key}_pixel_min'] = np.where(valid, low, 0.)
                values[f'{method}_{key}_pixel_max'] = np.where(valid, high, 0.)
            selected = (row['time'] >= self.start) & (row['time'] <= self.end)
            clock_kind = 'recorded_D6_rate' if method == 'C4' else 'no_D6_unit_clock'
            values[f'{method}_{clock_kind}_channels'] = row['pacing']['rate_channels'][selected]
            values[f'{method}_{clock_kind}_time'] = row['time'][selected]
            values[f'{method}_motif_cuts'] = row['pacing']['motif_cuts']
            if row['pacing']['source_time'] is not None:
                values[f'{method}_saved_D4_source_clock'] = row['pacing']['source_time'][selected]
        for key, _ in ROWS:
            time, delta = raw_difference(self.data['values']['D7_reference']['rows'][key],
                                         self.data['values'][self.method]['rows'][key])
            selected = (time >= self.start) & (time <= self.end)
            values[f'pair_delta_{key}_time'] = time[selected]
            values[f'pair_delta_{key}_raw'] = delta[selected]
        with path.open('xb') as stream:
            np.savez_compressed(stream, **values)
        base.save(path.with_suffix('.json'), {'npz_sha256': sha(path), **self.evidence()})
        return {str(path): sha(path), str(path.with_suffix('.json')): sha(path.with_suffix('.json'))}
