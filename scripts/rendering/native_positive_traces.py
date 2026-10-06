"""Measured hybrid-body, unlimited-emission and saved pacing traces.

Native C is the gray reference. Rate and motif markers come from saved artifacts;
fragment cuts are never differentiated to fabricate a playback-rate curve.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
from scipy.ndimage import uniform_filter1d
from PIL import Image, ImageDraw, ImageFont
from scripts.rendering import render_native_dynamics as base
from scripts.rendering.native_unlimited_emission import apply as apply_unlimited, path as emission_path
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from lightsem.experiments.dynamic_dance.diagnostics import dynamics_series

METHODS = ('D7_reference', 'C3', 'C4')
ROWS = (('left_arm', 'Left arm (m/s)'), ('right_arm', 'Right arm (m/s)'),
        ('lower', 'Lower body (m/s)'), ('torso', 'Torso / head (m/s)'),
        ('root', 'Root speed (m/s)'), ('requested', 'Requested motors (deg/s)'),
        ('emitted', 'Emitted motors (deg/s)'), ('brightness', 'Beam dimmer (0–1)'),
        ('rate', 'Saved D6 rate (×)'))
SOURCE_COLORS = ('#f9b35a', '#a7df6c', '#71a9ff', '#de9edf')
REFERENCE, CANDIDATE = '#91a0b1', '#4bd6e6'
WIDTH, HEIGHT = 1280, 304


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interval_speed(values, time):
    velocity = np.diff(values, axis=0) / np.diff(time).reshape((-1,) + (1,) * (values.ndim-1))
    speed = np.linalg.norm(velocity, axis=-1)
    if speed.ndim > 1:
        speed = speed.mean(1)
    return np.r_[speed, speed[-1]]


def body_speeds(world, time):
    series = dynamics_series(world, time)
    velocity = np.linalg.norm(series['relative_velocity_m_s'], axis=-1)
    return {'left_arm': velocity[:, [16, 18, 20]].mean(1), 'right_arm': velocity[:, [17, 19, 21]].mean(1),
            'lower': velocity[:, [1, 2, 4, 5, 7, 8, 10, 11]].mean(1),
            'torso': velocity[:, [3, 6, 9, 12, 15]].mean(1), 'root': series['root_speed_m_s']}


def saved_pacing(method, arrays, time):
    """Use recorded D6 rates without smoothing or deriving across motif cuts."""
    if method == 'D7_reference':
        return {'rate_channels': np.ones((len(time), 1)), 'source_time': None,
                'motif_cuts': np.empty(0), 'clock_scope': 'native generated output, no retiming'}
    if 'playback_rate' not in arrays:
        raise ValueError('Retimed hybrid requires actual saved playback_rate')
    rates = np.asarray(arrays['playback_rate'])
    rates = rates[:, None] if rates.ndim == 1 else rates
    if rates.ndim != 2 or len(rates) != len(time) or rates.shape[1] < 1 or not np.isfinite(rates).all() or rates.min() < -1e-8:
        raise ValueError('Invalid saved hybrid playback rates')
    source = arrays.get('source_time')
    if source is not None:
        source = np.asarray(source)
        source = source[:, None] if source.ndim == 1 else source
        if source.ndim != 2 or len(source) != len(time) or not np.isfinite(source).all():
            raise ValueError('Invalid saved source clock')
    if 'intervention_time' not in arrays:
        raise ValueError('Hybrid requires saved motif-cut timestamps')
    cuts = np.asarray(arrays['intervention_time'])
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(np.diff(cuts) < 0):
        raise ValueError('Invalid saved motif-cut timestamps')
    return {'rate_channels': rates.copy(), 'source_time': None if source is None else source.copy(),
            'motif_cuts': cuts.copy(), 'clock_scope': 'saved D6 clock into the D4-max composition'}


@lru_cache(maxsize=2)
def song_traces(root, song):
    root = Path(root)
    plan = base.read(root/'PLAN.json')
    track = next(track for track in plan['tracks'] if track['id'] == song)
    music = MusicTimeline.load(track['music_timeline'])
    pins = {str(root/'PLAN.json'): sha(root/'PLAN.json'), track['music_timeline']: sha(track['music_timeline'])}
    all_values, common_time = {}, None
    for method in METHODS:
        pair = apply_unlimited(root, method, song, base.load_pair(root, plan, method, track))
        emission = emission_path(root, method, song)
        pins[str(emission)] = sha(emission)
        pins[str(emission.with_suffix('.json'))] = sha(emission.with_suffix('.json'))
        body_path = root/'bodies'/method/song/'body.npz'
        body = DanceResult.load(body_path)
        time = pair.times
        if common_time is not None and not np.array_equal(common_time, time):
            raise ValueError('Comparisons must share a common song clock')
        common_time = time
        raw = {name: np.interp(time, body.time, value) for name, value in body_speeds(body.world, body.time).items()}
        moving = np.array([p['kind'] != 'bulb' for p in pair.patch])
        beams = np.array([p['kind'] == 'beam' for p in pair.patch])
        timeline_path = root/'timelines'/method/song/'timeline.npz'
        with np.load(timeline_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ('targets', 'playback_rate', 'source_time', 'intervention_time') if key in data}
        targets = arrays['targets'][:, moving]
        vector = targets - np.array([p['position'] for p in pair.patch])[moving]
        length = np.linalg.norm(vector, axis=-1)
        if np.any(length <= 1e-12):
            raise ValueError('Requested target coincides with fixture')
        requested = np.stack((np.rad2deg(np.arctan2(vector[..., 0], -vector[..., 1])),
            np.rad2deg(np.arcsin(np.clip(vector[..., 2]/length, -1, 1)))), -1)
        raw['requested'] = interval_speed(requested, time)
        raw['emitted'] = interval_speed(pair.controls['angles_deg'][:, moving], time)
        raw['brightness'] = pair.controls['dimmer'][:, beams].mean(1)
        pacing = saved_pacing(method, arrays, time)
        raw['rate'] = pacing['rate_channels'].mean(1)
        display = {name: (value.copy() if name in ('brightness', 'rate') else uniform_filter1d(value, size=10, mode='nearest'))
                   for name, value in raw.items()}
        all_values[method] = {'raw': raw, 'display': display, 'pacing': pacing, 'metadata': body.metadata}
        for item in (body_path, body_path.with_suffix('.json'), timeline_path, timeline_path.parent/'report.json', timeline_path.parent/'patch.json'):
            pins[str(item)] = sha(item)
    if any(not np.isfinite(value).all() for rows in all_values.values() for kind in ('raw', 'display') for value in rows[kind].values()):
        raise ValueError('Nonfinite measured hybrid trace')
    selected = (common_time >= 90) & (common_time <= 150)
    scales = {name: max(float(all_values[method]['display'][name][selected].max()) for method in METHODS) for name, _ in ROWS}
    scales['brightness'] = 1.
    scales['rate'] = max(1.5, max(float(all_values[method]['pacing']['rate_channels'][selected].max()) for method in METHODS))
    scales['requested'] = scales['emitted'] = max(scales['requested'], scales['emitted'])
    scales = {name: max(value, 1e-8) for name, value in scales.items()}
    p95 = np.maximum(np.quantile(music.sources[:, :, 0], .95, axis=0), 1e-9)
    sources = np.stack([np.interp(common_time, music.time, music.sources[:, j, 0]/p95[j]) for j in range(4)], -1)
    return {'time': common_time, 'selected': selected, 'values': all_values, 'scales': scales, 'sources': sources,
            'source_scale': max(1., float(sources[selected].max())), 'source_p95': p95,
            'sections': music.sections, 'inputs': pins}


class Strip:
    def __init__(self, root, method, song):
        self.data = song_traces(str(root), song)
        self.method, self.left, self.right = method, 254, 1268
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
        self.image = Image.new('RGB', (WIDTH, HEIGHT), (12, 18, 27))
        draw = ImageDraw.Draw(self.image)
        draw.text((10, 3), 'Gray: original native C | cyan: current output | speed: 250 ms display average; saved rates unsmoothed', font=self.font, fill='#cfdaea')
        draw.text((10, 22), 'Sources / full-song p95', font=self.font, fill='#bccadc')
        for j, (label, color) in enumerate(zip(('drums', 'bass', 'other', 'vocals'), SOURCE_COLORS)):
            draw.text((600+j*100, 20), label, font=self.font, fill=color)
            self.line(draw, self.data['sources'][:, j], 39, self.data['source_scale'], color)
        draw.text((10, 39), 'Estimated source RMS', font=self.font, fill='#91a0b1')
        reference = self.data['values']['D7_reference']['display']
        current = self.data['values'][method]['display']
        for index, (key, label) in enumerate(ROWS):
            y = 63+index*23
            scale = self.data['scales'][key]
            draw.text((10, y+1), label, font=self.font, fill='#bccadc')
            draw.text((216, y+1), f'{scale:.2g}', font=self.font, fill='#91a0b1')
            draw.line((self.left, y+19, self.right, y+19), fill='#354152')
            self.line(draw, reference[key], y, scale, REFERENCE)
            if key == 'rate':
                rates = self.data['values'][method]['pacing']['rate_channels']
                for column in range(rates.shape[1]):
                    self.line(draw, rates[:, column], y, scale, CANDIDATE if rates.shape[1] == 1 else SOURCE_COLORS[column % 4])
            else:
                self.line(draw, current[key], y, scale, CANDIDATE)
        for section in self.data['sections']:
            if 90 <= section['start'] <= 150:
                draw.line((self.x(section['start']), 39, self.x(section['start']), 269), fill='#546174')
        for stamp in self.data['values'][method]['pacing']['motif_cuts']:
            if 90 <= stamp <= 150:
                x = self.x(stamp)
                draw.polygon(((x-3, 267), (x+3, 267), (x, 273)), fill='#ffd06a')
        for stamp in range(90, 151, 10):
            draw.text((min(self.x(stamp)-12, self.right-30), 275), str(stamp), font=self.font, fill='#b7c5d7')
        draw.text((10, 290), 'Song seconds | yellow: saved motif cuts | shared scales | A: native 1×; B/C: actual D6 pacing; audio always 1×', font=self.font, fill='#b7c5d7')

    def x(self, stamp):
        return round(self.left+(float(stamp)-90)/60*(self.right-self.left))

    def line(self, draw, values, y, scale, color):
        xs = np.arange(self.left, self.right+1)
        stamps = 90+(xs-self.left)/(self.right-self.left)*60
        values = np.interp(stamps, self.data['time'], values)
        yy = y+18-np.clip(values/scale, 0, 1)*17
        draw.line(list(zip(xs.tolist(), yy.tolist())), fill=color, width=1)

    def pacing_label(self, stamp):
        if self.method == 'D7_reference':
            return 'Original native C joint-planner body; native clock 1.00×; no motif recomposition or retiming.'
        row = self.data['values'][self.method]
        rates = np.array([np.interp(stamp, self.data['time'], column) for column in row['pacing']['rate_channels'].T])
        rate = f'{rates[0]:.2f}×' if len(rates) == 1 else f'{rates.min():.2f}–{rates.max():.2f}×'
        text = f'Recomposed own-song motion | actual D6 rate {rate}'
        source = row['pacing']['source_time']
        source_clock = None
        if source is not None:
            source_clock = float(np.interp(stamp, self.data['time'], source[:, 0]))
            text += f' | D4 source clock {source_clock:.2f}s'
        fragments = row['metadata'].get('fragment_provenance', [])
        lookup = stamp if source_clock is None else source_clock
        for index, fragment in enumerate(fragments):
            bounds = fragment.get('target_seconds', [])
            if len(bounds) == 2 and (bounds[0] <= lookup < bounds[1] or index == len(fragments)-1 and lookup >= bounds[0]):
                text += f' | motif {fragment.get("motif_family", fragment.get("source_fragment", index))}'
                break
        return text

    def frame(self, stamp):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        x = self.x(stamp)
        draw.line((x, 39, x, 271), fill='#fff2bb', width=2)
        draw.text((1100, 3), f'Song {stamp:.2f} s', font=self.font, fill='#fff2bb')
        return image

    def evidence(self):
        return {'inputs': self.data['inputs'], 'scales': self.data['scales'],
            'source_scale': self.data['source_scale'], 'source_p95': self.data['source_p95'].tolist(),
            'reference': 'D7_reference alias means original native C joint planner in this pilot',
            'body_speed': 'Saved 30Hz output-body forward differences; hip-relative regions and separate root. Interpolate measured speeds to40Hz for graph.',
            'motor_speed': 'Mean13moving-fixture pan/tilt velocity norm; requested targets versus actual unlimited decodedDMX.',
            'smoothing': 'Ten40Hzsamples(250ms) average only for displayed speed curves. Rates and dimmer unsmoothed; no body/DMX changes.',
            'pacing': {method: {'scope': row['pacing']['clock_scope'], 'rate_columns': row['pacing']['rate_channels'].shape[1],
                               'motif_cut_count': len(row['pacing']['motif_cuts'])} for method, row in self.data['values'].items()},
            'meaning': 'Measured behaviour and saved source clocks, not musical-correctness errors or an improvement verdict.'}

    def export(self, path):
        selected = self.data['selected']
        values = {'time': self.data['time'][selected], 'source_rms_over_p95': self.data['sources'][selected].astype('f4')}
        for label, method in (('reference', 'D7_reference'), ('current', self.method)):
            row = self.data['values'][method]
            for kind in ('raw', 'display'):
                for key, value in row[kind].items():
                    values[f'{label}_{kind}_{key}'] = value[selected].astype('f4')
            rate_key = '_native_clock_rate' if method == 'D7_reference' else '_saved_playback_rate'
            values[label+rate_key] = row['pacing']['rate_channels'][selected]
            values[label+'_motif_cut_time'] = row['pacing']['motif_cuts']
            if row['pacing']['source_time'] is not None:
                values[label+'_saved_source_time'] = row['pacing']['source_time'][selected]
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **values)
