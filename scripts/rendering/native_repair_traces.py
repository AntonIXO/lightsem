"""Raw tremor/activity traces for the fixed old-D7 / broken-C / repaired-C review.

No filtering, retiming or corrective body processing. Graph pixel min/max
segments preserve every raw peak; native-time numerical arrays remain exported.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.rendering import render_native_dynamics as base
from scripts.rendering.native_unlimited_emission import apply as apply_unlimited, path as emission_path
from lightsem.experiments.dynamic_dance.contracts import DanceResult

METHODS = ('D7_reference', 'C3', 'C4')
ROLES = {'D7_reference': 'old_D7', 'C3': 'broken_C', 'C4': 'repaired_C'}
ROWS = (('left_arm', 'Raw left arm (m/s)'), ('right_arm', 'Raw right arm (m/s)'),
        ('lower', 'Raw lower body (m/s)'), ('body_jerk', 'Joint RMS jerk (m/s³)'),
        ('root_y', 'Raw root Y (m)'), ('root_vertical_jerk', 'Root Y jerk (m/s³)'),
        ('requested', 'Requested motors (°/s)'), ('emitted', 'Emitted motors (°/s)'),
        ('brightness', 'Beam dimmer (0–1)'))
WIDTH, HEIGHT = 1280, 304
REFERENCE, CANDIDATE = '#939eae', '#4bd6e6'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def derivative(values, time, order=1):
    """Forward divided differences, with interval-center timestamps; no padding."""
    values, time = np.asarray(values, float), np.asarray(time, float)
    if (len(values) != len(time) or len(time) <= order or not np.isfinite(time).all()
            or not np.isfinite(values).all() or np.any(np.diff(time) <= 0)):
        raise ValueError('Finite aligned values and increasing native timestamps required')
    for _ in range(order):
        values = np.diff(values, axis=0)/np.diff(time).reshape((-1,)+(1,)*(values.ndim-1))
        time = (time[:-1]+time[1:])/2
    return time, values


def body_traces(world, time):
    world, time = np.asarray(world, float), np.asarray(time, float)
    if world.shape != (len(time), 22, 3):
        raise ValueError('Expected saved 22-joint body')
    speed_time, velocity = derivative(world-world[:, :1], time)
    speed = np.linalg.norm(velocity, axis=-1)
    jerk_time, jerk = derivative(world, time, 3)
    return {'left_arm': (speed_time, speed[:, [16, 18, 20]].mean(1)),
            'right_arm': (speed_time, speed[:, [17, 19, 21]].mean(1)),
            'lower': (speed_time, speed[:, [1, 2, 4, 5, 7, 8, 10, 11]].mean(1)),
            'body_jerk': (jerk_time, np.sqrt(np.mean(np.sum(jerk**2, axis=-1), axis=-1))),
            'root_y': (time, world[:, 0, 1]),
            'root_vertical_jerk': (jerk_time, jerk[:, 0, 1])}


def motor_speed(angles, time):
    derivative_time, velocity = derivative(angles, time)
    return derivative_time, np.linalg.norm(velocity, axis=-1).mean(1)


def pixel_envelope(time, values, *, left, right, pixels=1015):
    """Bin every raw sample, preserving signed minima and maxima, not averages."""
    time, values = np.asarray(time), np.asarray(values)
    if (time.ndim != 1 or values.shape != time.shape or not np.isfinite(time).all()
            or not np.isfinite(values).all() or not isinstance(pixels, int) or pixels < 1 or right <= left):
        raise ValueError('Finite raw series and positive pixel range required')
    mask = (time >= left) & (time <= right)
    ids = np.minimum(((time[mask]-left)/(right-left)*pixels).astype(int), pixels-1)
    low, high = np.full(pixels, np.inf), np.full(pixels, -np.inf)
    np.minimum.at(low, ids, values[mask]); np.maximum.at(high, ids, values[mask])
    empty = ~np.isfinite(low)
    low[empty] = np.nan; high[empty] = np.nan
    return low, high


def waveform(track):
    """Complete original L/R-mean waveform extrema, including the fractional final bin."""
    import soundfile as sf
    duration = float(track['duration_s'])
    if not np.isfinite(duration) or duration <= 0: raise ValueError('Positive full-song duration required')
    with sf.SoundFile(track['audio']) as source:
        rate = source.samplerate
        samples = source.read(dtype='float32', always_2d=True)
    if abs(len(samples)-duration*rate) > 1.01 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Original waveform differs from frozen full-song duration')
    mono = samples.mean(axis=1)
    bin_time = np.minimum(np.arange(int(np.ceil(duration*40))+1)/40, duration)
    edges = np.rint(bin_time*rate).astype(int); edges[-1] = len(samples)
    if np.any(np.diff(edges) <= 0): raise ValueError('Waveform bins require positive source samples')
    low = np.array([mono[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    high = np.array([mono[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    return {'time': (bin_time[:-1]+bin_time[1:])/2, 'bin_edges': bin_time,
            'minimum': low, 'maximum': high, 'sample_rate': rate, 'channels': samples.shape[1],
            'source_samples': len(samples), 'source_duration_s': duration, 'bin_s': 1/40}


@lru_cache(maxsize=2)
def song_traces(root, song):
    root = Path(root); plan = base.read(root/'PLAN.json')
    track = next(t for t in plan['tracks'] if t['id'] == song)
    if base.verify_source(plan, track) != sha(track['audio']):
        raise ValueError('Original audio identity changed')
    duration = float(track['duration_s'])
    pins = {str(root/'PLAN.json'): sha(root/'PLAN.json'), track['audio']: sha(track['audio'])}
    values, common_time = {}, None
    for method in METHODS:
        pair = apply_unlimited(root, method, song, base.load_pair(root, plan, method, track))
        if common_time is not None and not np.array_equal(common_time, pair.times):
            raise ValueError('Roles must share song timestamps')
        common_time = pair.times
        body_path = root/'bodies'/method/song/'body.npz'; body = DanceResult.load(body_path)
        if body.source_time is not None or body.metadata.get('retiming') is True or body.metadata.get('motif_recomposition') is True:
            raise ValueError('Repair review requires native output without posthoc retiming/recomposition')
        rows = body_traces(body.world, body.time)
        moving = np.array([p['kind'] != 'bulb' for p in pair.patch]); beams = np.array([p['kind'] == 'beam' for p in pair.patch])
        timeline = root/'timelines'/method/song/'timeline.npz'
        with np.load(timeline, allow_pickle=False) as data:
            targets = data['targets'][:, moving].copy()
        vector = targets-np.array([p['position'] for p in pair.patch])[moving]
        length = np.linalg.norm(vector, axis=-1)
        if np.any(length <= 1e-12): raise ValueError('Requested target coincides with fixture')
        requested = np.stack((np.rad2deg(np.arctan2(vector[..., 0], -vector[..., 1])),
            np.rad2deg(np.arcsin(np.clip(vector[..., 2]/length, -1, 1)))), -1)
        rows['requested'] = motor_speed(requested, pair.times)
        rows['emitted'] = motor_speed(pair.controls['angles_deg'][:, moving], pair.times)
        rows['brightness'] = pair.times, pair.controls['dimmer'][:, beams].mean(1)
        values[method] = rows
        emission = emission_path(root, method, song)
        for item in (body_path, body_path.with_suffix('.json'), timeline, timeline.parent/'report.json', timeline.parent/'patch.json', emission, emission.with_suffix('.json')):
            pins[str(item)] = sha(item)
    scales = {}
    for name, _ in ROWS:
        selected = np.concatenate([v[(t >= 0) & (t <= duration)] for t, v in (values[m][name] for m in METHODS)])
        if not selected.size or not np.isfinite(selected).all(): raise ValueError('Invalid raw review trace')
        low, high = float(selected.min()), float(selected.max())
        if name == 'root_vertical_jerk': low, high = -max(abs(low), abs(high), 1e-6), max(abs(low), abs(high), 1e-6)
        elif name == 'root_y':
            margin = max((high-low)*.05, .005); low -= margin; high += margin
        elif name == 'brightness': low, high = 0., 1.
        else: low, high = 0., max(high, 1e-8)
        scales[name] = [low, high]
    motor_max = max(scales['requested'][1], scales['emitted'][1]); scales['requested'] = scales['emitted'] = [0., motor_max]
    return {'values': values, 'scales': scales, 'waveform': waveform(track), 'inputs': pins,
            'roles': dict(ROLES), 'interval_s': [0., duration]}


class Strip:
    def __init__(self, root, method, song, interval=None):
        self.data, self.method, self.left, self.right = song_traces(str(root), song), method, 254, 1268
        self.start, self.end = self.data['interval_s'] if interval is None else map(float, interval)
        if not 0 <= self.start < self.end <= self.data['interval_s'][1]:
            raise ValueError('Graph interval must be inside the complete source song')
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
        self.image = Image.new('RGB', (WIDTH, HEIGHT), (12, 18, 27)); draw = ImageDraw.Draw(self.image)
        draw.text((10, 2), 'RAW DATA | gray: old D7; cyan: current role | per-pixel min/max preserves raw peaks | no smoothing', font=self.font, fill='#cdd8e8')
        draw.text((10, 24), 'Original mono waveform', font=self.font, fill='#a6b9cf')
        wave = self.data['waveform']; maximum = max(float(np.abs(wave['minimum']).max()), float(np.abs(wave['maximum']).max()), 1e-8)
        lo, _ = pixel_envelope(wave['time'], wave['minimum'], left=self.start, right=self.end, pixels=self.right-self.left+1)
        _, hi = pixel_envelope(wave['time'], wave['maximum'], left=self.start, right=self.end, pixels=self.right-self.left+1)
        self.envelope(draw, lo, hi, 25, (-maximum, maximum), '#b4c4dc')
        for index, (name, label) in enumerate(ROWS):
            y = 55+index*24; scale = self.data['scales'][name]
            draw.text((10, y+1), label, font=self.font, fill='#bdc9d9')
            compact = lambda value: f'{value:.2g}'.replace('e+0', 'e').replace('e+', 'e')
            draw.text((207, y), compact(scale[1]), font=self.font, fill='#93a2b6')
            draw.text((207, y+10), compact(scale[0]), font=self.font, fill='#93a2b6')
            draw.line((self.left, y+19, self.right, y+19), fill='#354152')
            for role, color in (('D7_reference', REFERENCE), (method, CANDIDATE)):
                time, raw = self.data['values'][role][name]
                low, high = pixel_envelope(time, raw, left=self.start, right=self.end, pixels=self.right-self.left+1)
                self.envelope(draw, low, high, y, scale, color)
        for stamp in np.linspace(self.start, self.end, 7):
            draw.text((min(self.x(stamp)-12, self.right-35), 273), f'{stamp:.1f}', font=self.font, fill='#b7c5d7')
        draw.text((10, 288), 'Song seconds | root Y: metres, not speed | root jerk is signed | joint RMS jerk is world-space | shared per-song scales', font=self.font, fill='#b7c5d7')

    def x(self, stamp): return round(self.left+(float(stamp)-self.start)/(self.end-self.start)*(self.right-self.left))

    def envelope(self, draw, low, high, y, scale, color):
        minimum, maximum = scale
        for index in np.flatnonzero(np.isfinite(low)):
            a = y+18-(low[index]-minimum)/(maximum-minimum)*17
            b = y+18-(high[index]-minimum)/(maximum-minimum)*17
            draw.line((self.left+int(index), a, self.left+int(index), b), fill=color, width=1)

    def motion_label(self, stamp):
        return {'D7_reference': 'OLD D7 | historical pre-native generator; saved native motion.',
                'C3': 'BROKEN NATIVE C | original joint seed0; saved native motion.',
                'C4': 'REPAIRED NATIVE C | new joint seed0; repair qualification is shown in the unified page evidence.'}[self.method]

    def frame(self, stamp):
        image = self.image.copy(); draw = ImageDraw.Draw(image); x = self.x(stamp)
        draw.line((x, 24, x, 269), fill='#fff2bb', width=2)
        draw.text((1110, 2), f'Song {stamp:.2f}s', font=self.font, fill='#fff2bb')
        return image

    def evidence(self):
        return {'inputs': self.data['inputs'], 'roles': self.data['roles'], 'scales': self.data['scales'],
            'body_speed': 'Unfiltered native30Hz forward differences; region velocity is relative to pelvis XYZ.',
            'body_jerk': 'Three forward divided differences at interval-center timestamps. World-space RMS over22 joint-vector jerk magnitudes, in m/s³.',
            'root_y': 'Original signed world-space root Y in metres; no absolute-value or speed conversion.',
            'root_vertical_jerk': 'Signed third forward derivative of original root Y, m/s³; no detrending or filtering.',
            'motor_speed': 'Raw forward pan/tilt velocity norms, mean over13moving fixtures; actual unlimited emittedDMX.',
            'graph': 'No temporal smoothing. Per-pixel raw minima and maxima are drawn so peak glitches remain visible.',
            'waveform': 'Complete original audio, arithmetic mono channel mean; raw min/max in25ms bins and final fractional bin; no audio modification.',
            'interval_s': [self.start, self.end], 'scale_scope': 'Complete song, shared across the three roles.',
            'scope': 'Measured behaviour only; high-frequency/support/activity acceptance requires independent evidence. No inferred artistic verdict.'}

    def export(self, path):
        wave = self.data['waveform']; values = {'waveform_bin_time': wave['time'], 'waveform_bin_edges': wave['bin_edges'],
            'waveform_minimum': wave['minimum'], 'waveform_maximum': wave['maximum'],
            'graph_interval_s': np.array([self.start, self.end])}
        for label, method in (('reference', 'D7_reference'), ('current', self.method)):
            for name, (time, raw) in self.data['values'][method].items():
                selected = (time >= self.start) & (time <= self.end)
                values[f'{label}_{name}_time'] = time[selected]
                values[f'{label}_{name}_raw'] = raw[selected]
                low, high = pixel_envelope(time, raw, left=self.start, right=self.end, pixels=self.right-self.left+1)
                # Empty bins, if any, are explicitly represented by a valid mask.
                valid = np.isfinite(low)
                values[f'{label}_{name}_pixel_valid'] = valid
                values[f'{label}_{name}_pixel_min'] = np.where(valid, low, 0.)
                values[f'{label}_{name}_pixel_max'] = np.where(valid, high, 0.)
        with Path(path).open('xb') as stream: np.savez_compressed(stream, **values)
