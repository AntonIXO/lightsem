"""Bounded CPU diagnostics of body intent and the actual emitted DMX.

These descriptive instruments measure activity, information loss and geometric
clutter. They do not assign artistic quality or infer palette events from RGB.
"""
from __future__ import annotations

import hashlib
import numpy as np

from lightsem.conductor.body_transfer import decode
from scripts.rendering.render_body_transfer import project, ray_hits


def _sample(n, maximum):
    return np.unique(np.linspace(0, n - 1, min(n, maximum)).round().astype(int))


def _stats(x):
    x = np.asarray(x)
    if not x.size:
        return {'min': None, 'median': None, 'p95': None, 'max': None}
    return dict(zip(('min', 'median', 'p95', 'max'),
                    map(float, np.quantile(x, [0, .5, .95, 1]))))


def _ratio(a, b):
    return float(a / b) if b > 1e-15 else None


def _correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if not a.size:
        return None
    a, b = a - a.mean(), b - b.mean()
    return _ratio(np.dot(a, b), np.linalg.norm(a) * np.linalg.norm(b))


def _longest(mask, fps):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    spans = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(spans.max() / fps) if spans.size else 0.


def _duplicates(x, names):
    groups = {}
    for i, name in enumerate(names):
        # Canonicalize signed floating zero before exact numeric hashing.
        values = np.ascontiguousarray(x[:, i] + 0)
        key = hashlib.sha256(values.tobytes()).hexdigest()
        groups.setdefault(key, []).append(name)
    return {'unique_streams': len(groups), 'total_streams': len(names),
            'duplicate_groups': [g for g in groups.values() if len(g) > 1],
            'scope': 'Exact full-window streams; different origins can yield different angles for identical targets.'}


def _pair_indices(n, maximum=4096):
    total = n * (n - 1) // 2
    if not total:
        return np.empty(0, int), np.empty(0, int), total
    k = _sample(total, maximum)
    rows = np.arange(n, dtype=np.int64)
    starts = rows * (2 * n - rows - 1) // 2
    i = np.searchsorted(starts, k, side='right') - 1
    j = i + 1 + k - starts[i]
    return i, j, total


def _dimension(x):
    if x is None:
        return None
    x = np.asarray(x)
    flat = x[_sample(len(x), 240)].reshape(min(len(x), 240), -1)
    if not flat.shape[1]:
        return None
    selected = _sample(flat.shape[1], 192)
    flat = flat[:, selected]
    centered = flat - flat.mean(0)
    eig = np.linalg.svd(centered, compute_uv=False) ** 2
    energy = float(eig.sum())
    return {'participation_ratio': _ratio(energy ** 2, np.dot(eig, eig)) if energy > 1e-20 else 0.,
            'constant': bool(np.all(flat == flat[:1])),
            'mean_coordinate_std': float(flat.std(0).mean()),
            'sampled_frames': len(flat), 'sampled_coordinates': len(selected),
            'total_coordinates': int(np.prod(x.shape[1:])),
            'scope': 'Coordinate-dependent descriptive dimension, not representation-collapse evidence.'}


def _distance_vector(x, samples):
    if x is None or not np.asarray(x).size:
        return None
    flat = np.asarray(x)[samples].reshape(len(samples), -1)
    flat = flat[:, _sample(flat.shape[1], 192)]
    flat -= flat.mean(0)
    sq = np.sum(flat * flat, axis=1)
    distances = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * flat @ flat.T, 0))
    return distances[np.triu_indices(len(samples), 1)]


def _recurrence(body, field, desired, emitted):
    samples = _sample(len(body), 240)
    values = [_distance_vector(x, samples) for x in (body, field, desired, emitted)]
    names = ('body', 'field', 'intent', 'emitted')
    out = {'sampled_frames': len(samples), 'maximum_coordinates': 192,
           'scope': 'Pearson agreement of temporal self-distance matrices; not annotated motifs or phrase quality.'}
    for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3)):
        out[names[a] + '_vs_' + names[b]] = (
            _correlation(values[a], values[b]) if values[a] is not None and values[b] is not None else None)
    return out


def _angular_bands(desired, actual, fps):
    bands = ((0., .5), (.5, 1.), (1., 2.), (2., 4.), (4., 8.))
    if len(desired) < 4 or not desired.size:
        return {f'{lo:g}-{hi:g}Hz': None for lo, hi in bands}
    window = np.hanning(len(desired))[:, None, None]
    f = np.fft.rfftfreq(len(desired), 1 / fps)
    a = np.abs(np.fft.rfft((desired - desired.mean(0)) * window, axis=0)) ** 2
    b = np.abs(np.fft.rfft((actual - actual.mean(0)) * window, axis=0)) ** 2
    result = {}
    for lo, hi in bands:
        selected = (f >= lo) & (f < hi) & (f > 0)
        denominator = a[selected].sum()
        result[f'{lo:g}-{hi:g}Hz'] = (
            float(np.sqrt(b[selected].sum() / denominator)) if selected.any() and denominator > 1e-15 else None)
    return result


def _extrema(x, fps):
    # Source-angle landmarks only: prominence >= max(.1deg, 3% channel range).
    from scipy.signal import find_peaks
    prominence = max(.1, .03 * float(np.ptp(x)))
    distance = max(1, round(.15 * fps))
    return [find_peaks(sign * x, prominence=prominence, distance=distance)[0]
            for sign in (1, -1)]


def _temporal(desired, actual, fps):
    out = {'scope': 'Requested-angle gesture proxies after constraints; positive lag means emitted motion is later.',
           'hold_speed_threshold_deg_s': 1., 'extrema_tolerance_s': .5,
           'extrema_min_prominence_deg': .1, 'extrema_range_prominence_fraction': .03,
           'extrema_min_separation_s': .15}
    if not desired.size or len(desired) < 3:
        out.update({'best_velocity_lag_s': None, 'velocity_correlation_at_best_lag': None,
                    'velocity_correlation_zero_lag': None, 'hold_iou': None,
                    'requested_extrema': 0, 'emitted_extrema': 0, 'matched_extrema': 0,
                    'extrema_recall': None, 'extrema_precision': None,
                    'extrema_absolute_timing_error_s': _stats([])})
        return out
    # Limit the expensive proxy work to 64 deterministically selected outputs.
    selected = _sample(desired.shape[1], 64)
    desired, actual = desired[:, selected], actual[:, selected]
    dv, av = np.diff(desired, axis=0) * fps, np.diff(actual, axis=0) * fps
    out['sampled_moving_outputs'] = len(selected)
    # Sample comparison timestamps, but retain the original frame-level lag grid.
    # Decimating both trajectories would hide sub-250ms delays on long clips.
    max_lag = min(round(fps), max(0, (len(dv) - 1) // 4))
    timestamps = _sample(len(dv) - 2 * max_lag, 480) + max_lag
    d, a = dv[timestamps], av[timestamps]
    scored = []
    for lag in range(-max_lag, max_lag + 1):
        score = _correlation(d, av[timestamps + lag])
        if score is not None:
            scored.append((score, -abs(lag), lag))
    best = max(scored) if scored else None
    out['best_velocity_lag_s'] = float(best[2] / fps) if best else None
    out['velocity_lag_resolution_s'] = 1 / fps
    out['velocity_correlation_at_best_lag'] = best[0] if best else None
    out['velocity_correlation_zero_lag'] = _correlation(d, a)
    dh = np.linalg.norm(dv, axis=-1) < 1.
    ah = np.linalg.norm(av, axis=-1) < 1.
    out['requested_hold_fraction'] = float(dh.mean())
    out['emitted_hold_fraction'] = float(ah.mean())
    out['hold_iou'] = _ratio(np.sum(dh & ah), np.sum(dh | ah))
    out['requested_longest_hold_s_by_output'] = [_longest(dh[:, i], fps) for i in range(dh.shape[1])]
    out['emitted_longest_hold_s_by_output'] = [_longest(ah[:, i], fps) for i in range(ah.shape[1])]
    source_count, output_count, errors = 0, 0, []
    for i in range(desired.shape[1]):
        for axis in range(2):
            source = _extrema(desired[:, i, axis], fps)
            output = _extrema(actual[:, i, axis], fps)
            for expected, observed in zip(source, output):
                source_count += len(expected)
                output_count += len(observed)
                remaining = set(map(int, observed))
                for event in expected:
                    if remaining:
                        nearest = min(remaining, key=lambda t: (abs(t - event), t))
                        if abs(nearest - event) <= .5 * fps:
                            errors.append(abs(nearest - event) / fps)
                            remaining.remove(nearest)
    out.update({'requested_extrema': source_count, 'emitted_extrema': output_count,
                'matched_extrema': len(errors), 'extrema_recall': _ratio(len(errors), source_count),
                'extrema_precision': _ratio(len(errors), output_count),
                'extrema_absolute_timing_error_s': _stats(errors)})
    return out


def _dark_camera(points):
    """Inverse of sim/shaders.py camera ray: declared fixed 1080p dark camera."""
    origin = np.array([0., 6., 11.])
    forward = np.array([0., 2., -3.]) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points) - origin
    depth = delta @ forward
    if np.any(depth <= 1e-8):
        raise ValueError('Geometry behind declared diagnostic camera')
    x = (delta @ right) / depth / .62 / (1920 / 1080)
    y = (delta @ up) / depth / .62
    return np.stack((960 * (1 + x), 540 * (1 - y)), -1)


def _spatial(hits, positions, dimmer, rgb, projector=project):
    frames = _sample(len(hits), 480)
    i, j, total = _pair_indices(hits.shape[1])
    h, levels, colors = hits[frames], dimmer[frames], rgb[frames]
    out = {'sampled_frames': len(frames), 'sampled_pairs': len(i), 'total_pairs': total,
           'projection': 'Legacy geometric orthographic diagnostic: scripts.rendering.render_body_transfer.project; not dark-stage camera.',
           'scope': 'Line intersections and endpoint distances, not volumetric cone overlap or artistic quality.'}
    if not hits.shape[1]:
        out.update({'fixed_brightness_ray_crossing_fraction': None,
                    'dimmer_weighted_ray_crossing_fraction': None,
                    'radiance_proxy_weighted_ray_crossing_fraction': None,
                    'hit_pair_below_0_5m_fraction': None,
                    'screen_hit_pair_below_15px_fraction': None,
                    'horizontal_span_m': _stats([]), 'wall_hit_fraction': None})
        return out
    out['horizontal_span_m'] = _stats(np.ptp(h[..., 0], axis=1))
    out['wall_hit_fraction'] = float(np.mean(np.isclose(h[..., 2], -6, atol=1e-5)))
    out['floor_hit_fraction'] = float(np.mean(np.isclose(h[..., 1], 0, atol=1e-5)))
    if not len(i):
        out.update({k: None for k in ('fixed_brightness_ray_crossing_fraction',
                    'dimmer_weighted_ray_crossing_fraction', 'radiance_proxy_weighted_ray_crossing_fraction',
                    'hit_pair_below_0_5m_fraction', 'screen_hit_pair_below_15px_fraction')})
        return out
    endpoints, origins = projector(h), projector(positions)
    luma = colors @ np.array([.2126, .7152, .0722])
    sums = np.zeros(7)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for first in range(0, len(i), 256):
        a, b = i[first:first + 256], j[first:first + 256]
        oi, oj, ei, ej = origins[a], origins[b], endpoints[:, a], endpoints[:, b]
        crossing = ((cross(ei - oi, oj - oi) * cross(ei - oi, ej - oi) < 0) &
                    (cross(ej - oj, oi - oj) * cross(ej - oj, ei - oj) < 0))
        weight = levels[:, a] * levels[:, b]
        radiant = weight * luma[:, a] * luma[:, b]
        sums += [crossing.sum(), (crossing * weight).sum(), weight.sum(),
                 (crossing * radiant).sum(), radiant.sum(),
                 (np.linalg.norm(h[:, a] - h[:, b], axis=-1) < .5).sum(),
                 (np.linalg.norm(ei - ej, axis=-1) < 15).sum()]
    population = len(frames) * len(i)
    out.update({'fixed_brightness_ray_crossing_fraction': float(sums[0] / population),
                'dimmer_weighted_ray_crossing_fraction': _ratio(sums[1], sums[2]),
                'radiance_proxy_weighted_ray_crossing_fraction': _ratio(sums[3], sums[4]),
                'hit_pair_below_0_5m_fraction': float(sums[5] / population),
                'screen_hit_pair_below_15px_fraction': float(sums[6] / population)})
    return out


def _color(rgb, patch, fps):
    result = {'palette_cue_count': None, 'palette_id_occupancy': None,
              'scope': 'Exact quantized RGB observations only. Palette cues/IDs cannot be inferred from RGB transitions or fades.',
              'groups': {}}
    bytes_ = np.rint(rgb * 255).astype(np.uint8)
    for group, side in (('left', True), ('right', False)):
        indices = [i for i, p in enumerate(patch) if (p['position'][0] < 0) == side]
        groups = {}
        for i in indices:
            key = hashlib.sha256(bytes_[:, i].tobytes()).hexdigest()
            groups.setdefault(key, []).append(i)
        streams = []
        for indices_ in groups.values():
            values = bytes_[:, indices_[0]]
            changed = np.any(np.diff(values.astype(int), axis=0), axis=1)
            starts = np.r_[0, np.flatnonzero(changed) + 1]
            lengths = np.diff(np.r_[starts, len(values)])
            colors, counts = np.unique(values, axis=0, return_counts=True)
            order = np.argsort(-counts, kind='stable')
            streams.append({'fixture_names': [patch[i]['name'] for i in indices_],
                'RGB_changing_ticks_including_fades': int(changed.sum()),
                'distinct_RGB_colors': len(colors), 'longest_constant_RGB_s': float(lengths.max() / fps),
                'constant_RGB_run_count': len(lengths),
                'occupied_RGB': [{'rgb': colors[i].tolist(), 'frames': int(counts[i]),
                                  'fraction': float(counts[i] / len(values))} for i in order[:64]],
                'occupancy_list_truncated': bool(len(colors) > 64),
                'occupied_RGB_omitted_count': max(0, len(colors) - 64)})
        result['groups'][group] = {'fixture_count': len(indices), 'distinct_RGB_streams': len(groups), 'streams': streams}
    return result


def measure(body, intent, dmx, patch, fps=40., start_s=60., duration_s=120.):
    """Return strict-JSON diagnostics; short control inputs use their full window.

    Angles, RGB and dimmers always come from decoding ``dmx``. Pair geometry is
    bounded to 480 frames/4096 pairs; dimension/recurrence use <=240 frames and
    <=192 coordinates. Full-window exact duplicates, limits and activity remain
    un-subsampled. Invalid shapes raise; nonfinite inputs return a failed report.
    """
    if not np.isfinite([fps, start_s, duration_s]).all() or fps <= 0 or start_s < 0 or duration_s <= 0:
        raise ValueError('Finite positive fps/duration and nonnegative start required')
    body = np.asarray(body, float)
    targets, levels = np.asarray(intent['targets'], float), np.asarray(intent['dimmer'], float)
    n, t = len(patch), len(body)
    if body.shape != (t, 22, 3) or t < 2 or targets.shape != (t, n, 3) or levels.shape != (t, n):
        raise ValueError('Body, intent and patch shapes do not match')
    dmx = np.asarray(dmx)
    if len(dmx) != t or not n:
        raise ValueError('Nonempty patch and matching body/DMX timestamps required')
    arrays = {'body': body, 'targets': targets, 'dimmer': levels,
              'positions': np.asarray([p['position'] for p in patch], float)}
    for key in ('source_points', 'field_points'):
        if key in intent and intent[key] is not None:
            arrays[key] = np.asarray(intent[key], float)
            if arrays[key].ndim != 3 or len(arrays[key]) != t or arrays[key].shape[-1] != 3:
                raise ValueError(key + ' must be (T,points,3)')
    finite = {k: bool(np.isfinite(v).all()) for k, v in arrays.items()}
    report = {'scope': 'Engineering diagnostics only; activity and numerical diversity do not establish choreography.',
              'finite': all(finite.values()), 'input_finite': finite}
    if not report['finite']:
        report['status'] = 'invalid_nonfinite_input'
        return report
    decoded = decode(dmx, patch)
    if not all(np.isfinite(v).all() for v in decoded.values()):
        report.update({'finite': False, 'status': 'invalid_nonfinite_decoded_output'})
        return report
    first = round(start_s * fps)
    fallback = first >= t - 1
    if fallback:
        first, stop = 0, t
    else:
        stop = min(t, first + max(2, round(duration_s * fps)))
    sl = slice(first, stop)
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    names = [p['name'] for p in patch]
    moving_names = [name for name, yes in zip(names, moving) if yes]
    positions = arrays['positions'][moving]
    body = body[sl] - body[sl, :1] * np.array([1., 0., 1.])
    targets, levels = targets[sl], levels[sl]
    angles = decoded['angles_deg'][sl][:, moving]
    dimmer, rgb = decoded['dimmer'][sl], decoded['rgb'][sl]
    hits = ray_hits(positions, decoded['directions'][sl][:, moving])
    vectors = targets[:, moving] - positions
    length = np.linalg.norm(vectors, axis=-1)
    if np.any(length <= 1e-12):
        raise ValueError('Moving target coincides with fixture origin')
    requested = np.stack((np.rad2deg(np.arctan2(vectors[..., 0], -vectors[..., 1])),
                          np.rad2deg(np.arcsin(np.clip(vectors[..., 2] / length, -1, 1)))), axis=-1)
    speed = np.linalg.norm(np.diff(angles, axis=0) * fps, axis=-1)
    acceleration = np.linalg.norm(np.diff(angles, n=2, axis=0) * fps ** 2, axis=-1)
    desired_rays = vectors / length[..., None]
    emitted_rays = decoded['directions'][sl][:, moving]
    def ray_angle(a, b):
        return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                    np.sum(a*b, axis=-1)))
    report.update({'status': 'measured',
        'window': {'first_frame': first, 'stop_frame_exclusive': stop,
                   'start_s': first / fps, 'duration_s': (stop - first) / fps,
                   'requested_start_s': start_s, 'requested_duration_s': duration_s,
                   'used_full_short_control': fallback, 'fps': fps},
        'counts': {'moving': int(moving.sum()), 'bulbs': int((~moving).sum()), 'total': n},
        'motion': {'fixture_names': moving_names, 'pan_tilt_range_deg_by_output': np.ptp(angles, axis=0).tolist(),
            'speed_deg_s': _stats(speed), 'acceleration_deg_s2': _stats(acceleration),
            'fraction_below_1_deg_s_by_output': (speed < 1).mean(0).tolist(),
            'constant_by_output': np.all(angles == angles[:1], axis=(0, 2)).tolist(),
            'longest_below_1_deg_s_by_output': [_longest(speed[:, i] < 1, fps) for i in range(len(moving_names))]},
        'duplicates': {'desired_targets': _duplicates(targets[:, moving], moving_names),
                       'emitted_angles': _duplicates(angles, moving_names),
                       'emitted_dimmer': _duplicates(dimmer, names)},
        'brightness': {},
        'spatial': _spatial(hits, positions, dimmer[:, moving], rgb[:, moving]),
        'angular_transfer': {'angle_error_deg': _stats(np.linalg.norm(requested - angles, axis=-1)),
            'band_emitted_to_requested_RMS_ratio': _angular_bands(requested, angles, fps),
            'band_method': 'Hann-window pooled angular energy, DC removed; descriptive ratio, not a linear transfer gain.',
            'temporal': _temporal(requested, angles, fps)},
        'ray_transfer': {'angle_error_deg': _stats(ray_angle(desired_rays, emitted_rays)),
            'requested_speed_deg_s': _stats(ray_angle(desired_rays[1:], desired_rays[:-1])*fps),
            'emitted_speed_deg_s': _stats(ray_angle(emitted_rays[1:], emitted_rays[:-1])*fps),
            'scope': 'True 3D ray-direction angle, separate from motor pan/tilt changes near a coordinate pole.'},
        'color': _color(rgb, patch, fps)})
    report['spatial']['crossing_fraction'] = report['spatial']['fixed_brightness_ray_crossing_fraction']
    report['motion']['max_speed_deg_s'] = report['motion']['speed_deg_s']['max']
    report['motion']['max_acceleration_deg_s2'] = report['motion']['acceleration_deg_s2']['max']
    duplicate_groups = report['duplicates']['desired_targets']['duplicate_groups']
    pairs = []
    pair_count = sum(len(g) * (len(g) - 1) // 2 for g in duplicate_groups)
    for group in duplicate_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if len(pairs) == 4096:
                    break
                pairs.append([group[i], group[j]])
            if len(pairs) == 4096:
                break
        if len(pairs) == 4096:
            break
    report['duplicates']['desired_target_pairs'] = pairs
    report['duplicates']['desired_target_pair_count'] = pair_count
    report['duplicates']['desired_target_pairs_truncated'] = pair_count > len(pairs)
    try:
        secondary = _spatial(hits, positions, dimmer[:, moving], rgb[:, moving], _dark_camera)
        report['spatial']['fixed_dark_camera_secondary'] = {
            'crossing_fraction': secondary['fixed_brightness_ray_crossing_fraction'],
            'dimmer_weighted_crossing_fraction': secondary['dimmer_weighted_ray_crossing_fraction'],
            'screen_hit_pair_below_15px_fraction': secondary['screen_hit_pair_below_15px_fraction'],
            'camera': {'position': [0., 6., 11.], 'target': [0., 2., -3.],
                       'scale': .62, 'width': 1920, 'height': 1080},
            'scope': 'Fixed declared perspective camera; no occlusion, clipping or volume integration.'}
    except ValueError:
        report['spatial']['fixed_dark_camera_secondary'] = {'crossing_fraction': None,
            'scope': 'Undefined: one or more points behind the fixed camera.'}
    for label, mask in (('moving', moving), ('bulbs', ~moving)):
        values = dimmer[:, mask]
        report['brightness'][label] = {'level': _stats(values),
            'range_by_output': np.ptp(values, axis=0).tolist(),
            'fraction_at_zero': float(np.mean(values == 0)) if values.size else None,
            'fraction_at_full': float(np.mean(values == 1)) if values.size else None,
            'fraction_at_or_below_5pct': float(np.mean(values <= .05)) if values.size else None,
            'fraction_at_or_above_95pct': float(np.mean(values >= .95)) if values.size else None,
            'mean_simultaneously_above_10pct': float((values > .1).sum(1).mean()) if values.size else None}
    field = arrays['field_points'][sl] if 'field_points' in arrays else None
    sources = arrays['source_points'][sl] if 'source_points' in arrays else None
    report['dimensions'] = {key: _dimension(value) for key, value in
        (('body_root_recentered', body), ('field', field), ('source_points', sources),
         ('desired_targets', targets[:, moving]), ('actual_hits', hits), ('emitted_dimmer', dimmer))}
    # Bulb-only installations use their intended/emitted levels for recurrence.
    desired_repr, emitted_repr = (targets[:, moving], hits) if moving.any() else (levels, dimmer)
    report['recurrence'] = _recurrence(body, field, desired_repr, emitted_repr)
    return report
