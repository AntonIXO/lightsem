"""Continuous, arbitrary-N color fields for an isolated lighting experiment.

The accepted P2 layer is intentionally unchanged.  This module is an
experimental decoder that turns two authenticated P2 side colors plus the
saved body into a continuous field over fixture coordinates.  The field is
source conditioned: its phase only advances when body displacement or the
audio color stream changes.  It therefore freezes when both sources freeze;
there is no free-running oscillator or brightness controller here.

Colors are interpolated in OKLab and gamut-mapped back to sRGB.  Fixture count
is a query, not a model dimension, so the same field works for any N.
"""
from __future__ import annotations

import numpy as np


# Björn Ottosson's OKLab matrices (CSS Color 4, D65).
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= .04045, rgb / 12.92,
                    ((rgb + .055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .0031308, 12.92 * rgb,
                    1.055 * np.maximum(rgb, 0.) ** (1. / 2.4) - .055)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3 or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError('sRGB values must be finite and in [0,1]')
    lms = np.einsum('...j,ij->...i', _srgb_to_linear(rgb), _M1)
    return np.einsum('...j,ij->...i', np.cbrt(lms), _M2)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    if lab.shape[-1] != 3 or not np.isfinite(lab).all():
        raise ValueError('OKLab values must be finite')
    lms = np.einsum('...j,ij->...i', lab, _M2_INV)
    linear = np.einsum('...j,ij->...i', lms ** 3, _M1_INV)
    return _linear_to_srgb(linear)


def _gamut_map(lab: np.ndarray) -> np.ndarray:
    """Reduce chroma along the OKLab hue ray until sRGB is in gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    result = lab.copy()
    rgb = oklab_to_srgb(result)
    outside = np.any((rgb < 0.) | (rgb > 1.), axis=-1)
    if np.any(outside):
        flat = result.reshape(-1, 3)
        mask = outside.reshape(-1)
        for index in np.flatnonzero(mask):
            l, a, b = flat[index]
            low, high = 0., 1.
            for _ in range(14):
                scale = (low + high) / 2.
                candidate = np.array([l, a * scale, b * scale])
                candidate_rgb = oklab_to_srgb(candidate)
                if np.all((candidate_rgb >= 0.) & (candidate_rgb <= 1.)):
                    low = scale
                else:
                    high = scale
            flat[index] = [l, a * low, b * low]
        result = flat.reshape(result.shape)
    return np.clip(oklab_to_srgb(result), 0., 1.)


def _hex_palette(values):
    rows = []
    for value in values:
        text = str(value).lstrip('#')
        if len(text) != 6:
            raise ValueError('palette entries must be six-digit #RRGGBB colors')
        try:
            rows.append([int(text[i:i + 2], 16) / 255. for i in (0, 2, 4)])
        except ValueError as exc:
            raise ValueError('invalid hexadecimal palette entry') from exc
    if len(rows) < 2:
        raise ValueError('at least two palette anchors are required')
    return np.asarray(rows, dtype=np.float64)


DEFAULT_PALETTE = ('#11c8ff', '#5533ff', '#ee2e9a', '#ff9f1c', '#2de38b', '#f6f7ff')


def _cyclic_interpolate(palette_lab: np.ndarray, positions: np.ndarray) -> np.ndarray:
    count = len(palette_lab)
    scaled = np.mod(positions, 1.) * count
    left = np.floor(scaled).astype(np.int64) % count
    weight = scaled - np.floor(scaled)
    right = (left + 1) % count
    return (1. - weight)[..., None] * palette_lab[left] + weight[..., None] * palette_lab[right]


def _source_flow(world40: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world = np.asarray(world40, dtype=np.float64)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or not np.isfinite(world).all():
        raise ValueError('world40 must be finite [T,22,3]')
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError('left/right must be matching [T,3] RGB arrays')
    if len(world) != len(left):
        raise ValueError('body and color clocks must have equal length')
    relative = world - world[:, 0:1]
    velocity = np.diff(relative, axis=0, prepend=relative[:1]) * 40.
    # Signed articulation controls direction; speed controls transport amount.
    hands = relative[:, [7, 11, 15, 19]]
    signed = np.tanh((hands[..., 0] + .45 * hands[..., 2]).mean(axis=1) / .7)
    speed = np.linalg.norm(velocity[:, [7, 11, 15, 19]], axis=-1).mean(axis=1)
    p2 = np.linalg.norm(np.diff(left.astype(np.float64), axis=0, prepend=left[:1]), axis=1)
    p2 += np.linalg.norm(np.diff(right.astype(np.float64), axis=0, prepend=right[:1]), axis=1)
    drive = np.clip(.012 * speed + .002 * p2, 0., .35)
    phase = np.cumsum(drive) + .12 * signed
    return phase, np.clip(.18 + .82 * np.tanh(speed / 1.5), 0., 1.)


def continuous_color_field(world40: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray,
                           outputs, *, palette=DEFAULT_PALETTE, spatial_cycles=1.35,
                           temporal_gain=1.0) -> tuple[np.ndarray, dict]:
    """Return quantized ``[T,N,3]`` RGB for arbitrary fixture counts.

    The left/right P2 stream contributes the audio-conditioned hue anchor;
    the supplied palette is an arbitrary user-editable color vocabulary.  A
    body-derived transport phase moves the gradient through fixture space.
    """
    world = np.asarray(world40, dtype=np.float64)
    left = np.asarray(left_rgb, dtype=np.uint8)
    right = np.asarray(right_rgb, dtype=np.uint8)
    if left.ndim != 2 or left.shape[1:] != (3,) or right.shape != left.shape:
        raise ValueError('left_rgb/right_rgb must have shape [T,3]')
    palette_rgb = _hex_palette(palette)
    palette_lab = srgb_to_oklab(palette_rgb)
    base_left, base_right = srgb_to_oklab(left / 255.), srgb_to_oklab(right / 255.)
    phase, activity = _source_flow(world, left, right)
    n = len(outputs)
    if n < 1:
        raise ValueError('at least one fixture is required')
    coordinates = np.array([float(o.get('position', [i, 0, 0])[0]) for i, o in enumerate(outputs)])
    if np.ptp(coordinates) < 1e-12:
        coordinates = np.arange(n, dtype=np.float64)
    coordinates = (coordinates - coordinates.min()) / max(np.ptp(coordinates), 1e-12)
    # P2 hue is a slowly changing audio anchor; lightness/chroma stay in the
    # palette field so brightness remains exclusively the dimmer channel.
    anchor_hue = np.arctan2((base_left[:, 2] + base_right[:, 2]) * .5,
                            (base_left[:, 1] + base_right[:, 1]) * .5)
    anchor_hue = np.mod(anchor_hue / (2. * np.pi), 1.)
    positions = (anchor_hue[:, None] + temporal_gain * phase[:, None] * .055
                 + spatial_cycles * coordinates[None, :]) % 1.
    lab = _cyclic_interpolate(palette_lab, positions)
    # Preserve the source's audio-conditioned left/right lightness envelope
    # without coupling it to dimmer; this is a color appearance parameter.
    source_lightness = np.clip(.72 * (base_left[:, 0] + base_right[:, 0]) * .5 + .14, .18, .92)
    lab[..., 0] = np.clip(.74 * lab[..., 0] + .26 * source_lightness[:, None], .12, .96)
    rgb = np.rint(_gamut_map(lab) * 255.).astype(np.uint8)
    metadata = {
        'schema_version': 1,
        'algorithm': 'oklab_continuous_fixture_field',
        'palette': list(palette),
        'fixture_count': n,
        'spatial_cycles': float(spatial_cycles),
        'temporal_gain': float(temporal_gain),
        'temporal_source': 'saved body displacement/speed plus authenticated P2 RGB changes',
        'brightness_separate': True,
        'arbitrary_n': True,
        'no_free_running_oscillator': True,
        'gamut_mapping': 'OKLab chroma reduction then sRGB quantization',
        'activity_mean': float(activity.mean()),
    }
    return rgb, metadata


def color_field_metrics(rgb: np.ndarray, *, dimmer=None) -> dict:
    """Diagnostics for flow, diversity and color/brightness separation."""
    values = np.asarray(rgb, dtype=np.float64) / 255.
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('rgb must be finite [T,N,3]')
    lab = srgb_to_oklab(values)
    delta_t = np.diff(lab, axis=0) if len(lab) > 1 else np.empty((0,) + lab.shape[1:])
    delta_x = np.diff(lab, axis=1) if lab.shape[1] > 1 else np.empty((len(lab), 0, 3))
    flat = lab.reshape(len(lab), -1)
    if flat.shape[1] and np.any(np.std(flat, axis=0) > 1e-9):
        singular = np.linalg.svd(flat - flat.mean(0), full_matrices=False, compute_uv=False)
        effective_rank = float((singular.sum() ** 2) / max((singular ** 2).sum(), 1e-12))
    else:
        effective_rank = 0.
    unique_streams = len({np.ascontiguousarray(rgb[:, i]).tobytes() for i in range(rgb.shape[1])})
    report = {
        'finite': bool(np.isfinite(values).all()),
        'temporal_oklab_velocity_mean': float(np.linalg.norm(delta_t, axis=-1).mean()) if delta_t.size else 0.,
        'spatial_oklab_gradient_mean': float(np.linalg.norm(delta_x, axis=-1).mean()) if delta_x.size else 0.,
        'spatial_gradient_active_fraction': float(np.mean(np.linalg.norm(delta_x, axis=-1) > .01)) if delta_x.size else 0.,
        'fixture_stream_effective_rank': effective_rank,
        'unique_fixture_streams': int(unique_streams),
        'exact_duplicate_stream_fraction': float(1. - unique_streams / rgb.shape[1]),
        'rgb_min': float(values.min()), 'rgb_max': float(values.max()),
    }
    if dimmer is not None:
        dim = np.asarray(dimmer, dtype=np.float64)
        if dim.shape != values.shape[:2] or not np.isfinite(dim).all():
            raise ValueError('dimmer must be finite [T,N]')
        lab_l = lab[..., 0]
        report['brightness_color_lstar_corr'] = float(np.corrcoef(dim.ravel(), lab_l.ravel())[0, 1]) if np.std(dim) > 1e-12 and np.std(lab_l) > 1e-12 else 0.
        report['dimmer_mean'] = float(dim.mean())
    return report

