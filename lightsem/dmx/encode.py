"""Vectorised DMX packing: per-fixture parameter arrays -> (T, 512) universe.

``dmx/scenes.py`` writes one frame at a time through ``Fixture.write``; that is
fine for a single render and far too slow for 1408 clips. This module packs a
whole stream at once and is the exact inverse of ``dmx/decode.py`` — the pair
is round-trip tested, which is what makes the evaluation metrics trustworthy:
a metric computed on a decoded stream is a metric on what the rig would
actually emit, not on what the conductor intended.
"""

from __future__ import annotations

import numpy as np

from .fixtures import UNIVERSE_SIZE, Rig

PARAM_NAMES = ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom", "gobo")


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray):
    """Vectorised HSV -> RGB, all inputs in 0..1 (hue wraps)."""
    h = np.mod(h, 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return r, g, b


def _u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if not np.all(np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError('DMX normalised parameters must be finite and within [0, 1]')
    return (x * 255.0 + 0.5).astype(np.uint8)


def pack_universe(rig: Rig, params: dict[str, np.ndarray]) -> np.ndarray:
    """Pack ``{param: (T, n_fixtures)}`` into a (T, 512) uint8 universe stream."""
    n_fix = len(rig.fixtures)
    occupied = np.zeros(UNIVERSE_SIZE, dtype=bool)
    for f in rig.fixtures:
        if f.kind not in {"moving_head", "wash", "blinder", "strobe", "projector"}:
            raise ValueError(f"unknown fixture kind {f.kind}")
        if isinstance(f.address, (bool, np.bool_)) or not isinstance(f.address, (int, np.integer)) or f.address < 1 or f.address + f.n_channels - 1 > UNIVERSE_SIZE:
            raise ValueError(f"fixture {f.name!r} address range is outside the 512-channel universe")
        span = slice(f.address - 1, f.address - 1 + f.n_channels)
        if occupied[span].any():
            raise ValueError(f"fixture {f.name!r} overlaps another fixture's DMX address range")
        occupied[span] = True
    if not params:
        raise ValueError('at least one parameter array is required')
    first = np.asarray(next(iter(params.values())))
    if first.ndim != 2:
        raise ValueError('parameter arrays must have shape (T, n_fixtures)')
    T = int(first.shape[0])
    if T <= 0:
        raise ValueError('parameter arrays must contain at least one frame')
    for k, v in params.items():
        if k not in PARAM_NAMES:
            raise ValueError(f'unknown DMX parameter {k!r}')
        v = np.asarray(v)
        if not np.issubdtype(v.dtype, np.number) or not np.all(np.isfinite(v)):
            raise ValueError(f'param {k} must contain finite numeric values')
        if v.shape != (T, n_fix):
            raise ValueError(f"param {k} has shape {v.shape}, expected {(T, n_fix)}")
        lo, hi = (0.0, 255.0) if k == 'gobo' else (0.0, 1.0)
        if k != 'hue' and np.any((v < lo) | (v > hi)):
            raise ValueError(f'param {k} is outside [{lo}, {hi}]')
        if k == 'gobo' and np.any(v != np.floor(v)):
            raise ValueError('param gobo must contain integer channel values')
    get = lambda k, d: params.get(k, np.full((T, n_fix), d))   # noqa: E731
    dim, strobe = get("dimmer", 0.0), get("strobe", 0.0)
    hue, sat, val = get("hue", 0.0), get("sat", 1.0), get("val", 1.0)
    pan, tilt, zoom = get("pan", 0.5), get("tilt", 0.5), get("zoom", 0.5)
    gobo = get("gobo", 0.0)
    r, g, b = hsv_to_rgb(hue, sat, val)

    out = np.zeros((T, UNIVERSE_SIZE), dtype=np.uint8)
    for i, f in enumerate(rig.fixtures):
        a = f.address - 1
        if f.kind == "moving_head":
            p16 = (pan[:, i] * 65535).astype(np.uint32)
            t16 = (tilt[:, i] * 65535).astype(np.uint32)
            out[:, a] = (p16 >> 8).astype(np.uint8)
            out[:, a + 1] = (p16 & 0xFF).astype(np.uint8)
            out[:, a + 2] = (t16 >> 8).astype(np.uint8)
            out[:, a + 3] = (t16 & 0xFF).astype(np.uint8)
            out[:, a + 4] = _u8(dim[:, i]); out[:, a + 5] = _u8(strobe[:, i])
            out[:, a + 6] = _u8(r[:, i]); out[:, a + 7] = _u8(g[:, i])
            out[:, a + 8] = _u8(b[:, i]); out[:, a + 9] = _u8(zoom[:, i])
            out[:, a + 10] = _u8(np.full(T, 0.5))
            out[:, a + 11] = gobo[:, i].astype(np.uint8)
        elif f.kind == "wash":
            out[:, a] = _u8(dim[:, i]); out[:, a + 1] = _u8(strobe[:, i])
            out[:, a + 2] = _u8(r[:, i]); out[:, a + 3] = _u8(g[:, i])
            out[:, a + 4] = _u8(b[:, i]); out[:, a + 5] = _u8(zoom[:, i])
            out[:, a + 6] = _u8(sat[:, i])
        elif f.kind == "blinder":
            out[:, a] = _u8(dim[:, i]); out[:, a + 1] = _u8(strobe[:, i])
        elif f.kind == "strobe":
            out[:, a] = _u8(dim[:, i]); out[:, a + 1] = _u8(strobe[:, i])
            out[:, a + 2] = _u8(val[:, i])
        elif f.kind == "projector":
            out[:, a] = _u8(dim[:, i]); out[:, a + 1] = _u8(strobe[:, i])
            out[:, a + 2] = _u8(r[:, i]); out[:, a + 3] = _u8(g[:, i])
            out[:, a + 4] = _u8(b[:, i]); out[:, a + 5] = _u8(zoom[:, i])
            out[:, a + 6] = _u8(np.full(T, 0.5))
            out[:, a + 7] = gobo[:, i].astype(np.uint8)
            out[:, a + 8] = _u8(pan[:, i]); out[:, a + 9] = _u8(tilt[:, i])
        else:
            raise ValueError(f"unknown fixture kind {f.kind}")
    return out


def slew_limit(x: np.ndarray, fps: float, max_rate: np.ndarray | float,
               wrap: bool = False) -> np.ndarray:
    """Rate-limit each column of ``x`` to ``max_rate`` units per second.

    Real fixtures cannot teleport: a wash dimmer takes ~0.1 s, a moving head
    pans at ~540 deg/s. Targets that ignore this teach the model to ask for
    motion the rig cannot make, and the render then looks smoother than the
    real venue would.
    """
    x = np.asarray(x, dtype=np.float64)
    step = np.broadcast_to(np.atleast_1d(max_rate) / fps, (x.shape[1],))
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        d = x[t] - out[t - 1]
        if wrap:
            d = (d + 0.5) % 1.0 - 0.5
        out[t] = out[t - 1] + np.clip(d, -step, step)
    if wrap:
        out = np.mod(out, 1.0)
    return out
