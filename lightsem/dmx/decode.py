"""Decode a DMX universe stream back into per-fixture parameters.

Every metric in ``lightsem/eval`` reads the light through this decoder rather
than through raw channel bytes. Raw bytes would be actively misleading: a
moving head's pan is 16-bit, so its fine channel wraps 0->255 many times per
sweep and a naive "byte changed a lot" detector would score a smooth pan as a
storm of events.

The renderer has its own copy of this logic (``sim/renderer._decode_fixture``)
because it decodes one frame at a time on the GPU path. This module is the
vectorised, whole-stream version. If a fixture personality changes, both must
change; the shared test in ``lightsem/eval/tests`` compares them.
"""

from __future__ import annotations

import colorsys

import numpy as np

from .fixtures import Rig

PARAMS = ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom")


def _rgb_to_hsv(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    hue = np.zeros_like(mx)
    m = d > 1e-9
    rm, gm, bm = r[m], g[m], b[m]
    mxm, dm = mx[m], d[m]
    h = np.where(mxm == rm, ((gm - bm) / dm) % 6,
                 np.where(mxm == gm, (bm - rm) / dm + 2, (rm - gm) / dm + 4))
    hue[m] = h / 6.0
    sat = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return hue, sat, mx


def decode_stream(rig: Rig, frames: np.ndarray) -> dict[str, np.ndarray]:
    """(T, 512) uint8 universe stream -> {param: (T, n_fixtures) float}."""
    frames = np.asarray(frames)
    if frames.ndim != 2 or frames.shape[1] < 1:
        raise ValueError(f"expected (T, 512) frames, got {frames.shape}")
    u = frames.astype(np.float64) / 255.0
    raw = frames.astype(np.float64)
    T = frames.shape[0]
    fx = rig.fixtures
    out = {p: np.zeros((T, len(fx))) for p in PARAMS}
    kinds = []
    for i, f in enumerate(fx):
        a = f.address - 1
        kinds.append(f.kind)

        def ch(off: int) -> np.ndarray:
            j = a + off
            return u[:, j] if 0 <= j < frames.shape[1] else np.zeros(T)

        if f.kind == "moving_head":
            pan = (raw[:, a] * 256 + raw[:, a + 1]) / 65535.0
            tilt = (raw[:, a + 2] * 256 + raw[:, a + 3]) / 65535.0
            dim, strobe = ch(4), ch(5)
            r, g, b = ch(6), ch(7), ch(8)
            zoom = ch(9)
        elif f.kind == "wash":
            pan = tilt = np.full(T, 0.5)
            dim, strobe = ch(0), ch(1)
            r, g, b = ch(2), ch(3), ch(4)
            zoom = ch(5)
        elif f.kind == "blinder":
            pan = tilt = np.full(T, 0.5)
            dim, strobe = ch(0), ch(1)
            r = g = b = dim
            zoom = np.full(T, 0.5)
        elif f.kind == "strobe":
            pan = tilt = np.full(T, 0.5)
            dim, strobe = ch(0), ch(1)
            r = g = b = ch(2)
            zoom = np.full(T, 0.5)
        elif f.kind == "projector":
            dim, strobe = ch(0), ch(1)
            r, g, b = ch(2), ch(3), ch(4)
            zoom = ch(5)
            pan, tilt = ch(8), ch(9)
        else:
            raise ValueError(f"unknown fixture kind {f.kind}")

        hue, sat, val = _rgb_to_hsv(r, g, b)
        out["dimmer"][:, i] = dim
        out["strobe"][:, i] = strobe
        out["hue"][:, i] = hue
        out["sat"][:, i] = sat
        out["val"][:, i] = val
        out["pan"][:, i] = pan
        out["tilt"][:, i] = tilt
        out["zoom"][:, i] = zoom
    out["kind"] = np.array(kinds)
    return out


def emitted_light(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """(T, n_fixtures) actually-emitted intensity = dimmer x value."""
    return decoded["dimmer"] * np.maximum(decoded["val"], 1e-6)


def brightness(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """(T,) mean emitted intensity across the rig — the A/B matching quantity."""
    return emitted_light(decoded).mean(axis=1)
