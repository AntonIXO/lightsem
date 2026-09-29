"""Pure full-record, nonangular overlays on an accepted DMX byte stream.

This module never decodes/re-encodes the baseline, changes a fixture state, or
creates timing/color decisions. Callers supply complete trajectories or events.
RGB means normalized DMX channel values, NOT sRGB or calibrated linear-light
values: 0 maps to byte 0, 1 to 255, with round-half-up quantization. No gamma,
HSV conversion, brightness normalization, or dimmer compensation is applied.

Only moving-head/wash/projector RGB and moving-head/wash dimmers are writable.
All other bytes, including projector dimmers and every strobe/blinder byte,
remain exact. Apply to the entire record before taking any review crop.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ...dmx.fixtures import Fixture, Rig, UNIVERSE_SIZE

# Zero-based offsets match dmx/fixtures.py and dmx/encode.py. Never use the
# general encoder here: it would rewrite fields that this overlay does not own.
_CHANNEL_COUNTS = {"moving_head": 12, "wash": 7, "projector": 10,
                   "blinder": 2, "strobe": 3}
_RGB_OFFSETS = {"moving_head": (6, 7, 8), "wash": (2, 3, 4),
                "projector": (2, 3, 4)}
_DIMMER_OFFSETS = {"moving_head": 4, "wash": 0}


@dataclass(frozen=True)
class OverlayConfig:
    """Integer byte ceilings, applied ONLY to explicitly requested fields.

    These are software bounds, not a physical safety certification. A config
    alone changes nothing, even when the baseline exceeds these ceilings.
    """

    rgb_ceiling_byte: int = 255
    dimmer_ceiling_byte: int = 255

    def __post_init__(self):
        for name in ("rgb_ceiling_byte", "dimmer_ceiling_byte"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                    or not 0 <= value <= 255):
                raise ValueError(f"{name} must be an integer byte in [0,255]")


def _numbers(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite real numbers")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return array.astype(np.float64, copy=False)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{label} must fit finite float64 values") from exc


def _times(times, count=None):
    raw = np.asarray(times)
    if raw.dtype.kind != "f":
        raise ValueError("times must be floating-point full-record seconds")
    times = _numbers(raw, "times")
    if times.ndim != 1 or (count is not None and times.shape != (count,)):
        raise ValueError("times must have shape (T,) matching the full baseline")
    if len(times):
        if times[0] != 0.0:
            raise ValueError("full-record times must start at zero; overlay before cropping")
        steps = np.diff(times)
        if np.any(steps <= 0):
            raise ValueError("times must be strictly increasing")
        if len(steps) and not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-8):
            raise ValueError("full-record DMX timestamps must be uniformly spaced")
    return times


def _patch(rig):
    if not isinstance(rig, Rig):
        raise ValueError("rig must be a Rig")
    by_name = {}
    occupied = np.zeros(UNIVERSE_SIZE, dtype=bool)
    for fixture in rig.fixtures:
        if not isinstance(fixture, Fixture):
            raise ValueError("rig entries must be Fixture instances")
        if not isinstance(fixture.name, str) or not fixture.name or fixture.name in by_name:
            raise ValueError("fixture names must be nonempty and unique")
        if not isinstance(fixture.kind, str) or fixture.kind not in _CHANNEL_COUNTS:
            raise ValueError(f"unsupported fixture kind: {fixture.kind}")
        address = fixture.address
        if isinstance(address, (bool, np.bool_)) or not isinstance(address, Integral):
            raise ValueError(f"{fixture.name}: DMX address must be an integer")
        start = int(address) - 1
        end = start + _CHANNEL_COUNTS[fixture.kind]
        if start < 0 or end > UNIVERSE_SIZE:
            raise ValueError(f"{fixture.name}: fixture patch must fit channels 1..512")
        if occupied[start:end].any():
            raise ValueError(f"{fixture.name}: overlapping fixture patch")
        occupied[start:end] = True
        by_name[fixture.name] = fixture
    return by_name


def allowed_slot_mask(rig: Rig) -> np.ndarray:
    """Return a new (512,) bool mask of all fields this module may change."""
    fixtures = _patch(rig)
    mask = np.zeros(UNIVERSE_SIZE, dtype=bool)
    for fixture in fixtures.values():
        start = fixture.address - 1
        for offset in _RGB_OFFSETS.get(fixture.kind, ()):
            mask[start + offset] = True
        if fixture.kind in _DIMMER_OFFSETS:
            mask[start + _DIMMER_OFFSETS[fixture.kind]] = True
    return mask


def _mapping(value, label):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must map fixture names to tracks")
    return value


def _fixture(fixtures, name, supported, label):
    if not isinstance(name, str) or name not in fixtures:
        raise ValueError(f"{label}: unknown fixture {name!r}")
    fixture = fixtures[name]
    if fixture.kind not in supported:
        raise ValueError(f"{label}: {fixture.kind} is not permitted")
    return fixture


def _track(value, shape, label, upper=None):
    track = _numbers(value, label)
    if track.shape != shape:
        raise ValueError(f"{label} must have exact full-record shape {shape}")
    if np.any(track < 0) or (upper is not None and np.any(track > upper)):
        bounds = "[0,1]" if upper == 1 else "[0,+infinity)"
        raise ValueError(f"{label} must be in {bounds}")
    return track


def _bytes(value, ceiling):
    return np.floor(np.clip(value, 0, ceiling) + 0.5).astype(np.uint8)


def apply_overlay(
    baseline: np.ndarray, rig: Rig, times: np.ndarray, *,
    rgb: Mapping[str, np.ndarray] | None = None,
    dimmer_gain: Mapping[str, np.ndarray] | None = None,
    dimmer_envelope: Mapping[str, np.ndarray] | None = None,
    config: OverlayConfig | None = None,
) -> np.ndarray:
    """Return an independent uint8 (T,512) copy with only requested edits.

    ``rgb`` maps fixture names to exact (T,3) channel-RGB tracks in [0,1].
    ``dimmer_gain`` maps names to exact (T,) finite nonnegative multipliers on
    saved dimmer bytes; results saturate at the configured integer ceiling.
    ``dimmer_envelope`` maps names to absolute (T,) dimmers in [0,1]. A fixture
    cannot appear in both dimmer mappings. No scalar broadcasting/resampling.
    Empty mappings and default calls are byte-exact copies. RGB never scales
    dimmers; dimmers never scale RGB. Unknown names/unsupported fields fail.

    ``times`` is the matching uniform, zero-origin floating-point timeline.
    Cropped calls with a nonzero origin are rejected. A caller must not rebase
    crop timestamps to pretend they represent the full record.
    """
    if not isinstance(baseline, np.ndarray) or baseline.dtype != np.uint8:
        raise ValueError("baseline must be a uint8 numpy array")
    if baseline.ndim != 2 or baseline.shape[1] != UNIVERSE_SIZE:
        raise ValueError("baseline must have shape (T,512)")
    _times(times, len(baseline))
    fixtures = _patch(rig)
    cfg = OverlayConfig() if config is None else config
    if not isinstance(cfg, OverlayConfig):
        raise ValueError("config must be OverlayConfig")
    colors = _mapping(rgb, "rgb")
    gains = _mapping(dimmer_gain, "dimmer_gain")
    envelopes = _mapping(dimmer_envelope, "dimmer_envelope")
    if set(gains).intersection(envelopes):
        raise ValueError("a fixture cannot have both dimmer_gain and dimmer_envelope")

    out = baseline.copy()
    requested = np.zeros(UNIVERSE_SIZE, dtype=bool)
    count = len(baseline)
    for name, values in colors.items():
        fixture = _fixture(fixtures, name, _RGB_OFFSETS, "rgb")
        track = _track(values, (count, 3), f"rgb[{name}]", upper=1)
        slots = fixture.address - 1 + np.array(_RGB_OFFSETS[fixture.kind])
        out[:, slots] = _bytes(track * 255, cfg.rgb_ceiling_byte)
        requested[slots] = True
    for label, tracks in (("dimmer_gain", gains), ("dimmer_envelope", envelopes)):
        for name, values in tracks.items():
            fixture = _fixture(fixtures, name, _DIMMER_OFFSETS, label)
            track = _track(values, (count,), f"{label}[{name}]",
                           upper=1 if label == "dimmer_envelope" else None)
            slot = fixture.address - 1 + _DIMMER_OFFSETS[fixture.kind]
            # Clamping gains before multiplication avoids overflow even for
            # finite 1e308 gains. Every positive baseline byte then saturates.
            value = (baseline[:, slot].astype(float) * np.minimum(track, 255)
                     if label == "dimmer_gain" else track * 255)
            out[:, slot] = _bytes(value, cfg.dimmer_ceiling_byte)
            requested[slot] = True
    allowed = allowed_slot_mask(rig)
    if np.any(requested & ~allowed) or not np.array_equal(out[:, ~requested], baseline[:, ~requested]):
        raise AssertionError("overlay changed an unrequested or forbidden DMX slot")
    return out


@dataclass(frozen=True)
class AccentEvent:
    """One externally supplied pulse: peak time, strength, full support width."""

    time: float
    strength: float
    width_s: float


@dataclass(frozen=True)
class AccentConfig:
    """Caller-supplied fixed bounds; register these before using model events.

    No defaults choose musical timing or pulse decisions. Strength is a dimmer
    gain increment (0..1), and width is the complete raised-cosine support.
    Refractory spacing is a validation rule, not a top-k/drop-event policy.
    These software bounds are NOT a hardware or photosensitivity guarantee.
    """

    max_strength: float
    min_width_s: float
    max_width_s: float
    refractory_s: float

    def __post_init__(self):
        values = _numbers([self.max_strength, self.min_width_s,
                           self.max_width_s, self.refractory_s], "accent config")
        if (not 0 <= values[0] <= 1 or not 0 < values[1] <= values[2]
                or values[3] <= 0):
            raise ValueError("accent config needs strength in [0,1], positive widths/refractory")


def sample_accent_events(times, events: Sequence[AccentEvent], *, config: AccentConfig):
    """Sample bounded smooth single pulses; never generate, reorder, or rank events.

    Empty events return zeros. Overlaps use maximum, never accumulate energy.
    Event times must lie on the full record and be sorted with the configured
    refractory spacing. Peaks need not align to DMX frame timestamps.
    """
    times = _times(times)
    if not isinstance(config, AccentConfig):
        raise ValueError("config must be an explicit AccentConfig")
    if not isinstance(events, Sequence):
        raise ValueError("events must be a sequence of AccentEvent")
    pulse = np.zeros(len(times), dtype=float)
    previous = None
    for event in events:
        if not isinstance(event, AccentEvent):
            raise ValueError("events must contain AccentEvent instances")
        values = _numbers([event.time, event.strength, event.width_s], "accent event")
        time, strength, width = values
        if not len(times) or not 0 <= time <= times[-1]:
            raise ValueError("accent time is outside the full record")
        if not 0 <= strength <= config.max_strength:
            raise ValueError("accent strength exceeds registered bounds")
        if not config.min_width_s <= width <= config.max_width_s:
            raise ValueError("accent width exceeds registered bounds")
        if previous is not None and time - previous < config.refractory_s - 1e-12:
            raise ValueError("accent times must be sorted and obey refractory spacing")
        previous = time
        distance = np.abs(times - time) / (width / 2)
        active = distance < 1
        single = np.zeros_like(pulse)
        single[active] = strength * (0.5 + 0.5 * np.cos(np.pi * distance[active]))
        np.maximum(pulse, single, out=pulse)
    return pulse


def apply_accent_events(
    baseline: np.ndarray, rig: Rig, times: np.ndarray,
    events: Mapping[str, Sequence[AccentEvent]] | None = None, *,
    accent_config: AccentConfig, config: OverlayConfig | None = None,
) -> np.ndarray:
    """Apply supplied dimmer accents only; preserve baseline outside each pulse.

    Zero/empty events abstain byte-exactly, even with restrictive byte ceilings.
    No RGB, strobe, blinder, or projector-dimmer channels can be accented.
    """
    out = apply_overlay(baseline, rig, times, config=config)
    fixtures = _patch(rig)
    cfg = OverlayConfig() if config is None else config
    # Validate explicit pulse bounds even on an empty event mapping.
    sample_accent_events(times, (), config=accent_config)
    for name, sequence in _mapping(events, "events").items():
        fixture = _fixture(fixtures, name, _DIMMER_OFFSETS, "events")
        pulse = sample_accent_events(times, sequence, config=accent_config)
        active = pulse > 0
        slot = fixture.address - 1 + _DIMMER_OFFSETS[fixture.kind]
        value = baseline[active, slot].astype(float) * (1 + pulse[active])
        out[active, slot] = _bytes(value, cfg.dimmer_ceiling_byte)
    if not np.array_equal(out[:, ~allowed_slot_mask(rig)], baseline[:, ~allowed_slot_mask(rig)]):
        raise AssertionError("accent changed a forbidden DMX slot")
    return out
