"""Fixture model and club rig definition.

A fixture owns a contiguous block of DMX channels and knows how to write its
own parameter set into a 512-channel universe. Channel layouts follow common
real-world personalities closely enough to be plausible, but this is a
prototype model, not a GDTF-accurate fixture library.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

UNIVERSE_SIZE = 512


@dataclass
class FixtureState:
    """Human-facing fixture parameters, all normalised 0..1 unless noted."""

    dimmer: float = 0.0
    strobe: float = 0.0          # 0 = open, >0 = flash rate
    hue: float = 0.0             # 0..1
    saturation: float = 1.0
    pan: float = 0.5             # 0..1 -> full pan range
    tilt: float = 0.5
    zoom: float = 0.5            # beam angle: 0 = narrow, 1 = wide
    focus: float = 0.5
    gobo: int = 0                # index
    value: float = 1.0           # HSV value multiplier

    def rgb(self) -> tuple[float, float, float]:
        r, g, b = colorsys.hsv_to_rgb(self.hue % 1.0, self.saturation, self.value)
        return r, g, b


@dataclass
class Fixture:
    """One patched fixture: identity, DMX address, geometry, personality."""

    name: str
    kind: str                     # moving_head | wash | blinder | strobe | projector
    address: int                  # 1-based DMX start channel
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)
    pan_range_deg: float = 540.0
    tilt_range_deg: float = 270.0
    beam_min_deg: float = 8.0
    beam_max_deg: float = 45.0
    state: FixtureState = field(default_factory=FixtureState)

    @property
    def n_channels(self) -> int:
        return {
            "moving_head": 12,
            "wash": 7,
            "blinder": 2,
            "strobe": 3,
            "projector": 10,
        }[self.kind]

    def beam_angle_deg(self) -> float:
        return self.beam_min_deg + self.state.zoom * (self.beam_max_deg - self.beam_min_deg)

    def pan_deg(self) -> float:
        return (self.state.pan - 0.5) * self.pan_range_deg

    def tilt_deg(self) -> float:
        return (self.state.tilt - 0.5) * self.tilt_range_deg

    def write(self, universe: np.ndarray) -> None:
        """Write this fixture's state into a uint8 512-channel universe."""
        s = self.state
        r, g, b = s.rgb()
        a = self.address - 1  # to 0-based

        def put(offset: int, value: float) -> None:
            idx = a + offset
            if 0 <= idx < UNIVERSE_SIZE:
                universe[idx] = int(np.clip(value, 0.0, 1.0) * 255)

        if self.kind == "moving_head":
            # pan, pan-fine, tilt, tilt-fine, dimmer, strobe, R, G, B, zoom,
            # focus, gobo
            pan16 = int(np.clip(s.pan, 0, 1) * 65535)
            tilt16 = int(np.clip(s.tilt, 0, 1) * 65535)
            for off, val in (
                (0, pan16 >> 8), (1, pan16 & 0xFF),
                (2, tilt16 >> 8), (3, tilt16 & 0xFF),
            ):
                if 0 <= a + off < UNIVERSE_SIZE:
                    universe[a + off] = val
            put(4, s.dimmer); put(5, s.strobe)
            put(6, r); put(7, g); put(8, b)
            put(9, s.zoom); put(10, s.focus)
            if 0 <= a + 11 < UNIVERSE_SIZE:
                universe[a + 11] = int(np.clip(s.gobo, 0, 255))
        elif self.kind == "wash":
            put(0, s.dimmer); put(1, s.strobe)
            put(2, r); put(3, g); put(4, b)
            put(5, s.zoom); put(6, s.saturation)
        elif self.kind == "blinder":
            put(0, s.dimmer); put(1, s.strobe)
        elif self.kind == "strobe":
            put(0, s.dimmer); put(1, s.strobe); put(2, s.value)
        elif self.kind == "projector":
            put(0, s.dimmer); put(1, s.strobe)
            put(2, r); put(3, g); put(4, b)
            put(5, s.zoom); put(6, s.focus)
            if 0 <= a + 7 < UNIVERSE_SIZE:
                universe[a + 7] = int(np.clip(s.gobo, 0, 255))
            put(8, s.pan); put(9, s.tilt)


@dataclass
class Rig:
    """A patched collection of fixtures — a small club rig."""

    fixtures: list[Fixture]

    def by_kind(self, kind: str) -> list[Fixture]:
        return [f for f in self.fixtures if f.kind == kind]

    def universe(self) -> np.ndarray:
        u = np.zeros(UNIVERSE_SIZE, dtype=np.uint8)
        for f in self.fixtures:
            f.write(u)
        return u

    def summary(self) -> dict:
        return {
            "n_fixtures": len(self.fixtures),
            "by_kind": {
                k: len(self.by_kind(k))
                for k in sorted({f.kind for f in self.fixtures})
            },
            "channels_used": sum(f.n_channels for f in self.fixtures),
            "patch": [
                {"name": f.name, "kind": f.kind, "address": f.address,
                 "n_channels": f.n_channels}
                for f in self.fixtures
            ],
        }


def club_rig() -> Rig:
    """A small club rig: 6 moving heads on truss + 4 washes + 2 blinders +
    strobe + projector.

    Coordinate frame (metres): x right, y up, z toward the audience.
    Stage floor at y=0, truss at y=6, back wall at z=-6.
    """
    fixtures: list[Fixture] = []
    addr = 1

    # 6 moving heads spread along the truss.
    for i, x in enumerate(np.linspace(-5.0, 5.0, 6)):
        fixtures.append(Fixture(
            name=f"mh{i+1}", kind="moving_head", address=addr,
            position=(float(x), 6.0, -1.0), direction=(0.0, -1.0, 0.2),
            beam_min_deg=6.0, beam_max_deg=32.0,
        ))
        addr += 12

    # 4 washes closer in, lighting the stage volume.
    for i, x in enumerate(np.linspace(-3.5, 3.5, 4)):
        fixtures.append(Fixture(
            name=f"wash{i+1}", kind="wash", address=addr,
            position=(float(x), 5.0, 1.5), direction=(0.0, -1.0, -0.35),
            beam_min_deg=25.0, beam_max_deg=60.0,
        ))
        addr += 7

    # 2 audience blinders.
    for i, x in enumerate((-2.5, 2.5)):
        fixtures.append(Fixture(
            name=f"blinder{i+1}", kind="blinder", address=addr,
            position=(float(x), 4.0, 2.5), direction=(0.0, -0.2, 1.0),
        ))
        addr += 2

    # Central strobe.
    fixtures.append(Fixture(
        name="strobe1", kind="strobe", address=addr,
        position=(0.0, 5.5, 0.0), direction=(0.0, -0.3, 1.0),
    ))
    addr += 3

    # Projector aimed at the back wall.
    fixtures.append(Fixture(
        name="proj1", kind="projector", address=addr,
        position=(0.0, 3.0, 4.0), direction=(0.0, 0.0, -1.0),
        beam_min_deg=30.0, beam_max_deg=55.0,
    ))
    addr += 10

    return Rig(fixtures)
