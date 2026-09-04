"""Structure → scene mapping, and the deterministic micro-timing layer.

THE ARCHITECTURAL SPLIT (the one point the original project sketch got right)
-----------------------------------------------------------------------------
Two layers, deliberately kept apart:

  MACRO / MESO  — driven by musical STRUCTURE (segment label) and the
      anticipation curve. Slow, semantic, and the place where a learned model
      would eventually live. Chooses the *look*: palette, movement style,
      beam width, intensity envelope.

  MICRO         — driven by the beatgrid. Fast, deterministic, zero-latency
      by construction: strobe flashes, hit accents, beat-synchronous chases.
      This layer must NEVER wait on a model.

Mixing them is how you get a rig that is either sluggish on the beat or
semantically incoherent. Keeping them apart means the slow layer can be
replaced by a learned policy without touching timing correctness.

Per the recorded rule-out: scenes are selected by SEGMENT LABEL, never by
raw waveform amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio.anticipation import AnticipationCurves
from ..audio.features import AudioFeatures
from ..audio.structure import Segment
from .fixtures import Fixture, Rig


@dataclass
class Scene:
    """A named, parameterised look."""

    name: str
    palette: list[tuple[float, float]]   # (hue, saturation) pairs
    base_dimmer: float
    movement: str                        # static | slow_sweep | fan | chase | search
    movement_rate: float                 # Hz
    zoom: float
    strobe_rate: float = 0.0
    blinder: float = 0.0
    projector_gobo: int = 0
    beat_accent: float = 0.0             # micro-layer accent depth 0..1


# Structure → scene table. This is the L3 mapping: musical function, not FFT.
SCENES: dict[str, Scene] = {
    "intro": Scene(
        name="intro", palette=[(0.60, 0.85), (0.68, 0.75)], base_dimmer=0.30,
        movement="slow_sweep", movement_rate=0.05, zoom=0.75,
        beat_accent=0.12, projector_gobo=1,
    ),
    "breakdown": Scene(
        name="breakdown", palette=[(0.55, 0.90), (0.75, 0.70), (0.48, 0.60)],
        base_dimmer=0.35, movement="slow_sweep", movement_rate=0.08, zoom=0.85,
        beat_accent=0.18, projector_gobo=2,
    ),
    "build-up": Scene(
        name="build-up", palette=[(0.08, 0.95), (0.02, 0.90)], base_dimmer=0.55,
        movement="search", movement_rate=0.45, zoom=0.35,
        beat_accent=0.45, projector_gobo=3,
    ),
    "drop": Scene(
        name="drop", palette=[(0.00, 1.00), (0.14, 0.95), (0.85, 0.90), (0.55, 1.0)],
        base_dimmer=1.00, movement="fan", movement_rate=0.9, zoom=0.15,
        strobe_rate=0.55, blinder=0.85, beat_accent=0.95, projector_gobo=4,
    ),
    "outro": Scene(
        name="outro", palette=[(0.62, 0.70), (0.70, 0.55)], base_dimmer=0.25,
        movement="static", movement_rate=0.0, zoom=0.9,
        beat_accent=0.08, projector_gobo=0,
    ),
}


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class SceneEngine:
    """Turns (time, structure, anticipation, beatgrid) into fixture states."""

    rig: Rig
    feats: AudioFeatures
    segments: list[Segment]
    antic: AnticipationCurves
    transition_s: float = 0.6
    cue_list: list[dict] = field(default_factory=list)

    def segment_at(self, t: float) -> Segment:
        for s in self.segments:
            if s.start_s <= t < s.end_s:
                return s
        return self.segments[-1] if self.segments else Segment(0, 0, "intro", 0, 0, 0, 0)

    def _blend_weight(self, t: float, seg: Segment) -> float:
        """0→1 ramp across a segment boundary so looks cross-fade."""
        return _smoothstep((t - seg.start_s) / max(self.transition_s, 1e-6))

    def beat_phase(self, t: float) -> tuple[float, int]:
        """Phase within the current beat (0..1) and the beat index."""
        beats = self.feats.beats
        if beats.size == 0:
            return 0.0, 0
        i = int(np.searchsorted(beats, t, side="right") - 1)
        if i < 0:
            return 0.0, 0
        nxt = beats[i + 1] if i + 1 < beats.size else beats[i] + (60.0 / max(self.feats.tempo, 1e-6))
        span = max(nxt - beats[i], 1e-6)
        return float(np.clip((t - beats[i]) / span, 0.0, 1.0)), i

    def is_downbeat(self, t: float, tol: float = 0.06) -> bool:
        db = self.feats.downbeats
        if db.size == 0:
            return False
        return bool(np.min(np.abs(db - t)) <= tol)

    def apply(self, t: float) -> None:
        """Compute every fixture's state at time ``t`` seconds."""
        seg = self.segment_at(t)
        scene = SCENES.get(seg.label, SCENES["breakdown"])
        prev_idx = max(seg.index - 1, 0)
        prev_seg = self.segments[prev_idx] if self.segments else seg
        prev_scene = SCENES.get(prev_seg.label, scene)
        w = self._blend_weight(t, seg)

        # --- MACRO: anticipation modulates the look BEFORE the event lands.
        tension = float(np.interp(t, self.antic.frame_times, self.antic.tension))
        cue = bool(np.interp(t, self.antic.frame_times,
                             self.antic.cue_gate.astype(float)) > 0.5)

        # --- MICRO: beat-deterministic accent (never model-gated).
        phase, beat_i = self.beat_phase(t)
        accent = (1.0 - phase) ** 2.5          # sharp decay after each beat
        downbeat_boost = 1.35 if self.is_downbeat(t) else 1.0

        base_dim = prev_scene.base_dimmer + w * (scene.base_dimmer - prev_scene.base_dimmer)
        zoom = prev_scene.zoom + w * (scene.zoom - prev_scene.zoom)
        accent_depth = prev_scene.beat_accent + w * (scene.beat_accent - prev_scene.beat_accent)

        # Anticipation raises intensity and narrows beams ahead of a drop.
        # Clamp the LOWER bound: zoom 0.0 pins a moving head to its minimum
        # beam angle (6 deg), which is a near-invisible pencil in haze and
        # made every rendered frame look identical. Keep beams renderable.
        antic_gain = 1.0 + 0.35 * tension + (0.25 if cue else 0.0)
        zoom = float(np.clip(zoom - 0.25 * tension, 0.18, 1.0))

        heads = self.rig.by_kind("moving_head")
        for k, f in enumerate(heads):
            pal = scene.palette[k % len(scene.palette)]
            f.state.hue, f.state.saturation = pal
            dim = base_dim * (1.0 - accent_depth + accent_depth * accent * downbeat_boost)
            f.state.dimmer = float(np.clip(dim * antic_gain, 0.0, 1.0))
            f.state.zoom = zoom
            f.state.value = 1.0
            f.state.gobo = scene.projector_gobo
            self._move(f, k, len(heads), t, scene, tension)
            f.state.strobe = scene.strobe_rate if scene.name == "drop" else 0.0

        washes = self.rig.by_kind("wash")
        for k, f in enumerate(washes):
            pal = scene.palette[(k + 1) % len(scene.palette)]
            f.state.hue, f.state.saturation = pal
            wash_dim = base_dim * 0.75 * (1.0 - 0.4 * accent_depth
                                          + 0.4 * accent_depth * accent)
            f.state.dimmer = float(np.clip(wash_dim * antic_gain, 0.0, 1.0))
            f.state.zoom = float(np.clip(zoom + 0.3, 0.0, 1.0))

        # Blinders: drop impact + anticipation flash.
        for f in self.rig.by_kind("blinder"):
            hit = scene.blinder * (accent ** 2) * downbeat_boost
            f.state.dimmer = float(np.clip(hit + (0.25 * tension if cue else 0.0), 0.0, 1.0))

        for f in self.rig.by_kind("strobe"):
            f.state.strobe = scene.strobe_rate
            f.state.dimmer = float(np.clip(scene.strobe_rate * accent, 0.0, 1.0))
            f.state.value = 1.0

        for f in self.rig.by_kind("projector"):
            pal = scene.palette[0]
            f.state.hue, f.state.saturation = pal
            f.state.dimmer = float(np.clip(0.4 + 0.6 * base_dim, 0.0, 1.0))
            f.state.gobo = scene.projector_gobo
            f.state.zoom = 0.5 + 0.3 * tension
            f.state.pan = 0.5 + 0.05 * np.sin(2 * np.pi * 0.05 * t)
            f.state.tilt = 0.5

    def _move(self, f: Fixture, k: int, n: int, t: float, scene: Scene,
              tension: float) -> None:
        """Movement styles. Pan/tilt are 0..1 normalised fixture space."""
        rate = scene.movement_rate * (1.0 + 1.5 * tension)
        ph = 2 * np.pi * rate * t + (k / max(n, 1)) * 2 * np.pi
        if scene.movement == "static":
            f.state.pan, f.state.tilt = 0.5, 0.45
        elif scene.movement == "slow_sweep":
            f.state.pan = 0.5 + 0.18 * np.sin(ph)
            f.state.tilt = 0.45 + 0.06 * np.sin(0.5 * ph)
        elif scene.movement == "search":
            f.state.pan = 0.5 + 0.30 * np.sin(ph)
            f.state.tilt = 0.40 + 0.15 * np.abs(np.sin(1.7 * ph))
        elif scene.movement == "fan":
            spread = 0.42
            base = -spread / 2 + spread * (k / max(n - 1, 1))
            f.state.pan = float(np.clip(0.5 + base + 0.10 * np.sin(ph), 0.0, 1.0))
            f.state.tilt = 0.38 + 0.10 * np.sin(2 * ph)
        elif scene.movement == "chase":
            active = int((t * rate * n) % n)
            f.state.pan = 0.5 + 0.25 * np.sin(ph)
            f.state.tilt = 0.42
            if k != active:
                f.state.dimmer *= 0.25
        f.state.pan = float(np.clip(f.state.pan, 0.0, 1.0))
        f.state.tilt = float(np.clip(f.state.tilt, 0.0, 1.0))
