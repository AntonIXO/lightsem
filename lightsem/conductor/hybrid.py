"""Conductor v0b — dance envelopes, audio-phased transients.

Stage 0a measured why this class has to exist. On AIST++ the dance is locked to
the beat *period* but its event *phase* belongs to the choreography, not the
music: footfall-derived light events score at the circular-shift null
(+0.019, CI [-0.006, +0.044]) while an audio beat trigger reaches +0.101. Only
the vertical bounce carries a real beat phase (+0.038, CI [+0.023, +0.054]),
exactly the channel Toiviainen 2010 and Burger 2013 predict.

So the split is:

* **from the dancer** — how much light, how wide, which side, how deep the
  bounce, which metrical level, when the look changes (envelopes and
  amplitudes; block A and the phase-free part of C);
* **from the audio** — when the transients land (beat phase and onsets;
  block B timing).

This is the same two-path architecture PLAN.md section 4 derives from latency
(transients on a 10-20 ms DSP path, learned model on the slow channels). Two
independent arguments, one design.
"""

from __future__ import annotations

import numpy as np

from ..audio.online import AudioBus
from ..dmx.encode import pack_universe, slew_limit
from ..dmx.universe import DmxStream
from ..motion.descriptors import MotionDescriptors
from .v0 import ConductorConfig, ConductorV0, impulse_train, novelty_segments


class ConductorHybrid(ConductorV0):
    """Dance-driven envelopes with the transient layer re-phased to the music."""

    motion_only = False

    def __init__(self, *a, onset_percentile: float = 88.0, **kw):
        super().__init__(*a, **kw)
        self.onset_percentile = onset_percentile

    def audio_events(self, bus: AudioBus, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Beat times and onset-peak times from the audio, in frames."""
        beats = np.asarray(bus.beats)
        onset = np.asarray(bus.channels["onset"])[:n]
        thr = float(np.percentile(onset, self.onset_percentile))
        peaks, last = [], -1e9
        for i in range(1, len(onset) - 1):
            if onset[i] > thr and onset[i] >= onset[i - 1] and onset[i] > onset[i + 1]:
                t = i / bus.fps
                if t - last >= 0.09:
                    peaks.append(t)
                    last = t
        return beats, np.asarray(peaks)

    def _snap_to_grid(self, desc, event_name, grid, amp, T, fps, decay,
                      tol_s: float = 0.25):
        """Predicted events, moved to the nearest audio grid point.

        Existence and strength come from the model's event head; timing comes
        from the music. An event with no grid point within ``tol_s`` keeps its
        own time rather than being dragged onto a beat it has nothing to do with.
        """
        ev = np.asarray(desc.events.get(event_name, []))
        if ev.size == 0 or np.asarray(grid).size == 0:
            return None
        grid = np.asarray(grid, dtype=np.float64)
        snapped = []
        for t in ev:
            k = int(np.argmin(np.abs(grid - t)))
            snapped.append(grid[k] if abs(grid[k] - t) <= tol_s else t)
        st = desc.event_strength.get(event_name)
        if st is None or len(st) != ev.size:
            st = np.ones(ev.size)
        st = np.asarray(st) * np.interp(np.asarray(snapped), desc.times, amp)
        return impulse_train(np.asarray(snapped), st, T, fps, decay)

    def params(self, desc: MotionDescriptors, bus: AudioBus | None = None
               ) -> dict[str, np.ndarray]:
        P = super().params(desc)
        if bus is None:
            return P
        cfg, fps = self.cfg, self.cfg.fps
        T = P["dimmer"].shape[0]
        beats, onsets = self.audio_events(bus, T)
        beats = beats[(beats >= 0) & (beats < T / fps)]

        # Dance decides HOW MUCH; audio decides WHEN.
        foot_rate = self.sq("footfall_rate", desc.channels["footfall_rate"])
        jump_amp = impulse_train(desc.events.get("jump"),
                                 desc.event_strength.get("jump"), T, fps,
                                 cfg.blinder_decay_s * 1.6)
        hand = self.sq("hand_speed", desc.channels["hand_speed"])
        depth = self.sq("vert_bounce_amp", desc.channels["vert_bounce_amp"])

        # Block B decides WHICH beats fire; the audio grid decides WHEN.
        # Before this, every beat got an impulse scaled by a smooth block A rate,
        # which made the light flash uniformly at 6.5 events per second — and a
        # uniform train sits at its own circular-shift null however well each
        # flash is phased. Snapping the predicted events onto the grid keeps the
        # model's selectivity and the music's timing.
        beat_hits = self._snap_to_grid(desc, "footfall", beats, foot_rate, T, fps,
                                       cfg.blinder_decay_s)
        onset_hits = self._snap_to_grid(desc, "hit", onsets, hand, T, fps,
                                        cfg.strobe_decay_s)
        if beat_hits is None:
            beat_hits = impulse_train(beats, np.interp(beats, desc.times, foot_rate),
                                      T, fps, cfg.blinder_decay_s)
        if onset_hits is None:
            onset_hits = impulse_train(onsets, np.interp(onsets, desc.times, hand),
                                       T, fps, cfg.strobe_decay_s)

        e = self.roles["event"]
        if e.size:
            P["dimmer"][:, e] = np.clip(0.8 * beat_hits + 0.9 * jump_amp, 0, 1)[:, None]
        f = self.roles["fast"]
        if f.size:
            P["dimmer"][:, f] = np.clip(onset_hits * (0.35 + 0.65 * hand), 0, 1)[:, None]

        # The beam layer keeps the dancer's bounce DEPTH but takes the music's
        # phase: a light pulse on the beat, as deep as the body is bouncing.
        b = self.roles["beam"]
        if b.size and beats.size >= 2:
            phase = np.zeros(T)
            t = desc.times
            ib = float(np.median(np.diff(beats))) if beats.size > 2 else 0.5
            idx = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, beats.size - 1)
            nxt = np.where(idx + 1 < beats.size, beats[np.clip(idx + 1, 0, beats.size - 1)],
                           beats[idx] + ib)
            phase = np.clip((t - beats[idx]) / np.maximum(nxt - beats[idx], 1e-6), 0, 1)
            pulse = (1.0 - phase) ** 2.0
            master = np.clip(cfg.base_level + 0.88 * (
                0.6 * self.sq("qom", desc.channels["qom"])
                + 0.4 * self.sq("weight", desc.channels["weight"])), 0, 1)
            d_ = cfg.bounce_depth * depth
            lvl = master * (1.0 - d_ + d_ * pulse)
            lateral = self.sq.signed("lateral_pos", desc.channels["lateral_pos"], gain=1.2)
            if cfg.mirror_to_audience:
                lateral = -lateral
            P["dimmer"][:, b] = np.clip(lvl[:, None] * self._spatial(lateral, b), 0, 1)
            P["dimmer"][:, b] = slew_limit(P["dimmer"][:, b], fps,
                                           cfg.slew["head_dimmer"])
        return P

    def render(self, desc: MotionDescriptors, bus: AudioBus | None = None) -> DmxStream:
        P = self.params(desc, bus)
        frames = pack_universe(self.rig, P)
        cues = [{"type": "look_change", "time_s": round(i / self.cfg.fps, 3),
                 "event": "motion_novelty"}
                for i in novelty_segments(desc, self.cfg.novelty_window_s)]
        return DmxStream(frames=frames, fps=self.cfg.fps, times=desc.times.copy(),
                         rig_summary=self.rig.summary(), cues=cues)
