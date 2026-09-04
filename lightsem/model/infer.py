"""Sound in, DMX out — the product path, with no dancer anywhere in it.

The trained trunk predicts the motion DESCRIPTORS that the dance corpus taught
it (block A envelopes, block B event hazards, a sampled block C style), and the
conductor turns those into light exactly as it does for a real dancer. Nothing
here generates a pose, a skeleton or a dance; the dancer was a training-time
modality and does not exist at inference. That distinction is the whole design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..audio.online import AudioBus, ema
from ..conductor.scalers import load_scalers
from ..motion.descriptors import MotionDescriptors
from .light_data import (AUDIO_SCALARS, BLOCK_A, BLOCK_A_V2, BLOCK_B, BLOCK_C,
                         BLOCK_C_V2)
from .light_trunk import LightTrunk, TrunkConfig

DEFAULT_CKPT = Path("/mnt/lightsem-data/ckpt/light_trunk_v1.pt")


def _peaks(p: np.ndarray, thr: float, fps: float,
           refractory_s: float = 0.10) -> np.ndarray:
    """Local maxima of a hazard above `thr`, thinned by a refractory period."""
    t, last = [], -1e9
    for k in range(1, len(p) - 1):
        if p[k] > thr and p[k] >= p[k - 1] and p[k] > p[k + 1]:
            tt = k / fps
            if tt - last >= refractory_s:
                t.append(tt)
                last = tt
    return np.asarray(t)


class LightModel:
    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt, map_location=self.device, weights_only=False)
        cfg = TrunkConfig(**blob["cfg"])
        self.model = LightTrunk(cfg).to(self.device).eval()
        self.model.load_state_dict(blob["model"])
        self.mu = torch.as_tensor(blob["mu"], device=self.device)
        self.sd = torch.as_tensor(blob["sd"], device=self.device)
        self.cfg = cfg
        self.scalers = load_scalers()
        # Which block split this checkpoint was trained with. Stored explicitly
        # when present; otherwise inferred from the head width, because a v1 and
        # a v2 checkpoint differ in exactly that.
        names = blob.get("block_names")
        if names:
            self.block_a, self.block_c = tuple(names["a"]), tuple(names["c"])
        elif cfg.n_a == len(BLOCK_A_V2):
            self.block_a, self.block_c = BLOCK_A_V2, BLOCK_C_V2
        else:
            self.block_a, self.block_c = BLOCK_A, BLOCK_C
        # Per-channel hazard levels fitted so the emitted event rate matches the
        # corpus rate. Written by calibrate_event_thresholds(); absent on an
        # uncalibrated checkpoint, which then falls back to the percentile.
        self.event_thr: dict[str, float] = dict(blob.get("event_thr") or {})
        # Per-channel output smoothing, in seconds. The targets are mocap put
        # through a 12 Hz Butterworth and are genuinely smooth; a flow head
        # sampled per frame is not, even conditioned on its own last output.
        # Measured on val_track, the model carries 11x to 452x the above-6-Hz
        # energy of the real channels, and that jitter is what makes the
        # rendered light flash 86 times a clip where real dance flashes 30.
        self.smooth_tau: dict[str, float] = dict(blob.get("smooth_tau") or {})

    def _smooth(self, name: str, v: np.ndarray, fps: float) -> np.ndarray:
        """Trailing one-pole at the fitted time constant, or nothing."""
        tau = self.smooth_tau.get(name)
        return v if not tau else ema(v, fps, float(tau))

    def receptive_field(self) -> int:
        """How far back the trunk can see, in frames."""
        c = self.cfg
        return c.n_blocks * ((c.attn_window - 1) + (c.conv_kernel - 1)) + 1

    def features(self, bus: AudioBus) -> np.ndarray:
        T = len(bus.times)
        mel = bus.mel[:, :T].T
        sca = np.stack([bus.channels[k][:T] for k in AUDIO_SCALARS], 1)
        return np.concatenate([mel, sca], 1).astype(np.float32)

    def _unscale(self, name: str, v: np.ndarray) -> np.ndarray:
        p = self.scalers.get(name, {"median": 0.0, "iqr": 1.0})
        return v * (1.5 * p["iqr"]) + p["median"]

    @torch.no_grad()
    def descriptors(self, bus: AudioBus, event_threshold: float = 0.5,
                    seed: int | None = None, chunk: int = 2400) -> MotionDescriptors:
        """AudioBus -> predicted MotionDescriptors at the control rate."""
        X = self.features(bus)
        T = X.shape[0]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        A_parts, B_parts, C_parts = [], [], []
        # Chunked with overlap. The priming length must be the trunk's FULL
        # receptive field, not one attention window: attention windows compose
        # across layers, so 6 blocks of a 160-frame window plus a 15-tap causal
        # convolution look back 6 x (159 + 14) = 1038 frames, or 26 s at 40 fps.
        # Priming with one window instead produced block A differences of 0.79
        # between chunk sizes — the same audio giving different light depending
        # on how it happened to be sliced.
        prime = self.receptive_field()
        pos = 0
        while pos < T:
            lo = max(pos - prime, 0)
            hi = min(pos + chunk, T)
            xb = torch.from_numpy(X[lo:hi]).to(self.device)[None]
            xb = (xb - self.mu) / self.sd
            out = self.model(xb)
            keep = slice(pos - lo, hi - lo)
            A_parts.append(out["a"][0, keep].float().cpu().numpy())
            B_parts.append(torch.sigmoid(out["b_logit"])[0, keep].float().cpu().numpy())
            C_parts.append(self.model.head_c.sample(out["h"], generator=gen)[0, keep]
                           .float().cpu().numpy())
            pos = hi
        A = np.concatenate(A_parts); B = np.concatenate(B_parts)
        C = np.concatenate(C_parts)
        self._hazard_cache = B          # for calibrate_event_thresholds

        fps = bus.fps
        ch = {}
        for i, name in enumerate(self.block_a):
            ch[name] = self._smooth(name, self._unscale(name, A[:, i]), fps)
        for i, name in enumerate(self.block_c):
            ch[name] = self._smooth(name, self._unscale(name, C[:, i]), fps)
        ch.setdefault("vert_bounce", np.zeros(T))
        ch["beat_period_s"] = bus.channels["beat_period_s"][:T]
        for need in ("twist", "vert_root", "vert_period_conf", "travel_sin",
                     "travel_cos", "lateral_vel", "metric_level", "footfall_rate",
                     "section_energy", "qom", "weight", "head_speed", "spread",
                     "hand_speed", "vert_bounce_amp", "time_eff"):
            ch.setdefault(need, np.zeros(T))

        events, strengths = {}, {}
        for i, name in enumerate(BLOCK_B):
            p = B[:, i]
            # A corpus-fitted level if the checkpoint carries one, otherwise the
            # old per-clip 97th percentile. The percentile floor makes the event
            # rate a constant of the code rather than a thing the model decides:
            # on a clip it has no confidence about it still fires on the top 3 %
            # of frames. That produced 88.9 light events per clip against 46.5
            # for light mapped from the real dance, and an event train denser
            # than the beat grid scores at its own circular-shift null however
            # well each individual flash is placed.
            thr = self.event_thr.get(name)
            if thr is None:
                thr = max(event_threshold, float(np.percentile(p, 97)))
            events[name] = _peaks(p, float(thr), fps)
            strengths[name] = np.interp(events[name], np.arange(T) / fps, p)

        return MotionDescriptors(
            fps=fps, times=np.arange(T) / fps, channels=ch,
            events=events, event_strength=strengths,
            meta={"source": "model", "n_frames": int(T),
                  "hazard_rate": {k: float(B[:, i].mean()) for i, k in enumerate(BLOCK_B)}})


def calibrate_event_thresholds(lm: "LightModel", clips, fps: float = 40.0,
                              lo: float = 0.01, hi: float = 0.99,
                              n_iter: int = 40) -> dict[str, float]:
    """Fit one hazard level per block-B channel to the corpus event rate.

    `clips` is a sequence of (AudioBus, MotionDescriptors) held out from
    training: the bus gives the model its input, the descriptors give the rate
    the real dancer actually produced. Bisection on the threshold, because the
    emitted rate is monotone in it and there are only three numbers to find.

    Matching the rate is not the same as matching the events -- the model can
    still fire on the wrong beats -- but it removes the one failure that no
    amount of accuracy elsewhere can fix, which is a train so dense that being
    right about which beats matter stops being visible in it.
    """
    target, got = {}, {}
    preds, dur_s = [], 0.0
    for bus, d in clips:
        dm = lm.descriptors(bus, seed=0)
        T = int(dm.meta["n_frames"])
        preds.append(lm._hazard_cache)
        dur_s += T / fps
        for name in BLOCK_B:
            target[name] = target.get(name, 0) + int(np.asarray(d.events.get(name, [])).size)
    if not preds or dur_s <= 0:
        raise ValueError("no clips to calibrate on")
    out = {}
    for i, name in enumerate(BLOCK_B):
        want = target.get(name, 0) / dur_s                     # events per second
        a, b = lo, hi
        for _ in range(n_iter):
            m = 0.5 * (a + b)
            rate = sum(_peaks(p[:, i], m, fps).size for p in preds) / dur_s
            if rate > want:
                a = m
            else:
                b = m
        out[name] = float(0.5 * (a + b))
        got[name] = sum(_peaks(p[:, i], out[name], fps).size for p in preds) / dur_s
        print(f"  {name:9s} threshold {out[name]:.4f} -> {got[name]:.3f} ev/s "
              f"(corpus {want:.3f})")
    return out


def hf_fraction(x: np.ndarray, fps: float = 40.0, cut_hz: float = 6.0) -> float:
    """Share of a channel's variance above `cut_hz`. Zero for a constant."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.size < 64 or np.allclose(x, 0):
        return 0.0
    f = np.fft.rfftfreq(x.size, 1.0 / fps)
    p = np.abs(np.fft.rfft(x)) ** 2
    return float(p[f > cut_hz].sum() / max(p.sum(), 1e-12))


def calibrate_output_smoothing(lm: "LightModel", clips, fps: float = 40.0,
                               lo: float = 0.004, hi: float = 1.0,
                               n_iter: int = 30, cut_hz: float = 6.0
                               ) -> dict[str, float]:
    """Fit one trailing time constant per channel to the corpus's own spectrum.

    Not a denoiser and not a taste call: the target for each channel is the
    above-`cut_hz` energy fraction that the REAL descriptors have, measured on
    the same held-out clips. A channel the model already emits as smoothly as
    the corpus gets no smoothing at all, which is what happens to most of the
    regressed block. The filter is a trailing one-pole, so nothing here costs
    causality -- it costs lag, which is why the fitted constants are reported.
    """
    lm.smooth_tau = {}
    preds, reals = [], []
    for bus, d in clips:
        dm = lm.descriptors(bus, seed=0)
        preds.append(dm)
        reals.append(d)
    names = [c for c in list(lm.block_a) + list(lm.block_c)
             if all(c in r.channels for r in reals)]
    out = {}
    for name in names:
        want = float(np.mean([hf_fraction(r.channels[name], fps, cut_hz) for r in reals]))
        series = [p.channels[name] for p in preds]
        raw = float(np.mean([hf_fraction(s, fps, cut_hz) for s in series]))
        if raw <= want:
            print(f"  {name:18s} already at {raw:.4f} <= corpus {want:.4f}, no smoothing")
            continue
        a, b = lo, hi
        for _ in range(n_iter):
            m = 0.5 * (a + b)
            got = float(np.mean([hf_fraction(ema(s, fps, m), fps, cut_hz) for s in series]))
            if got > want:
                a = m
            else:
                b = m
        tau = 0.5 * (a + b)
        got = float(np.mean([hf_fraction(ema(s, fps, tau), fps, cut_hz) for s in series]))
        out[name] = float(tau)
        print(f"  {name:18s} tau {tau*1000:6.1f} ms  HF {raw:.4f} -> {got:.4f} "
              f"(corpus {want:.4f})")
    return out


def render_audio_to_dmx(bus: AudioBus, ckpt: str | Path = DEFAULT_CKPT,
                        seed: int | None = 0, hybrid: bool = True):
    """The one-call product path: AudioBus -> DmxStream."""
    from ..conductor.hybrid import ConductorHybrid
    from ..conductor.v0 import ConductorV0
    lm = LightModel(ckpt)
    d = lm.descriptors(bus, seed=seed)
    cond = ConductorHybrid() if hybrid else ConductorV0()
    return (cond.render(d, bus) if hybrid else cond.render(d)), d
