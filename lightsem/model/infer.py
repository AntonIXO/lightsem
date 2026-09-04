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

from ..audio.online import AudioBus
from ..conductor.scalers import load_scalers
from ..motion.descriptors import MotionDescriptors
from .light_data import (AUDIO_SCALARS, BLOCK_A, BLOCK_A_V2, BLOCK_B, BLOCK_C,
                         BLOCK_C_V2)
from .light_trunk import LightTrunk, TrunkConfig

DEFAULT_CKPT = Path("/mnt/lightsem-data/ckpt/light_trunk_v1.pt")


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

        fps = bus.fps
        ch = {}
        for i, name in enumerate(self.block_a):
            ch[name] = self._unscale(name, A[:, i])
        for i, name in enumerate(self.block_c):
            ch[name] = self._unscale(name, C[:, i])
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
            thr = max(event_threshold, float(np.percentile(p, 97)))
            t, last = [], -1e9
            for k in range(1, T - 1):
                if p[k] > thr and p[k] >= p[k - 1] and p[k] > p[k + 1]:
                    tt = k / fps
                    if tt - last >= 0.10:
                        t.append(tt)
                        last = tt
            events[name] = np.asarray(t)
            strengths[name] = np.interp(events[name], np.arange(T) / fps, p)

        return MotionDescriptors(
            fps=fps, times=np.arange(T) / fps, channels=ch,
            events=events, event_strength=strengths,
            meta={"source": "model", "n_frames": int(T),
                  "hazard_rate": {k: float(B[:, i].mean()) for i, k in enumerate(BLOCK_B)}})


def render_audio_to_dmx(bus: AudioBus, ckpt: str | Path = DEFAULT_CKPT,
                        seed: int | None = 0, hybrid: bool = True):
    """The one-call product path: AudioBus -> DmxStream."""
    from ..conductor.hybrid import ConductorHybrid
    from ..conductor.v0 import ConductorV0
    lm = LightModel(ckpt)
    d = lm.descriptors(bus, seed=seed)
    cond = ConductorHybrid() if hybrid else ConductorV0()
    return (cond.render(d, bus) if hybrid else cond.render(d)), d
