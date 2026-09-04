"""H-I, the mixture of light: a group-effects decoder plus a deterministic effects engine.

PLAN.md 9.3 / 9.9. The decoder does not emit fixtures. It emits, per fixture ROLE
(wash, beam, strobe, blinder, back) and per frame on the 10 Hz operator grid, a small
set of programmer-level parameters -- intensity, palette (hue, saturation), effect id,
rate in beats, phase spread, participation density -- plus one look-change hazard.
Type enters as an output MASK (a strobe has no hue, a blinder is white), never as an
expert. Two consumers read the same parameters:

  * ``render_scene``: a differentiable renderer that turns the group parameters into the
    6-d scene vector every corpus in this project is scored on (hue cos/sin, saturation,
    value, value p90, hue concentration) under a RANDOM RIG (per-role fixture weights
    jittered and dropped), so the loss lives on a rig-invariant observable and two washes
    swapping places costs nothing. Absolute value is never regressed: the value channels
    are gain-normalised inside the training window on both sides.
  * ``GroupEngine``: the deterministic effects engine that binds roles to whatever rig is
    patched, realises the effects at 40 fps on the real beat grid (pulse, chase, strobe,
    static), keeps the H-B one-pole per channel, and packs a DMX universe. This is the
    hand-written conductor with its if-then table removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conductor.rig import canonical_rig, role_index, stage_x
from ..dmx.encode import pack_universe, slew_limit
from ..dmx.universe import DmxStream
from .light_trunk import ConformerBlock, FlowHead, TrunkConfig

ROLES = ("wash", "beam", "strobe", "blinder", "back")
KIND_OF_ROLE = {"wash": "wash", "beam": "moving_head", "strobe": "strobe", "blinder": "blinder", "back": "projector"}
ROLE_OF_KIND = {v: k for k, v in KIND_OF_ROLE.items()}
EFFECTS = ("static", "pulse", "chase", "strobe")
RATES = (0.25, 0.5, 1.0)           # beats per cycle
#: which parameters a role owns (the type mask). 1 = free, 0 = fixed by type.
TYPE_MASK = {
    #          hue  sat  static pulse chase strobe
    "wash":    (1,   1,   1,     1,    1,    0),
    "beam":    (1,   1,   1,     1,    1,    0),
    "strobe":  (0,   0,   0,     1,    0,    1),
    "blinder": (0,   0,   1,     1,    0,    0),
    "back":    (1,   1,   1,     1,    0,    0),
}
#: canonical fixture counts per role (the default rig weights of the renderer)
ROLE_COUNT = {"wash": 4, "beam": 6, "strobe": 1, "blinder": 2, "back": 1}
PER_ROLE = 1 + 2 + 1 + len(EFFECTS) + len(RATES) + 1 + 1     # inten, hue(2), sat, effect, rate, spread, density
OUT_DIM = len(ROLES) * PER_ROLE + 1                           # + look-change logit


@dataclass
class DecoderConfig:
    in_dim: int = 52
    cond_dim: int = 0            # frozen operator-head state appended to the input (0 = none)
    n_corpora: int = 3
    d_model: int = 192
    n_blocks: int = 3
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 9
    attn_window: int = 80        # 8 s at 10 Hz
    dropout: float = 0.1
    offline: bool = False
    #: v2 (PLAN.md 9.9, "flow only if the operator vector is multimodal after conditioning"): the
    #: regressed parameters collapsed to per-song constants (the conditional mean), so the LOOK
    #: (the 6-d scene vector) is sampled by a rectified flow and assigned to the groups by
    #: ``assign_look``; the categorical effect / rate / spread / density heads stay deterministic.
    flow: bool = False


class GroupDecoder(nn.Module):
    """Causal Conformer on the 10 Hz operator features -> group parameters per frame."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        tc = TrunkConfig(in_dim=cfg.in_dim + cfg.cond_dim, d_model=cfg.d_model, n_blocks=cfg.n_blocks,
                         n_heads=cfg.n_heads, ff_mult=cfg.ff_mult, conv_kernel=cfg.conv_kernel,
                         attn_window=cfg.attn_window, dropout=cfg.dropout, offline=cfg.offline)
        self.inp = nn.Sequential(nn.LayerNorm(tc.in_dim), nn.Linear(tc.in_dim, cfg.d_model), nn.SiLU())
        self.corpus = nn.Embedding(cfg.n_corpora, cfg.d_model)
        self.blocks = nn.ModuleList([ConformerBlock(tc) for _ in range(cfg.n_blocks)])
        self.out = nn.Linear(cfg.d_model, OUT_DIM)
        self.look_flow = FlowHead(cfg.d_model, 6) if cfg.flow else None
        mask = torch.ones(len(ROLES), PER_ROLE)
        for g, r in enumerate(ROLES):
            hue, sat, st, pu, ch, sb = TYPE_MASK[r]
            mask[g, 1:3] = hue; mask[g, 3] = sat
            mask[g, 4:8] = torch.tensor([st, pu, ch, sb], dtype=torch.float32)
        self.register_buffer("type_mask", mask)

    def forward(self, x: torch.Tensor, corpus: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.inp(x) + self.corpus(corpus)[:, None, :]
        for b in self.blocks:
            h = b(h)
        o = self.out(h)                                              # (B, T, OUT_DIM)
        out = self.split(o); out["h"] = h
        return out

    @torch.no_grad()
    def sample_look(self, h: torch.Tensor, steps: int = 4, generator=None) -> torch.Tensor:
        """(B, T, 6) sampled look in window-normalised units (value dims z-scored)."""
        return self.look_flow.sample(h, steps=steps, generator=generator)

    def split(self, o: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, _ = o.shape
        g = o[..., :-1].reshape(B, T, len(ROLES), PER_ROLE)
        m = self.type_mask
        inten = torch.sigmoid(g[..., 0])
        hue = g[..., 1:3] * m[:, 1:3]
        hue = hue / (hue.norm(dim=-1, keepdim=True) + 1e-6)
        sat = torch.sigmoid(g[..., 3]) * m[:, 3]
        eff_logit = g[..., 4:8].masked_fill(m[:, 4:8] == 0, -1e4)
        effect = torch.softmax(eff_logit, -1)
        rate = torch.softmax(g[..., 8:11], -1)
        spread = torch.sigmoid(g[..., 11])
        density = torch.sigmoid(g[..., 12])
        return {"inten": inten, "hue": hue, "sat": sat, "effect": effect, "rate": rate,
                "spread": spread, "density": density, "look_logit": o[..., -1]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------- renderer
def effect_modulation(p: dict[str, torch.Tensor], phase: torch.Tensor) -> torch.Tensor:
    """Group-MEAN intensity multiplier of each effect, given the beat phase (B, T) in [0, 1).

    A chase lights a `density` fraction of the group at any moment and a strobe is half
    duty, so on the scene vector (a mean over the group) they are their duty cycles; a
    pulse is the beat-locked envelope at the chosen rate; static is 1."""
    ph = phase[:, :, None]                                           # (B, T, 1)
    pulses = torch.stack([(1.0 - torch.frac(ph[..., 0] / r)) ** 2 for r in RATES], -1)   # (B, T, R)
    pulse = (p["rate"] * pulses[:, :, None, :]).sum(-1)              # (B, T, G)
    e = p["effect"]
    return e[..., 0] + e[..., 1] * pulse + e[..., 2] * p["density"] + e[..., 3] * 0.5


def random_rig(B: int, device, jitter: float = 0.5, p_drop: float = 0.15,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """(B, G) fixture weights per role: canonical counts, log-normal jitter, roles dropped."""
    base = torch.tensor([ROLE_COUNT[r] for r in ROLES], dtype=torch.float32, device=device)
    w = base[None] * torch.exp(jitter * torch.randn(B, len(ROLES), device=device, generator=generator))
    keep = (torch.rand(B, len(ROLES), device=device, generator=generator) > p_drop).float()
    keep[:, 0] = 1.0                                                 # a rig always has a wash
    return w * keep


def render_scene(p: dict[str, torch.Tensor], phase: torch.Tensor, rig_w: torch.Tensor,
                 p90_temp: float = 8.0) -> torch.Tensor:
    """Group parameters -> (B, T, 6) scene vector: hue cos, hue sin, sat, value, value p90, hue conc."""
    mod = effect_modulation(p, phase)                                # (B, T, G)
    v = p["inten"] * mod                                             # group-mean emitted value
    w = rig_w[:, None, :] / rig_w.sum(-1)[:, None, None]             # (B, 1, G)
    value = (w * v).sum(-1)
    p90 = (torch.softmax(p90_temp * v + torch.log(w + 1e-9), -1) * v).sum(-1)
    lw = w * v * p["sat"] + 1e-6                                     # coloured light only carries hue
    hc = (lw * p["hue"][..., 0]).sum(-1) / lw.sum(-1)
    hs = (lw * p["hue"][..., 1]).sum(-1) / lw.sum(-1)
    lv = w * v + 1e-6
    sat = (lv * p["sat"]).sum(-1) / lv.sum(-1)
    conc = torch.sqrt(hc ** 2 + hs ** 2 + 1e-9)
    return torch.stack([hc, hs, sat, value, p90, conc], -1)


def window_norm(y: torch.Tensor, dims=(3, 4), eps: float = 1e-3) -> torch.Tensor:
    """Gain-normalise the level dims of a (B, T, 6) scene vector within the window."""
    cols = []
    for d in range(y.shape[-1]):
        c = y[..., d]
        if d in dims:
            c = (c - c.mean(1, keepdim=True)) / (c.std(1, keepdim=True) + eps)
        cols.append(c)
    return torch.stack(cols, -1)


# ----------------------------------------------------------------------------- engine
class GroupEngine:
    """Deterministic effects engine: group parameters at 10 Hz + the 40 fps beat grid -> DMX.

    Binds each role to every fixture of that kind on the patched rig (sorted by stage x),
    fans the effect over the group with the phase spread, and keeps a one-pole on the
    intensity so a 10 Hz decision does not step at 40 fps."""

    def __init__(self, rig=None, fps: float = 40.0, smooth_tau: float = 0.08,
                 pan_sweep: bool = True):
        self.rig = rig or canonical_rig()
        self.fps = fps
        self.roles = role_index(self.rig)
        self.xs = stage_x(self.rig)
        self.smooth_tau = smooth_tau
        self.pan_sweep = pan_sweep

    @staticmethod
    def _hold(v: np.ndarray, t_src: np.ndarray, t_out: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(t_src, t_out, side="right") - 1, 0, len(t_src) - 1)
        return v[idx]

    def render(self, params: dict[str, np.ndarray], t_params: np.ndarray, beats: np.ndarray,
               bar_phase: np.ndarray | None, duration_s: float, gain: float = 1.0) -> DmxStream:
        """params: numpy arrays (T10, G[, ...]) from GroupDecoder.split; beats in seconds."""
        fps = self.fps
        T = int(round(duration_s * fps)); t = np.arange(T) / fps
        nf = len(self.rig.fixtures)
        # beat phase at 40 fps from the real grid, at the three rates
        if beats.size >= 2:
            idx = np.clip(np.searchsorted(beats, t, side="right") - 1, 0, beats.size - 1)
            ib = float(np.median(np.diff(beats)))
            nxt = np.where(idx + 1 < beats.size, beats[np.clip(idx + 1, 0, beats.size - 1)], beats[idx] + ib)
            phase = np.clip((t - beats[idx]) / np.maximum(nxt - beats[idx], 1e-6), 0, 1)
        else:
            phase = np.zeros(T)
        P = {k: np.zeros((T, nf)) for k in ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom", "gobo")}
        P["val"][:] = 1.0; P["zoom"][:] = 0.5; P["tilt"][:] = 0.5; P["pan"][:] = 0.5
        if self.pan_sweep and bar_phase is not None:
            bp = np.asarray(bar_phase, dtype=np.float64)[:T]
            if len(bp) < T:
                bp = np.concatenate([bp, np.mod(bp[-1] + np.arange(1, T - len(bp) + 1) / (2.0 * fps), 1.0)]) if len(bp) else np.zeros(T)
            P["pan"][:] = (0.5 + 0.35 * np.sin(2 * np.pi * bp))[:, None]
        hold = lambda a: self._hold(a, t_params, t)          # noqa: E731
        for g, role in enumerate(ROLES):
            fx = self.roles[{"wash": "big_slow", "beam": "beam", "strobe": "fast", "blinder": "event", "back": "back"}[role]]
            if fx.size == 0:
                continue
            inten = hold(params["inten"][:, g]) * gain
            eff = hold(params["effect"][:, g]); rate = hold(params["rate"][:, g])
            spread = hold(params["spread"][:, g]); dens = hold(params["density"][:, g])
            r_idx = rate.argmax(1); r = np.array(RATES)[r_idx]
            e_idx = eff.argmax(1)
            hue = np.mod(np.arctan2(hold(params["hue"][:, g, 1]), hold(params["hue"][:, g, 0])) / (2 * np.pi), 1.0)
            sat = hold(params["sat"][:, g])
            phr = np.mod(phase / r, 1.0)
            xs = self.xs[fx]
            x01 = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-6) if fx.size > 1 else np.zeros(fx.size)
            # per-fixture phase offset = spread * position in the group
            ph_f = np.mod(phr[:, None] - spread[:, None] * x01[None, :], 1.0)
            pulse = (1.0 - ph_f) ** 2
            chase = (ph_f < np.maximum(dens, 0.05)[:, None]).astype(float)
            strobe_on = (ph_f < 0.5).astype(float)
            mod = np.where((e_idx == 0)[:, None], 1.0,
                  np.where((e_idx == 1)[:, None], pulse,
                  np.where((e_idx == 2)[:, None], chase, strobe_on)))
            dim = np.clip(inten[:, None] * mod, 0, 1)
            P["dimmer"][:, fx] = dim
            P["hue"][:, fx] = hue[:, None]
            P["sat"][:, fx] = sat[:, None]
            if role in ("strobe", "blinder"):
                P["sat"][:, fx] = 0.0
            if role == "strobe":
                P["strobe"][:, fx] = np.where(e_idx == 3, np.clip(1.0 / (r * max(ib if beats.size >= 2 else 0.5, 0.2)) / 20.0, 0, 1), 0.0)[:, None]
            if role == "beam":
                P["zoom"][:, fx] = (0.35 + 0.3 * hold(params["density"][:, g]))[:, None] + 0.12 * ((1.0 - phase) ** 2)[:, None]
                if bar_phase is not None:
                    bp2 = np.asarray(bar_phase, dtype=np.float64)[:T]
                    bp2 = np.concatenate([bp2, np.zeros(T - len(bp2))]) if len(bp2) < T else bp2
                    P["tilt"][:, fx] = (0.5 + 0.15 * np.sin(4 * np.pi * bp2))[:, None] + 0.05 * (x01[None, :] - 0.5)
        # one-pole per channel (the H-B rule) and slew limits as the conductors use them
        if self.smooth_tau > 0:
            a = np.exp(-1.0 / (self.smooth_tau * fps))
            for k in ("dimmer",):
                out = np.empty_like(P[k]); acc = P[k][0].copy()
                for i in range(T):
                    acc = a * acc + (1 - a) * P[k][i]; out[i] = acc
                P[k] = out
        P["hue"] = slew_limit(P["hue"], fps, 0.6, wrap=True)
        P["pan"] = slew_limit(P["pan"], fps, 1.1)
        P["tilt"] = slew_limit(P["tilt"], fps, 1.1)
        P["zoom"] = slew_limit(P["zoom"], fps, 2.0)
        frames = pack_universe(self.rig, P)
        return DmxStream(frames=frames, fps=fps, times=t.copy(), rig_summary=self.rig.summary(),
                         cues=[{"type": "engine", "time_s": 0.0, "event": "group_decoder"}])


def assign_look(look: np.ndarray, p: dict[str, np.ndarray], fps: float = 10.0,
                tau_level: float = 0.3, tau_colour: float = 0.8) -> dict[str, np.ndarray]:
    """SeqLight's split, generate the look then assign: a sampled (T, 6) look -- hue cos, hue sin,
    sat, value z, p90 z, hue concentration -- becomes per-role intensity / hue / sat; the
    categorical effect heads in ``p`` are kept. Per-frame samples are smoothed by a one-pole
    (the H-B rule) before assignment."""
    from ..audio.online import ema
    L = np.stack([ema(look[:, j], fps, tau_colour if j in (0, 1, 2, 5) else tau_level) for j in range(6)], 1)
    hc, hs, sat, vz, pz, conc = [L[:, j] for j in range(6)]
    ang = np.arctan2(hs, hc); conc = np.clip(conc, 0, 1); sat = np.clip(sat, 0, 1)
    q = {k: v.copy() for k, v in p.items()}
    T = len(look)
    for g, role in enumerate(ROLES):
        if role == "wash":
            a = ang; inten = np.clip(0.55 + 0.25 * vz, 0.05, 1.0)
        elif role == "back":
            a = ang; inten = np.clip(0.50 + 0.25 * vz, 0.05, 1.0)
        elif role == "beam":
            a = ang + (1.0 - conc) * np.pi / 2; inten = np.clip(0.45 + 0.30 * pz, 0.02, 1.0)
        elif role == "blinder":
            a = ang; inten = np.clip(0.4 * np.maximum(pz - 1.0, 0.0), 0.0, 1.0)
        else:  # strobe
            a = ang; inten = np.clip(0.5 * np.maximum(pz - 1.5, 0.0), 0.0, 1.0)
        q["inten"][:T, g] = inten
        q["hue"][:T, g, 0], q["hue"][:T, g, 1] = np.cos(a), np.sin(a)
        if role not in ("strobe", "blinder"):
            q["sat"][:T, g] = sat
    return q


def slow_knobs(params: dict[str, np.ndarray], fps: float = 10.0, tau_s: float = 2.0) -> dict[str, np.ndarray]:
    """The engine-only baseline's input: the decoder's parameters with everything faster than
    ``tau_s`` removed, so only section-level intent survives and the effects are the beat
    trigger's (pulse on every beat)."""
    from ..audio.online import ema
    out = {}
    for k, v in params.items():
        if k == "look_logit":
            out[k] = v; continue
        flat = v.reshape(v.shape[0], -1)
        sm = np.stack([ema(flat[:, j], fps, tau_s) for j in range(flat.shape[1])], 1).reshape(v.shape)
        out[k] = sm
    eff = np.zeros_like(out["effect"]); eff[..., 1] = 1.0            # pulse on the beat, everywhere
    out["effect"] = eff
    rate = np.zeros_like(out["rate"]); rate[..., RATES.index(1.0)] = 1.0
    out["rate"] = rate
    out["spread"] = np.zeros_like(out["spread"])
    return out
