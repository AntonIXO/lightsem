"""Causal Conformer trunk with the three heads of PLAN.md section 4.

Causality is structural, not a convention: attention is masked to the past and
restricted to a finite window, the depthwise convolution is left-padded, and
normalisation never touches the time axis. ``causality_probe`` checks it
mechanically by perturbing the future and asserting the output does not move —
the plan's future-perturbation test, runnable in a unit test rather than only
on stage.

Heads:
  A  block A regression (+ a periodicity output for the metrical level)
  B  event hazard, one logit per event type, trained with BCE
  C  conditional rectified flow over the block C style residual: the head sees
     the trunk state, a noise sample and a flow time, and predicts the velocity
     field, so at inference a style is SAMPLED rather than averaged. This is the
     part that keeps a generative objective in the pipeline; regressing block C
     with MSE is what collapses it to the mean dancer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrunkConfig:
    in_dim: int = 92
    d_model: int = 256
    n_blocks: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    attn_window: int = 160          # 4 s of context at 40 fps
    #: H-A (PLAN.md 6.7): the same trunk with the look-ahead allowed. Attention
    #: sees a symmetric window of the same length and the depthwise convolution
    #: is padded on both sides, so the receptive field composes across the six
    #: blocks in both directions. Nothing else changes; the causal model is
    #: this flag off.
    offline: bool = False
    dropout: float = 0.1
    n_a: int = 13
    n_b: int = 3
    n_c: int = 10
    ar_flow: bool = False


class CausalDepthwiseConv(nn.Module):
    def __init__(self, d: int, k: int, offline: bool = False):
        super().__init__()
        self.pad = k - 1
        self.offline = offline
        self.dw = nn.Conv1d(d, d, k, groups=d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.norm = nn.LayerNorm(d)
        self.act = nn.SiLU()

    def forward(self, x):                      # (B, T, D)
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        if self.offline:
            y = F.pad(y, (self.pad // 2, self.pad - self.pad // 2))
        else:
            y = F.pad(y, (self.pad, 0))
        y = self.act(self.dw(y))
        y = self.pw2(y).transpose(1, 2)
        return x + y


class CausalWindowAttention(nn.Module):
    def __init__(self, d: int, h: int, window: int, dropout: float, offline: bool = False):
        super().__init__()
        self.h, self.dk, self.window = h, d // h, window
        self.offline = offline
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('_window_mask', None, persistent=False)

    def window_mask(self, length, device):
        if self._window_mask is None or self._window_mask.shape[-1] != length or self._window_mask.device != device:
            i = torch.arange(length, device=device)
            if self.offline:
                half = self.window // 2
                mask = (i[None, :] > i[:, None] - half) & (i[None, :] < i[:, None] + half)
            else:
                mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
            self._window_mask = mask[None, None]
        return self._window_mask

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        # past-only, and no further back than the window: a fixed KV budget is
        # what makes the deployed model a constant-latency streaming model.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.window_mask(T, x.device))
        y = y.transpose(1, 2).reshape(B, T, D)
        return x + self.drop(self.out(y))


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(d * mult, d),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return x + 0.5 * self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.ff1 = FeedForward(cfg.d_model, cfg.ff_mult, cfg.dropout)
        self.attn = CausalWindowAttention(cfg.d_model, cfg.n_heads, cfg.attn_window,
                                          cfg.dropout, offline=cfg.offline)
        self.conv = CausalDepthwiseConv(cfg.d_model, cfg.conv_kernel, offline=cfg.offline)
        self.ff2 = FeedForward(cfg.d_model, cfg.ff_mult, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        return self.norm(self.ff2(self.conv(self.attn(self.ff1(x)))))


class FlowHead(nn.Module):
    """Conditional rectified flow over the block C residual."""

    def __init__(self, d_model: int, n_c: int, hidden: int = 256):
        super().__init__()
        self.n_c = n_c
        self.net = nn.Sequential(
            nn.Linear(d_model + n_c + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_c))

    def velocity(self, h, z, t):
        return self.net(torch.cat([h, z, t], dim=-1))

    def loss(self, h, y):
        b, t_, _ = y.shape
        noise = torch.randn_like(y)
        tt = torch.rand(b, t_, 1, device=y.device)
        z = (1 - tt) * noise + tt * y            # rectified-flow interpolant
        target = y - noise                       # constant velocity
        return F.mse_loss(self.velocity(h, z, tt), target)

    @torch.no_grad()
    def sample(self, h, steps: int = 4, generator=None):
        b, t_, _ = h.shape
        z = torch.randn(b, t_, self.n_c, device=h.device, generator=generator)
        for i in range(steps):
            tt = torch.full((b, t_, 1), i / steps, device=h.device)
            z = z + self.velocity(h, z, tt) / steps
        return z


class ARFlowHead(nn.Module):
    """Conditional rectified flow that also conditions on its own last output.

    The v1 head sampled every frame independently given the trunk state, so its
    variety was high-frequency jitter around a smooth mean rather than the
    burst-and-rest structure real dance has. Conditioning the velocity field on
    y_{t-1} — teacher-forced in training, fed back in sampling — makes a sample
    a trajectory instead of a cloud of independent points. It is also the
    natural streaming form: at inference the previous frame is already known.
    """

    def __init__(self, d_model: int, n_c: int, hidden: int = 320,
                 prev_noise: float = 0.35, clamp: float = 6.0):
        super().__init__()
        self.n_c = n_c
        self.prev_noise = prev_noise
        self.clamp = clamp
        self.net = nn.Sequential(
            nn.Linear(d_model + 2 * n_c + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_c))

    def velocity(self, h, z, t, y_prev):
        return self.net(torch.cat([h, z, t, y_prev], dim=-1))

    def loss(self, h, y):
        y_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        # Exposure bias, and it is not academic: teacher-forced training with a
        # clean y_{t-1} and free-running sampling with its own output produced
        # diversity of 1.3e12 — the feedback loop diverged within a clip.
        # Corrupting the conditioning teaches the head to survive its own errors.
        if self.training and self.prev_noise > 0:
            y_prev = y_prev + self.prev_noise * torch.randn_like(y_prev)
        noise = torch.randn_like(y)
        tt = torch.rand(y.shape[0], y.shape[1], 1, device=y.device)
        z = (1 - tt) * noise + tt * y
        return F.mse_loss(self.velocity(h, z, tt, y_prev), y - noise)

    @torch.no_grad()
    def sample(self, h, steps: int = 3, generator=None):
        b, t_, _ = h.shape
        out = torch.zeros(b, t_, self.n_c, device=h.device)
        y_prev = torch.zeros(b, 1, self.n_c, device=h.device)
        for i in range(t_):
            hi = h[:, i:i + 1]
            z = torch.randn(b, 1, self.n_c, device=h.device, generator=generator)
            for s in range(steps):
                tt = torch.full((b, 1, 1), s / steps, device=h.device)
                z = z + self.velocity(hi, z, tt, y_prev) / steps
            # Targets are z-scored by 1.5 x IQR, so |6| is far outside anything
            # the corpus contains; the clamp is a guard rail, not a shaper.
            z = z.clamp(-self.clamp, self.clamp)
            out[:, i:i + 1] = z
            y_prev = z
        return out


class LightTrunk(nn.Module):
    def __init__(self, cfg: TrunkConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Sequential(nn.LayerNorm(cfg.in_dim),
                                 nn.Linear(cfg.in_dim, cfg.d_model), nn.SiLU())
        self.blocks = nn.ModuleList([ConformerBlock(cfg) for _ in range(cfg.n_blocks)])
        self.head_a = nn.Linear(cfg.d_model, cfg.n_a)
        self.head_b = nn.Linear(cfg.d_model, cfg.n_b)
        self.head_c = (ARFlowHead(cfg.d_model, cfg.n_c) if cfg.ar_flow
                       else FlowHead(cfg.d_model, cfg.n_c))

    def trunk(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x):
        h = self.trunk(x)
        return {"h": h, "a": self.head_a(h), "b_logit": self.head_b(h)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def causality_probe(model: nn.Module, x: torch.Tensor, t_cut: int) -> float:
    """Max change at times <= t_cut when everything after t_cut is replaced.

    Must be 0. This is the plan's future-perturbation test: a model that leaks
    the future is not a latency problem, it is a wrong claim, and the only way
    to be sure is to break the future and watch the past not move.
    """
    model.eval()
    y1 = model(x)["a"]
    x2 = x.clone()
    x2[:, t_cut:] = torch.randn_like(x2[:, t_cut:]) * 5.0
    y2 = model(x2)["a"]
    return float((y1[:, :t_cut] - y2[:, :t_cut]).abs().max())
