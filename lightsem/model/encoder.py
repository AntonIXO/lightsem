"""LightSem macro encoder -- the trained replacement for the rule table.

Maps a causal window of audio embedding to a stage-lighting scene vector, and
to that scene vector at several future offsets.

Every design choice below is justified against THIS task's measurements:

  patch_len 8 / stride 4  -- x is openl3 at 10 Hz, so a patch spans 0.8 s and
      hops 0.4 s. A musical bar at 120-150 BPM (the RPMC-L2 rock/punk/metal
      range) is 1.6-2.0 s, so ~2-2.5 patches per bar: fine enough to place a
      hit inside a bar, coarse enough that 6.4 s of context is 15 tokens
      rather than 64. Patching is what buys the long window cheaply.

  window 64 frames (6.4 s) -- must span at least two bars for a build-up to be
      visible as a trend rather than a level.

  d_model 192, 4 layers   -- 2.36 M params against 94 k training windows from
      99 songs. Larger was not tried first because the fair linear baseline
      scores BELOW the per-song-mean oracle (-0.109 vs 0.000), i.e. this
      signal is weak; capacity is not the binding constraint.

  CAUSAL mask            -- a lighting desk cannot see the future. Non-causal
      attention would inflate the metric and be unshippable. Enforced by an
      upper-triangular mask and verified in smoke_model.py by perturbing the
      future and asserting bit-identical output.

  multi-horizon heads    -- the brief is anticipation: "build-up now, drop in
      4 bars". Heads predict the scene at +0.0/+1.0/+2.0/+4.0 s from the SAME
      latent, forcing the representation to encode where the music is going.
      Horizon 0 alone would permit a purely reactive encoder.

Note on lineage: the FlexMLP/patching decomposition is a standard patched-
transformer arrangement (cf. PatchTST). It is named here for what it does,
not borrowed from any other project's design.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FrameAdapter(nn.Module):
    """Per-frame projection from the 512-d audio embedding to d_model."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPatching(nn.Module):
    """Non-overlapping-stride patching -> token sequence. [B,W,D]->[B,N,D]."""

    def __init__(self, patch_len: int, stride: int, d_in: int, d_model: int):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.proj = nn.Linear(patch_len * d_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,W,D] -> unfold along time
        p = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B,N,D,P]
        b, n, d, pl = p.shape
        p = p.permute(0, 1, 3, 2).reshape(b, n, pl * d)
        return self.proj(p)


def causal_mask(n: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)


class MacroEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 6,
        patch_len: int = 8,
        stride: int = 4,
        n_targets: int = 6,
        horizons: tuple[int, ...] = (0, 10, 20, 40),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.input_norm = nn.LayerNorm(input_dim)
        self.adapter = FrameAdapter(input_dim, hidden_dim, d_model, dropout)
        self.patching = TemporalPatching(patch_len, stride, d_model, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        # one head per horizon, from the SAME final latent
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, n_targets) for _ in self.horizons]
        )
        # scalar "tension" -- how strongly the scene is about to change
        self.tension = nn.Linear(d_model, 1)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """[B,W,512] -> [B,d_model] latent of the LAST patch (causal 'now')."""
        h = self.adapter(self.input_norm(x))
        t = self.patching(h)                       # [B,N,D]
        t = t + self.pos[:, : t.shape[1]]
        t = self.transformer(t, mask=causal_mask(t.shape[1], t.device))
        return self.out_norm(t[:, -1])             # last token only

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.latent(x)
        preds = torch.stack([h(z) for h in self.heads], dim=1)  # [B,H,n_targets]
        return preds, self.tension(z).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def sinusoid_check() -> float:
    """Sanity: patching+attention must not be degenerate. Returns latent std."""
    m = MacroEncoder()
    x = torch.randn(2, 64, 512)
    with torch.no_grad():
        z = m.latent(x)
    return float(z.std())


if __name__ == "__main__":
    m = MacroEncoder()
    print(f"params: {m.n_params():,}")
    x = torch.randn(4, 64, 512)
    p, t = m(x)
    print("preds", tuple(p.shape), "tension", tuple(t.shape))
    print("latent std", sinusoid_check())
    print("patches from W=64:", (64 - 8) // 4 + 1, f"= {math.floor((64-8)/4)+1}")
