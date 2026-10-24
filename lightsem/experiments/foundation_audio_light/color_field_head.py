"""Continuous fixture color field in OKLab coordinates."""

import torch
from torch import nn


class ColorFieldHead(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.field = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, fixture_tokens):
        # L in [0,1], a/b in a bounded chroma box.  RGB conversion remains a
        # renderer concern; keeping OKLab here avoids pretending RGB loss is
        # perceptual supervision.
        raw = self.field(fixture_tokens)
        lab = torch.cat((raw[..., :1].sigmoid(), raw[..., 1:].tanh() * 0.4), -1)
        return {"oklab": lab, "rgb": lab[..., [0, 1, 2]].clamp(0, 1)}
