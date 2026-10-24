"""Independent dimmer/brightness head."""

from torch import nn


class BrightnessHead(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 1))

    def forward(self, fixture_tokens):
        return self.head(fixture_tokens).sigmoid()
