"""Fixture motion head with explicit body gating and bounded dynamics."""

import torch
from torch import nn
from .safety_projection import bounded_dynamics_projection, SafetyConfig


class MotionHead(nn.Module):
    def __init__(self, width=96, safety=SafetyConfig()):
        super().__init__()
        self.proposal = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.safety = safety

    def forward(self, fixture_tokens, body_energy=None, previous=None):
        jerk = self.proposal(fixture_tokens)
        if body_energy is not None:
            gate = (body_energy / 0.15).clamp(0, 1)[..., None, None]
            jerk = jerk * gate
        (angle, velocity, acceleration, jerk), state = bounded_dynamics_projection(jerk, previous, self.safety)
        return {"angle": angle, "velocity": velocity, "acceleration": acceleration, "jerk": jerk, "state": state}
