"""Permutation-equivariant arbitrary-N fixture decoder."""

import torch
from torch import nn


class FixtureQueryDecoder(nn.Module):
    """Decode a shared latent into any fixture set without fixture ordering.

    Fixture metadata is embedded independently for each fixture.  Cross
    attention sees only the shared latent slots, so permuting metadata permutes
    outputs exactly (up to floating-point ordering), with no positional index.
    """

    def __init__(self, fixture_dim, width=96, heads=4, latent_slots=8):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.fixture = nn.Sequential(nn.LayerNorm(fixture_dim), nn.Linear(fixture_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.latent_slots = latent_slots

    def forward(self, temporal, fixture_metadata, latent=None):
        if temporal.ndim != 3:
            raise ValueError("temporal must be B,T,D")
        if fixture_metadata.ndim != 3:
            raise ValueError("fixture_metadata must be B,N,F")
        b, t, d = temporal.shape
        if fixture_metadata.shape[0] != b:
            raise ValueError("batch mismatch")
        q = self.fixture(fixture_metadata)
        if latent is None:
            latent = temporal.mean(1, keepdim=True)
        if latent.ndim != 3 or latent.shape[0] != b or latent.shape[-1] != d:
            raise ValueError("latent must be B,L,D")
        fixture_context, _ = self.cross(q, latent, latent, need_weights=False)
        # A shared temporal token is broadcast to all fixtures; the only
        # fixture-specific term is q/context, hence permutation equivariance.
        return self.norm(fixture_context[:, None] + temporal[:, :, None] + q[:, None])
