"""Hybrid audio/body/fixture model used by the first foundation pilot."""

from dataclasses import dataclass, asdict
import torch
from torch import nn
from .audio_encoder import AudioEncoder, AudioEncoderConfig
from .long_context import LongContextMemory, LongContextConfig, MemoryState
from .fixture_queries import FixtureQueryDecoder
from .motion_head import MotionHead
from .color_field_head import ColorFieldHead
from .brightness_head import BrightnessHead


@dataclass(frozen=True)
class FoundationConfig:
    audio_dim: int = 52
    body_dim: int = 66
    fixture_dim: int = 8
    width: int = 96
    heads: int = 4
    local_window: int = 32
    latent_slots: int = 8
    adapter_kind: str = "none"
    adapter_bottleneck: int = 32
    adapter_heads: int = 4
    adapter_local_window: int = 32


class FoundationAudioLightModel(nn.Module):
    """One body/audio model with separate output heads.

    The body is an input condition from an independently audited native body.
    This class never samples or edits that body.  The motion head is gated by
    body effort and uses only the safety projection for its light angles.
    """

    def __init__(self, config=FoundationConfig()):
        super().__init__()
        self.config = config
        self.audio = AudioEncoder(AudioEncoderConfig(
            config.audio_dim, config.width,
            adapter_kind=config.adapter_kind,
            adapter_bottleneck=config.adapter_bottleneck,
            adapter_heads=config.adapter_heads,
            adapter_local_window=config.adapter_local_window,
        ))
        self.body = nn.Sequential(nn.LayerNorm(config.body_dim), nn.Linear(config.body_dim, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.body_cross = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.context = LongContextMemory(LongContextConfig(config.width, config.heads, config.local_window, config.latent_slots))
        self.fixture = FixtureQueryDecoder(config.fixture_dim, config.width, config.heads, config.latent_slots)
        self.motion = MotionHead(config.width)
        self.color = ColorFieldHead(config.width)
        self.brightness = BrightnessHead(config.width)
        self.scene = nn.Linear(config.width, 6)
        self.event = nn.Linear(config.width, 1)

    def _local_body_audio(self, body, audio):
        """Cross-modal attention with the same bounded window as the stream."""
        pieces = []
        for start in range(0, body.shape[1], self.config.local_window):
            end = min(body.shape[1], start + self.config.local_window)
            q, kv = body[:, start:end], audio[:, start:end]
            piece, _ = self.body_cross(q, kv, kv, need_weights=False)
            pieces.append(piece)
        return torch.cat(pieces, 1) if pieces else body.new_zeros(body.shape)

    def forward(self, audio, body, fixtures, *, timestamps=None, valid=None, state=None,
                duration_seconds=None, return_state=False):
        if audio.shape[:2] != body.shape[:2]:
            raise ValueError("audio and body must share the experiment clock")
        a = self.audio(audio, timestamps=timestamps, valid=valid, duration_seconds=duration_seconds)
        b = self.body(body)
        body_from_audio = self._local_body_audio(b, a)
        tokens, new_state = self.context(a + b + body_from_audio, valid, state, return_state=True)
        latent = new_state.latent
        fixture_tokens = self.fixture(tokens, fixtures, latent)
        # Effort is an input-derived gate, not a beat oscillator.
        body_energy = torch.linalg.vector_norm(torch.diff(body, dim=1, prepend=body[:, :1]), dim=-1)
        motion = self.motion(fixture_tokens, body_energy)
        color = self.color(fixture_tokens)
        brightness = self.brightness(fixture_tokens)
        pooled = tokens.mean(1)
        out = {"scene": self.scene(tokens), "event": self.event(tokens),
               "fixture_tokens": fixture_tokens, "latent": latent,
               "angle": motion["angle"], "velocity": motion["velocity"],
               "acceleration": motion["acceleration"], "jerk": motion["jerk"],
               "oklab": color["oklab"], "rgb": color["rgb"],
               "brightness": brightness, "pooled": pooled,
               "state": new_state}
        return out

    def config_dict(self):
        return {"architecture": "local_attention+GRU_state+Perceiver_latents+fixture_queries",
                **asdict(self.config)}
